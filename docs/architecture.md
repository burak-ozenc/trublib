# Architecture

trublib is a stateful, single-threaded signal processing pipeline. One `TADProcessor` instance lives for the duration of one audio stream. Every 80 ms chunk passes through the same sequence of components in the same order.

---

## Pipeline

```
[raw chunk @ input_sr]
        │
        ▼
 Mono enforcement          stereo (N,2) → mono (N,) mean mix-down
        │
        ▼
 Resampler                 input_sr → 24 kHz  (soxr HQ preferred, resampy fallback)
        │
        ▼
 TwoStageNormalizer
   Stage 1 ─ RMS → target_rms=0.1    removes mic/distance variance for features
   Stage 2 ─ capture rms_db before   preserves loudness as an explicit feature
        │
        │ (normalised used only for feature extraction below)
        │ (original resampled chunk used for audio output)
        ▼
 FrameManager              ring buffer → Hann-windowed 512-sample frames
                           n_fft=512, hop=256, center=False, ~7 frames per chunk
        │
        ▼
 FeatureExtractor          53-dim vector per frame
                           4 families: spectral, harmonic, cepstral, formant/LPC
        │
        ▼
 TrumpetScorer             stack frames → pool mean+std → 106-dim
                           ONNX MLP inference → P(trumpet) ∈ [0, 1]
        │
        ▼
 TADStateMachine           SILENT → ONSET → ACTIVE → TRAILING → SILENT
                           asymmetric hysteresis, retroactive ring buffer flush
        │
        ▼
 SoftMaskGenerator         gain=1.0 (ACTIVE), linear fade (TRAILING), 0 (else)
                           applied to original un-normalised chunk
        │
        ▼
 TADResult
   .masked_audio           gated audio at original dynamics
   .state                  TADState enum
   .is_trumpet             bool
   .confidence             float P(trumpet)
   .flush                  list[np.ndarray] | None
```

---

## Components

### TwoStageNormalizer

Normalises each chunk to a fixed RMS target (`target_rms=0.1`) so that feature extraction is invariant to microphone gain and player distance. The pre-normalisation RMS in dB is captured first and passed separately to `FeatureExtractor` as `rms_db_override` — the classifier sees actual playing loudness, not the normalised level.

**Important:** the normaliser output is used exclusively for feature extraction. Audio passed to the output gate is the original (resampled, un-normalised) chunk, preserving the recording's natural dynamics.

### FrameManager

A sliding-window frame extractor with a pending-sample queue. Accepts variable-length chunks, emits fixed-size `Frame` objects.

| Parameter | Value | Notes |
|---|---|---|
| FRAME\_SIZE | 512 samples | 21.3 ms @ 24 kHz |
| HOP\_SIZE | 256 samples | 50% overlap |
| Window | Hann | Applied to windowed; raw stored separately for ring buffer |
| center | False | Streaming-safe — no lookahead |

A standard 80 ms chunk (1920 samples) yields 7–8 frames at steady state (8 once the pending buffer fills).

### FeatureExtractor

Computes 53 features per frame. Stateful: `_prev_mag` (for spectral flux) and `_mfcc_history` (a 7-frame deque, for delta MFCCs and variance) are maintained across frames within the same stream.

#### Feature families

**Spectral (4):** centroid, flatness (Wiener entropy), flux, rolloff.

`spectral_flatness` is the strongest single-feature separator: trumpet is highly tonal (flatness 0.01–0.10), white noise is flat (0.7–1.0). Measured acoustic separation: 0.000 (trumpet) vs 0.845 (noise).

`spectral_flux` is near-zero for sustained trumpet (spectrum barely changes within 80 ms) and high for chirp/speech (continuously changing vocal tract).

**Harmonic (5):** f0\_hz (autocorrelation, YIN-inspired), pitch salience (normalised autocorrelation peak), HNR (harmonic-to-noise ratio, clipped to [−10, 40] dB), inharmonicity (deviation from ideal f0 multiples), odd/even harmonic ratio.

Trumpet physics enforce near-zero inharmonicity (bore and bell force harmonic alignment) and a balanced odd/even ratio (open conical resonator — both odd and even harmonics present, unlike clarinet which has only odd harmonics from its closed cylindrical bore).

**Cepstral (39):** MFCCs 1–13, delta-MFCCs (causal, near-zero for sustained trumpet, non-zero for speech), MFCC variance over rolling 7-frame window.

Delta-MFCCs are the primary speech separator: a sustained trumpet note has stable timbre, so MFCC coefficients barely change frame-to-frame. Continuous speech has a vocal tract always in motion.

**Formant/LPC (4):** LPC order 12 (Levinson-Durbin), pole extraction, narrow-bandwidth pole detection in F1 (300–900 Hz), F2 (850–2500 Hz), F3 (2000–3500 Hz) ranges.

Trumpet cannot produce vocal-tract formant structure — it has no vocal tract. This is the strongest speech-rejection feature. Edge case: Harmon and wah-wah mutes produce resonances that mimic formants; `TADConfig.muted_mode=True` downweights LPC rejection when HNR and inharmonicity are simultaneously trumpet-positive.

**Energy (1):** rms\_db — pre-normalisation loudness in dBFS, floor-clamped to −80 dB.

#### Classifier input

Each frame produces a 53-dim vector via `FeatureVector.to_vector()` (soft-normalised to [0, 1] ranges). Across all frames in an 80 ms chunk:

```
mean(vec_0, ..., vec_N)   →  53 dims
std(vec_0, ..., vec_N)    →  53 dims
                             ──────
concatenated segment       → 106 dims  →  ONNX input
```

The `std` features encode temporal stability within the chunk. Sustained trumpet has near-zero stds on flux, inharmonicity, and rms\_db (all constant within 80 ms). Speech and noise have higher stds.

**Critical:** training and inference must use the same pooling window. The model was trained on 80 ms chunks. Pooling over longer windows (5 s, full file) produces out-of-distribution std values and the classifier will fail. See [`docs/training.md`](training.md).

### TrumpetScorer

Wraps an ONNX `InferenceSession`. Loaded once at construction via `importlib.resources` (bundled model) or from an explicit path. The model is a scikit-learn `Pipeline` (StandardScaler + MLP 256→128 → softmax) exported via skl2onnx.

`score(feature_vectors)` accepts the `~7` FeatureVector objects from one 80 ms chunk and returns a single `float` P(trumpet).

### TADStateMachine

A four-state FSM with asymmetric hysteresis. Designed to be "hard to start, slow to stop" — reducing the effect of brief transients or model uncertainty.

```
SILENT ──► ONSET ──► ACTIVE ──► TRAILING ──► SILENT
               ▼                     │
             SILENT              ACTIVE  (recovery)
```

| Transition | Condition | Default |
|---|---|---|
| SILENT → ONSET | 1 chunk ≥ threshold | always |
| ONSET → ACTIVE | `onset_chunks` consecutive chunks ≥ threshold | 3 chunks = 240 ms |
| ONSET → SILENT | any chunk below threshold | immediate |
| ACTIVE → TRAILING | 1 chunk below threshold | immediate |
| TRAILING → ACTIVE | any chunk ≥ threshold | recovery, no re-onset |
| TRAILING → SILENT | `trailing_chunks` consecutive chunks below threshold | 4 chunks = 320 ms |

**Retroactive flush:** on ACTIVE entry, the last `lookback_frames` raw audio frames are flushed with graduated gain `[0.0, 0.3, 0.6]` oldest→newest. This lets downstream models (MERT) see the note attack before the classifier confirms. Frames with alpha 0.0 are discarded.

### SoftMaskGenerator

Applies a scalar gain to the original (un-normalised) chunk:

| State | Gain | Behaviour |
|---|---|---|
| ACTIVE | 1.0 | Identity copy — original audio passes through unchanged |
| TRAILING | 1.0 → 0.0 | Linear fade over `trailing_chunks` — prevents clicks at note ends |
| ONSET / SILENT | 0.0 | Silence |

Gain is applied in the time domain (`audio * gain`). This is mathematically identical to the STFT domain for a uniform gain and avoids ISTFT reconstruction artifacts that occur with short chunks under `center=False / boundary=None`.

---

## STFT parameters

| Parameter | Value | Context |
|---|---|---|
| FeatureExtractor FFT size | 512 | 46.9 Hz/bin, used only for feature extraction |
| SoftMaskGenerator n\_fft | 1024 | Kept for future frequency-dependent masking |
| hop\_length | 256 | Shared |
| window | Hann | Shared |
| center | False | Streaming-safe throughout |
| sample\_rate | 24 000 Hz | Moshi's native rate — all internal processing at this rate |

---

## Design decisions

**Why not VAD?** Silero and similar speech VADs classify trumpet as silence or noise. TAD inverts this: detect trumpet, reject everything else.

**Why bore physics, not player skill?** Even a beginner's first note has the acoustic fingerprint of a brass resonating column — cup mouthpiece, tapered bore, and bell flare force a specific harmonic structure regardless of ability.

**Why odd/even ratio for trumpet (not clarinet)?** Trumpet approximates an open conical resonator, supporting both odd and even harmonics. Clarinet is a closed cylinder and supports only odd harmonics. These are frequently confused; the distinction matters for the odd/even feature.

**Why separate normalisation from output?** The normaliser exists purely to stabilise feature extraction — it must not contaminate the audio output. Passing normalised audio downstream amplifies quiet inter-note passages (breath, room tone) to a constant RMS, creating audible noise when the gate is open.

**Why time-domain gain instead of STFT masking?** The current mask is uniform across all frequency bins. `ISTFT(k · STFT(x)) = k · x` — the round-trip adds no information and introduces reconstruction artifacts (truncated ISTFT output with `boundary=None`). The STFT code path is preserved for future frequency-dependent masking.

**Why not thread-safe?** One instance per stream is the correct unit. Making the pipeline thread-safe would require locking around all stateful components (FrameManager pending buffer, FeatureExtractor MFCC history, TADStateMachine counters and ring buffer) with no benefit in the single-stream use case.

---

## Latency budget

Measured on a modern CPU (pure Python/NumPy, no GPU):

| Stage | Mean (ms) | Notes |
|---|---|---|
| TwoStageNormalizer | < 0.1 | Trivial |
| FrameManager | < 0.1 | NumPy concat + slice |
| FeatureExtractor | 7.5 | Hot path — LPC + MFCC |
| TrumpetScorer | 0.8 | ONNX C++ runtime |
| TADStateMachine | < 0.1 | Pure Python |
| SoftMaskGenerator | < 0.1 | `audio * gain` |
| **Total** | **~9.3** | Budget: 80 ms |

p99: 20.5 ms. Worst case observed: 39 ms. Headroom: ~40 ms worst case.

FeatureExtractor dominates at 85% of pipeline time. The bottleneck is `_lpc_formants` (Levinson-Durbin + `np.roots`) and `_compute_mfcc` (mel filterbank matmul). Rust porting would be the next step if tighter latency were needed, but at 9.3 ms mean this is not warranted.