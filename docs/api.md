# API Reference

---

## TADProcessor

The single entry point for all processing. One instance per audio stream.

```python
from trublib import TADProcessor, TADConfig

tad = TADProcessor(config=TADConfig(), model_path=None)
```

**Not thread-safe.** Do not share across threads or streams.

### `__init__(config, model_path)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `TADConfig \| None` | `None` | Pipeline configuration. `None` uses defaults. |
| `model_path` | `Path \| None` | `None` | Explicit path to a `.onnx` model file. `None` loads the bundled model. |

### `process(chunk) → TADResult`

Process one chunk of raw microphone audio.

| Parameter | Type | Description |
|---|---|---|
| `chunk` | `np.ndarray` | 1-D (mono) or 2-D (stereo) float32 audio at `config.input_sample_rate`. Stereo is mixed to mono. |

Returns a `TADResult`.

### `reset()`

Clear all internal state. Call when the audio stream is interrupted or restarted. Clears the ring buffer, pending frame buffer, MFCC history, and state machine counters.

### Properties

| Property | Type | Description |
|---|---|---|
| `config` | `TADConfig` | The configuration this processor was created with. |
| `state` | `TADState` | Current state machine state. |

---

## TADConfig

All tunable parameters. A plain Python dataclass — construct with keyword arguments.

```python
from trublib import TADConfig

cfg = TADConfig(
    input_sample_rate = 44100,
    threshold         = 0.6,
    onset_chunks      = 3,
    trailing_chunks   = 4,
    lookback_frames   = 3,
    muted_mode        = False,
    target_rms        = 0.1,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `input_sample_rate` | `int` | `44100` | Sample rate of incoming audio. Internally resampled to 24 kHz. |
| `threshold` | `float` | `0.6` | P(trumpet) cutoff in [0, 1]. Higher = fewer false alarms, more missed notes. |
| `onset_chunks` | `int` | `3` | Consecutive chunks above threshold required to enter ACTIVE. 3 × 80 ms = 240 ms onset guard. |
| `trailing_chunks` | `int` | `4` | Consecutive chunks below threshold required to exit ACTIVE. 4 × 80 ms = 320 ms hold-off. |
| `lookback_frames` | `int` | `3` | Depth of retroactive ring buffer flush on ACTIVE entry. |
| `muted_mode` | `bool` | `False` | Downweight LPC formant rejection for Harmon/wah-wah mutes. Set `True` when the player uses a mute that produces speech-like resonances. |
| `target_rms` | `float` | `0.1` | Internal RMS normalisation target for feature extraction. Not exposed in audio output. |

---

## TADResult

Returned by `TADProcessor.process()`. A plain dataclass — read fields directly.

```python
result = tad.process(chunk)

result.masked_audio   # np.ndarray — the audio to send downstream
result.state          # TADState
result.is_trumpet     # bool
result.confidence     # float
result.flush          # list[np.ndarray] | None
```

| Field | Type | Description |
|---|---|---|
| `masked_audio` | `np.ndarray` | Gated audio at original dynamics. Silence when not ACTIVE or TRAILING. Shape matches resampled input chunk (1920 samples at 24 kHz for an 80 ms 24 kHz input). |
| `state` | `TADState` | Current state after processing this chunk. |
| `is_trumpet` | `bool` | `True` only when `state == TADState.ACTIVE`. |
| `confidence` | `float` | Raw P(trumpet) in [0, 1] from the ONNX classifier. |
| `flush` | `list[np.ndarray] \| None` | Retroactive frames emitted on ACTIVE entry only. Each array is a 512-sample raw frame multiplied by its graduated alpha (0.3 or 0.6). `None` at all other times. Send these to downstream before `masked_audio` on the ACTIVE entry chunk. |

### Caller pattern

```python
result = tad.process(chunk)

# On ACTIVE entry: send retroactive note attack frames first
if result.flush:
    for frame in result.flush:
        downstream.send(frame)

# Send gated audio
if result.is_trumpet:
    downstream.send(result.masked_audio)
```

---

## TADState

```python
from trublib import TADState

TADState.SILENT    # output muted, ring buffer accumulating
TADState.ONSET     # evidence accumulating, output still muted
TADState.ACTIVE    # trumpet confirmed, audio passing through
TADState.TRAILING  # confidence dropped, fade-out in progress
```

`TADState` is a `str` enum — values are `"silent"`, `"onset"`, `"active"`, `"trailing"`. Compare with `state.value == "active"` or `state == TADState.ACTIVE`.

---

## TrumpetScorer

Low-level ONNX wrapper. Usually accessed indirectly through `TADProcessor`. Use directly for batch scoring or diagnostics.

```python
from trublib import TrumpetScorer

scorer = TrumpetScorer()                          # bundled model
scorer = TrumpetScorer(model_path=Path("x.onnx")) # custom model
```

### `score(feature_vectors) → float`

Score one 80 ms chunk. Accepts the output of `FeatureExtractor.extract()` for each frame in the chunk.

```python
score = scorer.score(fvs)   # list[FeatureVector], typically 7–8 items
```

Returns P(trumpet) in [0, 1]. Returns 0.0 for empty input.

### `score_raw(feature_matrix) → np.ndarray`

Batch scoring. Accepts a pre-built `(N, 106)` float32 matrix and returns P(trumpet) per row as a float32 array of shape `(N,)`.

```python
X = np.random.randn(100, 106).astype(np.float32)
probs = scorer.score_raw(X)   # shape (100,)
```

### Properties

| Property | Type | Description |
|---|---|---|
| `model_input_dim` | `int` | Expected feature dimensions. 106 for the bundled v1 model. |

---

## FeatureExtractor

Stateful feature extractor. One instance per stream.

```python
from trublib import FeatureExtractor
from trublib.frame_manager import FrameManager

fm = FrameManager()
ex = FeatureExtractor(sr=24_000)

frames = fm.push(chunk)
fvs = [ex.extract(frame, rms_db_override=rms_db) for frame in frames]
```

### `extract(frame, rms_db_override) → FeatureVector`

| Parameter | Type | Description |
|---|---|---|
| `frame` | `Frame` | Output of `FrameManager.push()`. |
| `rms_db_override` | `float \| None` | Pre-normalisation RMS in dBFS from `TwoStageNormalizer.process()`. Replaces the per-frame RMS so the classifier sees actual playing loudness. |

### `reset()`

Clear rolling state (`_prev_mag`, `_mfcc_history`). Call on stream restart.

---

## FrameManager

Sliding-window frame extractor. One instance per stream.

```python
from trublib.frame_manager import FrameManager

fm = FrameManager()
frames = fm.push(chunk)     # list[Frame]
frames = fm.flush()         # drain remaining samples
fm.reset()
```

| Constant | Value | Notes |
|---|---|---|
| `FRAME_SIZE` | 512 | 21.3 ms @ 24 kHz |
| `HOP_SIZE` | 256 | 50% overlap |

### `push(samples) → list[Frame]`

Accept a 1-D float32 mono array. Returns all complete frames that can be emitted. Expects audio already resampled to 24 kHz and mixed to mono.

### `flush() → list[Frame]`

Drain the pending buffer by zero-padding to one complete frame. Returns at most one frame. Call at end-of-stream.

### Properties

| Property | Type | Description |
|---|---|---|
| `pending_samples` | `int` | Samples in the buffer (latency estimate). |
| `frame_index` | `int` | Total frames emitted since construction or last reset. |

---

## TwoStageNormalizer

```python
from trublib.normalizer import TwoStageNormalizer

norm = TwoStageNormalizer(target_rms=0.1, silence_floor_db=-60.0)
normalised, rms_db = norm.process(chunk)
```

### `process(samples) → (np.ndarray, float)`

| Return value | Description |
|---|---|
| `normalised` | RMS-normalised copy of `samples`. Use only for feature extraction. |
| `rms_db` | Pre-normalisation RMS in dBFS (floor-clamped to `silence_floor_db`). Pass to `FeatureExtractor.extract()` as `rms_db_override`. |

Chunks below `silence_floor_db` are returned unchanged with no gain applied. `rms_db` will equal `silence_floor_db` exactly — these chunks should be skipped in training (see the silence guard in `extract_features.py`).

---

## SoftMaskGenerator

```python
from trublib.soft_mask import SoftMaskGenerator

gen = SoftMaskGenerator(sr=24_000)
out = gen.apply(audio, confidence=0.75)
out = gen.apply_with_fade(audio, gain_start=1.0, gain_end=0.0)
```

### `apply(audio, confidence) → np.ndarray`

Apply a uniform gain of `confidence` to `audio`. `confidence` is clipped to [`min_gain`, 1.0]. Output has the same shape and dtype as input.

### `apply_with_fade(audio, gain_start, gain_end) → np.ndarray`

Apply a linearly interpolated gain ramp across the chunk. Used internally during TRAILING to fade out smoothly within a single chunk.

---

## FeatureVector

The output of `FeatureExtractor.extract()`. Access fields directly or call `to_vector()` for the flat array.

```python
fv = ex.extract(frame)

fv.spectral_centroid   # Hz
fv.spectral_flatness   # [0, 1] — Wiener entropy
fv.spectral_flux       # normalised L2 distance from previous frame
fv.spectral_rolloff    # Hz — 85th percentile frequency
fv.f0_hz               # Hz — 0.0 if no pitch detected
fv.pitch_salience      # [0, 1] — normalised autocorrelation peak
fv.hnr_db              # dB, clipped to [-10, 40]
fv.inharmonicity       # [0, 1] — deviation from ideal harmonics
fv.odd_even_ratio      # odd/even harmonic energy ratio
fv.mfcc                # np.ndarray shape (13,)
fv.delta_mfcc          # np.ndarray shape (13,) — causal first-order delta
fv.mfcc_variance       # np.ndarray shape (13,) — variance over rolling 7-frame window
fv.lpc_formant_count   # int 0–3
fv.has_f1_formant      # bool — 300–900 Hz narrow pole
fv.has_f2_formant      # bool — 850–2500 Hz narrow pole
fv.has_f3_formant      # bool — 2000–3500 Hz narrow pole
fv.rms_db              # dBFS, floor -80

v = fv.to_vector()     # np.ndarray shape (53,), float32
```

---

## Frame

A named tuple returned by `FrameManager.push()` and `FrameManager.flush()`.

| Field | Type | Description |
|---|---|---|
| `windowed` | `np.ndarray` | Hann-windowed float32 samples, shape (512,). Pass to `FeatureExtractor.extract()`. |
| `raw` | `np.ndarray` | Un-windowed float32 samples, shape (512,). Stored in the TADStateMachine ring buffer for retroactive flush. |
| `index` | `int` | Monotonically increasing frame counter (0-based, resets on `FrameManager.reset()`). |