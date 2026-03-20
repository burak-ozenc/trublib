# trublib

**Trumpet Activity Detection for real-time audio streams.**

trublib strips all non-trumpet audio from a microphone stream in real time, passing only confirmed trumpet audio downstream. It is the signal-routing layer inside [TRUB.AI](https://github.com/your-org/trub-ai), an AI-powered trumpet teaching assistant, sitting between the microphone and the Moshi speech model + MERT audio encoder.

```
[microphone]  →  TADProcessor.process(chunk)  →  TADResult.masked_audio
```

**Latency:** 9.3 ms mean, 39 ms worst case per 80 ms chunk. Well within the 80 ms real-time budget.

---

## Installation

```bash
pip install trublib
```

trublib requires Python ≥ 3.11. The ONNX classifier model (`trumpet_scorer_v1.onnx`) ships inside the package.

**Optional — faster resampling:**

```bash
pip install soxr          # recommended: high-quality, fast
# pip install resampy     # fallback if soxr is unavailable
```

**Optional — ONNX inference:**

```bash
pip install onnxruntime   # required for TrumpetScorer (included transitively via trublib[full])
```

---

## Quickstart

```python
from trublib import TADProcessor, TADConfig
import numpy as np

# One processor per audio stream — not thread-safe
tad = TADProcessor(TADConfig(input_sample_rate=44100, threshold=0.6))

# Your microphone loop — 80ms chunks at whatever sample rate you record at
for chunk in mic_stream:                      # np.ndarray, shape (N,) or (N, 2)
    result = tad.process(chunk)

    # On ACTIVE entry: retroactive attack frames are flushed
    if result.flush:
        for frame in result.flush:
            downstream.send(frame)            # 512-sample raw frames, graduated gain

    # Every chunk: clean gated audio (silence when not playing)
    if result.is_trumpet:
        downstream.send(result.masked_audio)  # original dynamics, no normalisation

    # For UI / diagnostics
    print(result.state, f"P={result.confidence:.3f}")
```

### TADResult fields

| Field | Type | Description |
|---|---|---|
| `masked_audio` | `np.ndarray` | Gated audio at original dynamics (silence when not ACTIVE) |
| `state` | `TADState` | Current state: `SILENT`, `ONSET`, `ACTIVE`, `TRAILING` |
| `is_trumpet` | `bool` | `True` only when state is ACTIVE |
| `confidence` | `float` | Raw P(trumpet) in [0, 1] |
| `flush` | `list[np.ndarray] \| None` | Retroactive attack frames on ACTIVE entry; `None` otherwise |

---

## Configuration

```python
from trublib import TADConfig

cfg = TADConfig(
    input_sample_rate = 44100,   # your mic's sample rate
    threshold         = 0.6,     # P(trumpet) cutoff — raise to reduce false alarms
    onset_chunks      = 3,       # consecutive chunks above threshold to enter ACTIVE
    trailing_chunks   = 4,       # consecutive chunks below threshold to exit ACTIVE
    lookback_frames   = 3,       # retroactive flush depth on ACTIVE entry
    muted_mode        = False,   # set True for Harmon/wah-wah mutes
    target_rms        = 0.1,     # internal normalisation target for feature extraction
)
```

**Threshold guide:**

| Threshold | Use case |
|---|---|
| 0.4–0.5 | Quiet playing, student mic, low gain |
| 0.6 | Default — most practice room conditions |
| 0.7–0.8 | Noisy environment, reduce false alarms |

---

## State machine

```
SILENT → ONSET → ACTIVE → TRAILING → SILENT
              ↘ (confidence drops before onset_chunks)  → SILENT
                              TRAILING → ACTIVE  (confidence recovers)
```

- **SILENT:** output muted. Frames accumulate in the ring buffer silently.
- **ONSET:** first confirmed chunk above threshold. Accumulating evidence over `onset_chunks` — "hard to start".
- **ACTIVE:** trumpet confirmed. `masked_audio` passes through at original level. `flush` emitted once on entry with the note attack frames from the ring buffer.
- **TRAILING:** confidence dropped. Hold-off for `trailing_chunks` before silencing — "slow to stop". Gain fades linearly. Confidence recovery snaps back to ACTIVE without re-entering ONSET.

---

## Custom model

```python
from pathlib import Path
from trublib import TADProcessor, TADConfig

tad = TADProcessor(
    config=TADConfig(),
    model_path=Path("path/to/your_model.onnx"),
)
```

The model must accept `float32` input of shape `(N, 106)` and output class probabilities of shape `(N, 2)` — `[P(non_trumpet), P(trumpet)]`. See [`scripts/train_classifier.py`](scripts/train_classifier.py) for how the bundled model was trained.

---

## Running the demo server

```bash
pip install fastapi uvicorn python-multipart onnxruntime
python -m uvicorn tad_demo.app:app --port 8000
```

Open `http://localhost:8000` — drag-and-drop any audio file (any format, via ffmpeg), adjust threshold and hysteresis sliders, compare original vs processed audio side by side, download the cleaned WAV.

> **Windows / Anaconda:** always use `python -m uvicorn`, never bare `uvicorn`. Install onnxruntime in the same Python environment: `python -m pip install onnxruntime`.

---

## Development

```bash
git clone https://github.com/your-org/trublib
cd trublib
pip install -e ".[dev]"
pytest tests/ -v
```

127 tests, 4 test files covering frame extraction, feature correctness, acoustic separation, state machine transitions, and the full pipeline.

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full pipeline breakdown, feature family descriptions, and design decisions.

For retraining the classifier on your own data, see [`docs/training.md`](docs/training.md).

---

## Licence

**Source code:** MIT

**Bundled model weights:** Non-commercial / research / educational use only.
The model was trained on datasets that carry NonCommercial upstream terms (IRMAS, ESC-50, good-sounds).
See [`docs/data.md`](docs/data.md) for full attribution and [`NOTICE`](NOTICE) for the summary.

To produce commercially-usable weights, retrain using only CC0 / CC BY sources.
See [`docs/training.md`](docs/training.md) for instructions.