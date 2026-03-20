# Training

This document describes how to build a dataset, extract features, train the classifier, and export it as the ONNX model that ships inside trublib.

---

## Overview

The classifier is a scikit-learn MLP (256→128, relu, adam) wrapped in a StandardScaler pipeline and exported to ONNX via skl2onnx. It receives a 106-dimensional feature vector (mean + std of 53 features over one 80 ms chunk) and outputs P(trumpet).

**The single most important constraint:** training and inference must use the **same pooling window** — one 80 ms chunk = ~7 frames. Training on longer windows (5 s, full file) collapses std features to zero at inference and the model fails silently. This was the root cause of the original train/inference mismatch bug.

---

## Step 1 — Prepare raw audio

Organise your source audio into one folder per class:

```
data/raw/
├── trumpet/          ← trumpet recordings (any format)
├── speech/           ← speech in any language
├── noise/            ← ambient/environmental noise
├── trombone/         ← brass instrument to reject
├── violin/           ← non-brass instrument
└── breathing/        ← player breath sounds
```

All subdirectory names become class labels. The label `trumpet` is the positive class; everything else is `non_trumpet`.

**Current dataset:**

| Class | Source files |
|---|---|
| trumpet | isolated practice recordings, scales, études |
| non\_trumpet | speech (multi-language), street/ambient noise, french horn, trombone, tuba, tenor sax, violin, breathing |

---

## Step 2 — Preprocess (any format → 24 kHz mono WAV)

```bash
python scripts/preprocess_tad_dataset.py \
    --src  data/raw \
    --dst  data/processed \
    --sr   24000 \
    --chunk-sec 10 \
    --min-rms-db -40
```

This converts every source file to 24 kHz mono WAV via ffmpeg, splits at `--chunk-sec` boundaries, drops near-silent chunks below `--min-rms-db`, and writes a `manifest.csv` + `stats.json`.

ffmpeg must be on your PATH.

**Adding a new class without reprocessing everything:**

```bash
python scripts/preprocess_tad_dataset.py \
    --src  data/raw/new_class \
    --dst  data/processed \
    --append
```

---

## Step 3 — Extract features (80 ms chunks)

```bash
python scripts/extract_features.py \
    --manifest data/processed/manifest.csv \
    --out-dir  data \
    --stride   3 \
    --max-per-file 30 \
    --test-size 0.15 \
    --workers  8
```

This produces `data/train.npz` and `data/test.npz`.

**Key parameters:**

| Flag | Default | Effect |
|---|---|---|
| `--stride N` | 3 | Save every Nth 80 ms chunk per file. Reduces correlation between consecutive training samples. All chunks are still processed to maintain FrameManager/FeatureExtractor state continuity. |
| `--max-per-file N` | 30 | Cap saved chunks per WAV file. Prevents long files from dominating the dataset. |
| `--test-size` | 0.15 | Fraction of **source files** held out for test. |

**Why file-level split?** Consecutive 80 ms chunks from the same WAV file are highly correlated — same room, same player, same phrase. A chunk-level random split would leak correlated samples across train/test, inflating test metrics. The split is done at the source file level before any chunk extraction; every chunk from a given file lands in exactly one split.

**Why feed non-saved chunks through FM+FE?** FrameManager's pending buffer and FeatureExtractor's MFCC history must remain continuous across chunk boundaries. Skipping intermediate chunks cold-starts these state machines and corrupts the features of the next saved chunk. The fix is to always feed every chunk through the pipeline, but only persist every Nth chunk's feature vector.

**Silence filtering:** chunks with `rms_db < -55 dBFS` are skipped regardless of label. Without this, silent passages at the start/end of trumpet recordings enter the training set labeled "trumpet" — teaching the model that silence equals trumpet and causing false positives on any quiet passage.

**Expected output size** (79 k WAV files, stride=3, max\_per\_file=30):

| Split | Samples | Size |
|---|---|---|
| train | ~1.3 M | ~480 MB |
| test | ~210 k | ~77 MB |

---

## Step 4 — Train

```bash
python scripts/train_classifier.py \
    --train data/train.npz \
    --test  data/test.npz \
    --out-dir models \
    --model mlp \
    --skip-cv
```

Without `--skip-cv` the script runs 5-fold stratified cross-validation and benchmarks MLP against SVM, then trains the winner on the full training set. `--skip-cv` skips cross-validation and goes straight to training the specified model — use this when retraining after a known good architecture.

**Outputs:**

```
models/
├── trumpet_scorer_v1.onnx    ← deploy this
├── scaler.pkl                ← StandardScaler (reference / debugging)
└── training_report.json      ← metrics, threshold analysis, confusion matrix
```

**v2 training results** (80 ms windows, 1.3 M train samples):

```
              precision    recall  f1-score
non_trumpet      0.9956    0.9974    0.9965
    trumpet      0.9560    0.9288    0.9422

AUC-ROC: 0.9987  |  F1 (trumpet): 0.9422
```

Confusion matrix (208 k test samples):
```
                  predicted_non  predicted_trumpet
actual_non               196051                520   (0.26% false alarm rate)
actual_trumpet               866              11301   (7.12% miss rate)
```

---

## Step 5 — Deploy

Copy the ONNX model into the trublib package and reinstall:

```bash
# Windows
copy models\trumpet_scorer_v1.onnx trublib\trublib\models\trumpet_scorer_v1.onnx

# Linux / macOS
cp models/trumpet_scorer_v1.onnx trublib/trublib/models/trumpet_scorer_v1.onnx

pip install -e ".[dev]"    # refresh importlib.resources package data
```

---

## Step 6 — Validate

```bash
python scripts/diagnose_features.py \
    --file  path/to/trumpet_recording.wav \
    --features data/train.npz
```

This replays the exact TADProcessor inference pipeline chunk-by-chunk and prints a confidence timeline, state machine simulation, and comparison against training distribution statistics.

**Healthy output signs:**
- Silence regions: P(trumpet) = 0.000
- Sustained notes: P(trumpet) > 0.9 on most chunks
- State machine simulation shows one contiguous ACTIVE period covering the playing region
- Training distribution comparison: all key features within 1–2 σ of training trumpet mean

---

## Threshold selection

The `training_report.json` includes a threshold analysis. Default is 0.6.

```
Threshold   Precision   Recall      F1
0.30        0.945       0.941       0.943
0.50        0.956       0.929       0.942
0.60        0.961       0.923       0.942  ← default
0.70        0.967       0.917       0.941
0.90        0.978       0.891       0.932
```

F1 is nearly flat across 0.3–0.7 because the MLP produces well-calibrated probabilities — most samples score near 0 or near 1. The threshold mainly controls the tradeoff between false alarms (lower threshold → more false alarms) and missed notes (higher threshold → more misses), especially for quieter playing at the model's confidence boundary.

For quiet playing conditions (student mic, low-gain setup), lowering to 0.4–0.5 is reasonable.

---

## Dataset quality notes

**Silence contamination:** the most impactful data quality issue. If a trumpet WAV file has silence at the start or end (player not yet playing), those silent chunks get labeled "trumpet" during feature extraction. The silence floor guard in `extract_features.py` (`rms_db < -55`) prevents this, but you should also verify source recordings don't have extended silence.

**Class imbalance:** ~16:1 non\_trumpet:trumpet ratio in the current dataset. `class_weight='balanced'` in the MLP handles this automatically. Do not try to manually balance by removing non-trumpet samples — diversity in the non-trumpet class is what gives the model its generality.

**Muted trumpet:** Harmon and wah-wah mutes produce resonances that partially mimic speech formants. The model handles `muted_mode=True` via downweighted LPC rejection when HNR and inharmonicity are trumpet-positive, but the training data for muted trumpet is thin (~96 segments). If muted playing is important for your use case, add more muted recordings to `data/raw/trumpet/`.

**Mixed recordings:** the model is trained on isolated sources. Trumpet buried in a full band mix (piano, bass, drums) is a different distribution and will not be reliably detected. This is by design — the deployment target is a student practising solo in front of a microphone, not a recording.

---

## Diagnostics

```bash
# Full verbose timeline — every chunk
python scripts/diagnose_features.py --file recording.wav --verbose

# Lower threshold for quiet playing
python scripts/diagnose_features.py --file recording.wav --threshold 0.4

# Compare against training distribution
python scripts/diagnose_features.py --file recording.wav --features data/train.npz
```

```bash
# Latency profiling
python scripts/profile_pipeline.py --mode timing --chunks 200
python scripts/profile_pipeline.py --mode cprofile --chunks 500
```