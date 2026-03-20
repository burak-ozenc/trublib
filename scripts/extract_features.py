"""
extract_features.py
--------------------
Reads the manifest.csv produced by preprocess_tad_dataset.py and extracts
training features at 80ms resolution — one sample per 80ms chunk — exactly
matching what TrumpetScorer.score() does at inference.

Why this matters
~~~~~~~~~~~~~~~~
The previous version pooled mean+std over entire 10-second segments (~937 frames).
Inference pools mean+std over a single 80ms chunk (~7 frames).

Over 7 frames of a sustained note, std features collapse to near-zero because
nothing changes in 80ms.  The model trained on 5s stds never saw these near-zero
values, so it scored sustained notes at 0.0.  This script fixes that mismatch.

Per-chunk aggregation (matches TrumpetScorer.score exactly)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For each 80ms chunk (1920 samples @ 24kHz):

    frames = frame_manager.push(chunk)   # ~7 Frame objects (stateful, continuous)
    fvs    = [extractor.extract(f) for f in frames]
    mat    = np.stack([fv.to_vector() for fv in fvs])   # (n_frames, 53)
    vec    = np.concatenate([mat.mean(axis=0), mat.std(axis=0)])  # (106,)

FrameManager and FeatureExtractor are shared across all chunks within a single WAV
file (continuous state), then reset between files.  This matches real-time inference
where both components are persistent across chunks of the same stream.

File-level train/test split (critical)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Consecutive 80ms chunks from the same WAV file are highly correlated.
A random chunk-level split would leak: train chunks from file X right next to
test chunks from file X, producing artificially inflated test metrics.

We split on the source WAV file path first, then extract chunks only after
the split is decided.  All chunks from a given file land in exactly one split.

Subsampling
~~~~~~~~~~~
A 10s WAV file yields ~125 80ms chunks.  79k WAV files x 125 = ~9.8M chunks --
unmanageable.  We subsample two ways:

  --stride N       Only save every Nth chunk (default 3).
                   Non-saved chunks are still FED through FM+FE to maintain
                   state continuity.  This is important: delta_mfcc and
                   mfcc_variance depend on history.  We cannot skip feeding
                   intermediate chunks without corrupting the rolling state.

  --max-per-file N Stop saving after N chunks per WAV file (default 30).
                   Processing continues only as long as needed.

Typical output: ~300k-600k total samples, depending on dataset size and flags.

Output
~~~~~~
    data/train.npz   -- X_train, y_train, labels_train, feature_names
    data/test.npz    -- X_test,  y_test,  labels_test,  feature_names

Each .npz contains:
    X             : float32 (n_samples, 106) -- mean+std pooled feature vectors
    y             : int8    (n_samples,)     -- 0=non_trumpet, 1=trumpet
    labels        : object  (n_samples,)     -- raw label strings
    feature_names : object  (106,)           -- dim names for SHAP/inspection

Usage
~~~~~
    python extract_features.py \
        --manifest data/processed/manifest.csv \
        --out-dir  data \
        --sr       24000 \
        --workers  4 \
        --stride   3 \
        --max-per-file 30 \
        --test-size 0.15

Requirements
~~~~~~~~~~~~
    pip install numpy scipy tqdm scikit-learn
    trublib must be installed: pip install -e path/to/trublib
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from trublib.feature_extractor import FeatureExtractor
from trublib.frame_manager import FrameManager
from trublib.normalizer import TwoStageNormalizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SAMPLES = 1_920    # 80ms @ 24kHz -- must match TrumpetScorer inference
TRUMPET_LABEL = "trumpet"
FEATURE_DIM   = 53       # FeatureVector.feature_dim
SEGMENT_DIM   = FEATURE_DIM * 2   # mean + std = 106
MIN_FRAMES    = 3        # discard chunks yielding fewer than this many frames


# ---------------------------------------------------------------------------
# Feature names (for SHAP / inspection)
# ---------------------------------------------------------------------------

def build_feature_names() -> list[str]:
    base = (
            ["spectral_centroid", "spectral_flatness", "spectral_flux", "spectral_rolloff"]
            + ["f0_hz", "pitch_salience", "hnr_db", "inharmonicity", "odd_even_ratio"]
            + [f"mfcc_{i}" for i in range(1, 14)]
            + [f"delta_mfcc_{i}" for i in range(1, 14)]
            + [f"mfcc_var_{i}" for i in range(1, 14)]
            + ["lpc_formant_count", "has_f1", "has_f2", "has_f3"]
            + ["rms_db"]
    )
    assert len(base) == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {len(base)}"
    return [f"mean_{n}" for n in base] + [f"std_{n}" for n in base]


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV and return (float32 mono samples, sample_rate)."""
    sr, data = wavfile.read(str(path))

    if data.dtype == np.int16:
        samples = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        samples = data.astype(np.float32) / 2_147_483_648.0
    elif data.dtype == np.float32:
        samples = data
    else:
        samples = data.astype(np.float32)

    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    return samples, sr


# ---------------------------------------------------------------------------
# Per-file feature extraction
# ---------------------------------------------------------------------------

def extract_file_chunks(
        wav_path: Path,
        sr: int,
        stride: int,
        max_per_file: int,
) -> list[np.ndarray]:
    """
    Extract 80ms chunk feature vectors from a single WAV file.

    Processes the file sequentially, maintaining FrameManager and
    FeatureExtractor state continuously (matching real-time inference).
    Saves one 106-dim vector per sampled chunk.

    Parameters
    ----------
    wav_path : Path
    sr : int
        Expected sample rate (must be 24000).
    stride : int
        Save every stride-th chunk.  All chunks are still processed
        to maintain FM+FE state continuity.
    max_per_file : int
        Stop saving after this many chunks per file.

    Returns
    -------
    list of np.ndarray, each shape (106,) float32.
    Empty list if the file cannot be loaded or is too short.
    """
    try:
        samples, file_sr = load_wav_float32(wav_path)
    except Exception as exc:
        log.debug("Could not load %s: %s", wav_path.name, exc)
        return []

    if file_sr != sr:
        log.debug("SR mismatch %s: expected %d, got %d", wav_path.name, sr, file_sr)
        return []

    if len(samples) < CHUNK_SAMPLES:
        log.debug("File too short (%d samples): %s", len(samples), wav_path.name)
        return []

    # One normalizer, FrameManager, and FeatureExtractor per file.
    # State is maintained continuously across all chunks in this file,
    # exactly as in a real-time stream.
    norm = TwoStageNormalizer()
    fm   = FrameManager()
    ex   = FeatureExtractor(sr=sr)

    # TwoStageNormalizer silence floor.  Any chunk at or below this RMS is
    # returned unchanged (no gain applied).  Its rms_db is clamped to exactly
    # this value regardless of actual content, producing a constant feature
    # that is unrelated to whether the source is a trumpet or not.
    # Saving such chunks as "trumpet" (from a trumpet file) creates a
    # spurious training signal: the model learns that rms_db == SILENCE_FLOOR
    # with std_rms_db == 0 => trumpet, causing false positives on silence.
    NEAR_SILENCE_DB: float = -55.0  # skip chunks quieter than this

    saved: list[np.ndarray] = []
    chunk_idx = 0

    for start in range(0, len(samples) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
        chunk = samples[start : start + CHUNK_SAMPLES]

        # Always process through norm+FM+FE to maintain rolling state,
        # even for chunks we will not save.  Without this the MFCC history,
        # prev_mag (spectral flux), and pending buffer state that carry into
        # the next saved chunk would be stale, producing a distribution shift.
        normalised, rms_db = norm.process(chunk)
        frames = fm.push(normalised)
        fvs = [ex.extract(f, rms_db_override=rms_db) for f in frames]

        # Skip near-silent chunks regardless of file label.
        #
        # Two failure modes this catches:
        # 1. rms_db == -60.0  (silence floor)
        #    Normalizer returned samples unchanged; rms_db is a fixed constant.
        #    Saving as "trumpet" (from a trumpet WAV) teaches the model that
        #    silence == trumpet, causing false positives on any silent passage.
        # 2. rms_db in (-60, -55)  (near-silent)
        #    Signal is dominated by background noise, not the instrument.
        #    High flatness + random f0: the opposite of tonal trumpet features,
        #    but still labeled "trumpet" if inside a trumpet WAV.
        #
        # We still FEED these chunks through FM+FE (the lines above) to keep
        # the rolling state continuous.  We just do not save them to the dataset.
        if rms_db < NEAR_SILENCE_DB:
            chunk_idx += 1
            continue

        # Only persist every stride-th chunk
        if chunk_idx % stride == 0:
            if len(fvs) >= MIN_FRAMES:
                mat  = np.stack([fv.to_vector() for fv in fvs], axis=0)  # (n_frames, 53)
                mean = mat.mean(axis=0)                                    # (53,)
                std  = mat.std(axis=0)                                     # (53,)
                vec  = np.concatenate([mean, std]).astype(np.float32)      # (106,)
                saved.append(vec)

                if len(saved) >= max_per_file:
                    break

        chunk_idx += 1

    return saved


# ---------------------------------------------------------------------------
# Worker for multiprocessing
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> tuple[list[np.ndarray], str]:
    """Unpack args and call extract_file_chunks -- picklable top-level function."""
    wav_path_str, label, sr, stride, max_per_file = args
    vecs = extract_file_chunks(Path(wav_path_str), sr, stride, max_per_file)
    return vecs, label


# ---------------------------------------------------------------------------
# Manifest reader
# ---------------------------------------------------------------------------

def read_manifest(manifest_path: Path) -> list[dict]:
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# File-level train/test split
# ---------------------------------------------------------------------------

def file_level_split(
        rows: list[dict],
        test_size: float,
        random_state: int,
) -> tuple[list[dict], list[dict]]:
    """
    Split manifest rows into train and test at the SOURCE FILE level.

    All chunks from the same original audio file always land in the same
    split.  Stratification preserves the trumpet/non-trumpet ratio.
    """
    from sklearn.model_selection import train_test_split

    # Build source_file -> (label, [rows])
    file_to_rows: dict[str, list[dict]] = defaultdict(list)
    file_to_label: dict[str, str] = {}

    for row in rows:
        src = row["source_file"]
        file_to_rows[src].append(row)
        file_to_label[src] = row["label"]

    source_files  = list(file_to_rows.keys())
    source_labels = [file_to_label[f] for f in source_files]

    log.info("Unique source files: %d", len(source_files))

    train_files, test_files = train_test_split(
        source_files,
        test_size=test_size,
        stratify=source_labels,
        random_state=random_state,
    )

    train_rows = [row for f in train_files for row in file_to_rows[f]]
    test_rows  = [row for f in test_files  for row in file_to_rows[f]]

    for split_name, split_rows in [("train", train_rows), ("test", test_rows)]:
        counts = Counter(r["label"] for r in split_rows)
        log.info(
            "  %-6s WAV files: %d  (%s)",
            split_name,
            len(split_rows),
            ", ".join(f"{l}={n}" for l, n in sorted(counts.items())),
        )

    return train_rows, test_rows


# ---------------------------------------------------------------------------
# Extract + assemble one split
# ---------------------------------------------------------------------------

def extract_split(
        rows: list[dict],
        sr: int,
        stride: int,
        max_per_file: int,
        n_workers: int,
        split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run feature extraction over all WAV files in one split."""
    work_items = [
        (row["path"], row["label"], sr, stride, max_per_file)
        for row in rows
    ]

    results: list[tuple[list[np.ndarray], str]] = []
    t0 = time.perf_counter()

    if n_workers == 1:
        for item in tqdm(work_items, desc=f"Extracting {split_name}", unit="file"):
            results.append(_worker(item))
    else:
        with mp.Pool(processes=n_workers) as pool:
            for result in tqdm(
                    pool.imap(_worker, work_items, chunksize=8),
                    total=len(work_items),
                    desc=f"Extracting {split_name}",
                    unit="file",
            ):
                results.append(result)

    elapsed = time.perf_counter() - t0
    log.info(
        "  %s extraction done in %.1fs (%.1f files/s)",
        split_name, elapsed, len(work_items) / elapsed,
                             )

    # Flatten: (file_chunks, label) -> (X, y)
    X_list: list[np.ndarray] = []
    y_list: list[int]        = []
    label_list: list[str]    = []
    skipped_files = 0

    for vecs, label in results:
        if not vecs:
            skipped_files += 1
            continue
        for vec in vecs:
            X_list.append(vec)
            y_list.append(1 if label == TRUMPET_LABEL else 0)
            label_list.append(label)

    if not X_list:
        raise RuntimeError(
            f"No features extracted for {split_name} split. "
            "Check manifest paths and sample rate."
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int8)

    log.info(
        "  %s: %d samples from %d files (%d files had no usable chunks)",
        split_name, len(X), len(work_items) - skipped_files, skipped_files,
                            )

    n_trumpet     = int((y == 1).sum())
    n_non_trumpet = int((y == 0).sum())
    if n_trumpet > 0 and n_non_trumpet > 0:
        ratio = max(n_trumpet, n_non_trumpet) / min(n_trumpet, n_non_trumpet)
        log.info(
            "  %s class balance -- trumpet: %d  |  non_trumpet: %d  |  ratio: %.1f:1",
            split_name, n_trumpet, n_non_trumpet, ratio,
        )

    bad = int(np.sum(~np.isfinite(X)))
    if bad > 0:
        log.warning(
            "  %s: %d non-finite values -- replacing with 0", split_name, bad,
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, np.array(label_list, dtype=object)


# ---------------------------------------------------------------------------
# Feature health report
# ---------------------------------------------------------------------------

def feature_health_report(X: np.ndarray, y: np.ndarray, split_name: str) -> None:
    """
    Print mean/p5/p95 for key std features.

    After the fix, std features computed over ~7 frames will be near-zero
    for sustained notes.  This is CORRECT: the model now trains on what it
    will actually see at inference.  If these values are large (e.g. > 0.1),
    the old long-window pooling may still be in effect somewhere.
    """
    std_feature_indices = {
        "std_rms_db":              52 + 53,  # always 0.0 in old code -- key indicator
        "std_spectral_flux":        2 + 53,
        "std_inharmonicity":        7 + 53,
        "std_hnr_db":               6 + 53,
        "std_spectral_flatness":    1 + 53,
        "std_pitch_salience":       5 + 53,
    }

    log.info("")
    log.info(
        "Feature health -- %s split  "
        "(std features; near-zero is CORRECT for 80ms windows):", split_name
    )
    log.info("  %-28s  %8s  %8s  %8s", "Feature", "mean", "p5", "p95")
    log.info("  " + "-" * 58)

    y_trumpet = y == 1
    has_trumpet = y_trumpet.any()

    for feat_name, idx in std_feature_indices.items():
        col = X[y_trumpet, idx] if has_trumpet else X[:, idx]
        log.info(
            "  %-28s  %8.5f  %8.5f  %8.5f",
            feat_name,
            float(col.mean()),
            float(np.percentile(col, 5)),
            float(np.percentile(col, 95)),
        )

    log.info("")
    log.info(
        "  std_rms_db mean should be ~0.0 (rms_db_override is constant per chunk)."
    )
    log.info(
        "  Other stds should be small for sustained trumpet, larger for speech/noise."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract 80ms-window features for TAD classifier training. "
            "Produces separate train.npz and test.npz with file-level split."
        )
    )
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="Path to manifest.csv from preprocess_tad_dataset.py",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Output directory for train.npz and test.npz",
    )
    parser.add_argument(
        "--sr", type=int, default=24_000,
        help="Expected sample rate of WAV files (default: 24000)",
    )
    parser.add_argument(
        "--stride", type=int, default=3,
        help=(
            "Save every stride-th 80ms chunk per file (default: 3). "
            "Reduces correlation between consecutive training samples. "
            "All intermediate chunks are still processed to maintain FM+FE state."
        ),
    )
    parser.add_argument(
        "--max-per-file", type=int, default=30,
        help=(
            "Maximum saved chunks per WAV file (default: 30). "
            "Prevents long files from dominating the dataset."
        ),
    )
    parser.add_argument(
        "--test-size", type=float, default=0.15,
        help="Fraction of source files held out for test split (default: 0.15)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="RNG seed for train/test split (default: 42)",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, mp.cpu_count() - 4),
        help="Parallel workers for feature extraction (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N manifest rows (quick smoke test)",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        log.error("Manifest not found: %s", args.manifest)
        raise SystemExit(1)

    # --- Load manifest ---
    rows = read_manifest(args.manifest)
    log.info("Loaded manifest: %d WAV files", len(rows))

    if args.limit:
        rows = rows[: args.limit]
        log.info("Limiting to first %d rows (--limit flag)", args.limit)

    class_counts = Counter(r["label"] for r in rows)
    for label, count in sorted(class_counts.items()):
        log.info("  %-22s  %d WAV files", label, count)

    # --- File-level stratified split ---
    log.info("")
    log.info("Performing file-level train/test split (test_size=%.2f)...", args.test_size)
    train_rows, test_rows = file_level_split(rows, args.test_size, args.random_state)

    # --- Extract features ---
    log.info("")
    log.info(
        "Extraction parameters: stride=%d, max_per_file=%d, workers=%d",
        args.stride, args.max_per_file, args.workers,
    )

    X_train, y_train, labels_train = extract_split(
        train_rows, args.sr, args.stride, args.max_per_file, args.workers, "train"
    )
    log.info("")
    X_test, y_test, labels_test = extract_split(
        test_rows, args.sr, args.stride, args.max_per_file, args.workers, "test"
    )

    # --- Validate feature dimensions ---
    assert X_train.shape[1] == SEGMENT_DIM, (
        f"Expected {SEGMENT_DIM} features, got {X_train.shape[1]}"
    )
    assert X_test.shape[1] == SEGMENT_DIM

    # --- Save ---
    feature_names = np.array(build_feature_names(), dtype=object)

    train_path = args.out_dir / "train.npz"
    test_path  = args.out_dir / "test.npz"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("")
    for path, X, y, labels, name in [
        (train_path, X_train, y_train, labels_train, "train"),
        (test_path,  X_test,  y_test,  labels_test,  "test"),
    ]:
        np.savez_compressed(path, X=X, y=y, labels=labels, feature_names=feature_names)
        size_mb = path.stat().st_size / 1e6
        log.info("Saved %-6s -> %s  (shape %s, %.1f MB)", name, path, X.shape, size_mb)

    # --- Feature health report ---
    feature_health_report(X_train, y_train, "train")

    # --- Summary ---
    log.info("")
    log.info("=" * 60)
    log.info("Done.")
    log.info("  Train: %s  ->  %s", X_train.shape, train_path)
    log.info("  Test:  %s  ->  %s", X_test.shape,  test_path)
    log.info("")
    log.info("Next: retrain the classifier")
    log.info("  python train_classifier.py \\")
    log.info("      --train %s \\", train_path)
    log.info("      --test  %s \\", test_path)
    log.info("      --out-dir models --model mlp --skip-cv")
    log.info("=" * 60)


if __name__ == "__main__":
    main()