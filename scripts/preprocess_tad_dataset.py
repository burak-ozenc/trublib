"""
preprocess_tad_dataset.py
--------------------------
One-time ETL script for preparing TAD (Trumpet Activity Detection) training data.

What it does
~~~~~~~~~~~~
1. Walks a source directory tree (one subdirectory per class label)
2. Converts every audio file to mono WAV @ 24 kHz via ffmpeg
3. Splits long files into fixed-length chunks (default 10 s)
4. Discards near-silent chunks below an RMS floor (default -40 dB)
5. Writes processed chunks to a flat output directory
6. Emits a CSV manifest: output_path, label, source_file, chunk_index, rms_db

Input directory layout expected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    data/raw/
        trumpet/
            session1.wav
            session2.flac
            ...
        trombone/
            ...mp4
            ...wav
        speech/
            ...
        noise/
            ...

Output layout
~~~~~~~~~~~~~
    data/processed/
        trumpet__session1__000.wav
        trumpet__session1__001.wav
        trombone__clip1__000.wav
        ...
    manifest.csv

Usage
~~~~~
    python preprocess_tad_dataset.py \\
        --src  data/raw \\
        --dst  data/processed \\
        --chunk-sec 10 \\
        --min-rms-db -40 \\
        --sr 24000

Requirements
~~~~~~~~~~~~
    pip install numpy scipy tqdm
    ffmpeg must be on PATH (https://ffmpeg.org)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

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

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".mp4", ".m4a", ".ogg", ".aac", ".aiff", ".aif"}
MANIFEST_FILENAME = "manifest.csv"
STATS_FILENAME = "stats.json"


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------


def check_ffmpeg() -> None:
    """Abort early if ffmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        log.error(
            "ffmpeg not found on PATH.  "
            "Install it from https://ffmpeg.org or via your package manager."
        )
        sys.exit(1)


def convert_to_wav(src: Path, dst: Path, sr: int) -> bool:
    """
    Convert *src* (any format) to mono WAV at *sr* Hz, written to *dst*.

    Returns True on success, False if ffmpeg fails (file is skipped).
    """
    cmd = [
        "ffmpeg",
        "-y",               # overwrite without asking
        "-i", str(src),
        "-ac", "1",         # mono
        "-ar", str(sr),     # target sample rate
        "-sample_fmt", "s16",  # 16-bit PCM — scipy can read this directly
        str(dst),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def load_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    """
    Load a WAV file and return (samples: float32, sample_rate).

    Normalises int16 PCM to [-1, 1].  Handles stereo by averaging channels.
    """
    sr, data = wavfile.read(str(path))

    # Normalise dtype to float32
    if data.dtype == np.int16:
        samples = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        samples = data.astype(np.float32) / 2_147_483_648.0
    elif data.dtype == np.float32:
        samples = data
    elif data.dtype == np.float64:
        samples = data.astype(np.float32)
    else:
        samples = data.astype(np.float32)

    # Ensure mono
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    return samples, sr


def compute_rms_db(samples: np.ndarray) -> float:
    """RMS energy in dBFS, floored at -80 dB."""
    rms = float(np.sqrt(np.mean(samples ** 2)))
    return float(20.0 * np.log10(max(rms, 1e-10)))


def split_into_chunks(
    samples: np.ndarray,
    sr: int,
    chunk_sec: float,
    min_chunk_sec: float = 2.0,
) -> list[np.ndarray]:
    """
    Split *samples* into fixed-length chunks of *chunk_sec* seconds.

    The last chunk is kept only if it is at least *min_chunk_sec* long;
    shorter tails are discarded to avoid training on near-empty frames.
    """
    chunk_len = int(sr * chunk_sec)
    min_len = int(sr * min_chunk_sec)

    chunks = []
    start = 0
    while start < len(samples):
        end = start + chunk_len
        chunk = samples[start:end]
        if len(chunk) >= min_len:
            chunks.append(chunk)
        start = end

    return chunks


def save_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    """Save float32 mono samples as 16-bit PCM WAV."""
    # Clip to [-1, 1] before converting to int16
    clipped = np.clip(samples, -1.0, 1.0)
    int16 = (clipped * 32767).astype(np.int16)
    wavfile.write(str(path), sr, int16)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_file(
    src: Path,
    label: str,
    dst_dir: Path,
    sr: int,
    chunk_sec: float,
    min_rms_db: float,
    min_chunk_sec: float,
) -> list[dict]:
    """
    Process a single source file: convert → split → RMS filter → save.

    Returns a list of manifest row dicts for each chunk written.
    """
    rows = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_wav = Path(tmp) / "converted.wav"

        # Step 1: convert to target format
        if not convert_to_wav(src, tmp_wav, sr):
            log.warning("  ffmpeg failed on %s — skipping", src.name)
            return rows

        # Step 2: load
        try:
            samples, actual_sr = load_wav_float32(tmp_wav)
        except Exception as exc:
            log.warning("  Could not read %s: %s — skipping", src.name, exc)
            return rows

        if actual_sr != sr:
            log.warning(
                "  %s: expected sr=%d, got %d — skipping",
                src.name, sr, actual_sr
            )
            return rows

        # Step 3: split
        chunks = split_into_chunks(samples, sr, chunk_sec, min_chunk_sec)

        # Step 4: RMS filter + save
        for i, chunk in enumerate(chunks):
            rms_db = compute_rms_db(chunk)

            if rms_db < min_rms_db:
                log.debug(
                    "  Dropping chunk %d of %s (rms=%.1f dB < %.1f dB threshold)",
                    i, src.name, rms_db, min_rms_db
                )
                continue

            # Filename: label__stem__chunkindex.wav
            # Double underscore is the delimiter — avoids collisions with
            # single underscores in original filenames.
            out_name = f"{label}__{src.stem}__{i:04d}.wav"
            out_path = dst_dir / out_name

            save_wav(out_path, chunk, sr)

            rows.append({
                "path": str(out_path),
                "label": label,
                "source_file": str(src),
                "chunk_index": i,
                "duration_sec": len(chunk) / sr,
                "rms_db": round(rms_db, 2),
            })

    return rows


# ---------------------------------------------------------------------------
# Stats reporter
# ---------------------------------------------------------------------------


def print_stats(manifest_rows: list[dict]) -> dict:
    """Print per-class segment counts and return a summary dict."""
    from collections import defaultdict

    counts: dict[str, int] = defaultdict(int)
    for row in manifest_rows:
        counts[row["label"]] += 1

    total = sum(counts.values())

    log.info("")
    log.info("=" * 52)
    log.info("  %-22s  %8s  %7s", "Label", "Segments", "% total")
    log.info("-" * 52)
    for label, count in sorted(counts.items()):
        log.info("  %-22s  %8d  %6.1f%%", label, count, 100 * count / total)
    log.info("-" * 52)
    log.info("  %-22s  %8d", "TOTAL", total)
    log.info("=" * 52)
    log.info("")

    # Warn about class imbalance
    max_count = max(counts.values())
    for label, count in counts.items():
        ratio = max_count / count
        if ratio > 10:
            log.warning(
                "  Class '%s' is severely underrepresented (1:%d ratio). "
                "Consider augmentation or collecting more data.",
                label, int(ratio)
            )

    return dict(counts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess audio dataset for TAD classifier training."
    )
    parser.add_argument(
        "--src", type=Path, required=True,
        help="Source root directory.  Each subdirectory = one class label."
    )
    parser.add_argument(
        "--dst", type=Path, required=True,
        help="Output directory for processed WAV chunks and manifest."
    )
    parser.add_argument(
        "--chunk-sec", type=float, default=10.0,
        help="Chunk length in seconds (default: 10)."
    )
    parser.add_argument(
        "--min-rms-db", type=float, default=-40.0,
        help="Discard chunks quieter than this (dBFS, default: -40)."
    )
    parser.add_argument(
        "--sr", type=int, default=24_000,
        help="Target sample rate in Hz (default: 24000 — Moshi native rate)."
    )
    parser.add_argument(
        "--min-chunk-sec", type=float, default=2.0,
        help="Discard tail chunks shorter than this (default: 2 s)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and report without writing any files."
    )
    parser.add_argument(
        "--append", action="store_true",
        help=(
            "Append new chunks to an existing --dst instead of overwriting. "
            "New manifest rows are appended after existing ones. "
            "Use this to add a new class without reprocessing everything."
        )
    )
    args = parser.parse_args()

    check_ffmpeg()

    if not args.src.is_dir():
        log.error("Source directory not found: %s", args.src)
        sys.exit(1)

    # Collect all class subdirectories
    class_dirs = sorted([d for d in args.src.iterdir() if d.is_dir()])
    if not class_dirs:
        log.error(
            "No subdirectories found in %s.  "
            "Expected one subdirectory per class label.", args.src
        )
        sys.exit(1)

    labels = [d.name for d in class_dirs]
    log.info("Found %d class(es): %s", len(labels), ", ".join(labels))

    # Count source files
    all_files: list[tuple[Path, str]] = []
    for class_dir in class_dirs:
        for ext in SUPPORTED_EXTENSIONS:
            for f in class_dir.rglob(f"*{ext}"):
                all_files.append((f, class_dir.name))

    log.info("Found %d source files across all classes.", len(all_files))

    if args.dry_run:
        log.info("Dry run — exiting without writing files.")
        return

    # Create output directory
    args.dst.mkdir(parents=True, exist_ok=True)

    # Process all files
    all_rows: list[dict] = []
    skipped = 0

    with tqdm(total=len(all_files), unit="file", desc="Processing") as pbar:
        for src_path, label in all_files:
            pbar.set_postfix({"file": src_path.name[:30], "label": label})

            rows = process_file(
                src=src_path,
                label=label,
                dst_dir=args.dst,
                sr=args.sr,
                chunk_sec=args.chunk_sec,
                min_rms_db=args.min_rms_db,
                min_chunk_sec=args.min_chunk_sec,
            )

            if not rows:
                skipped += 1
            all_rows.extend(rows)
            pbar.update(1)

    log.info("Processed %d files → %d chunks (%d files skipped).",
             len(all_files), len(all_rows), skipped)

    # Write manifest CSV
    manifest_path = args.dst / MANIFEST_FILENAME
    fieldnames = ["path", "label", "source_file", "chunk_index", "duration_sec", "rms_db"]
    if args.append and manifest_path.exists():
        # Append — no header, just rows
        with open(manifest_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(all_rows)
        log.info("Appended %d rows to existing manifest %s", len(all_rows), manifest_path)
    else:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        log.info("Manifest written to %s", manifest_path)

    # Print stats and write JSON summary
    counts = print_stats(all_rows)
    stats_path = args.dst / STATS_FILENAME
    with open(stats_path, "w") as f:
        json.dump(
            {
                "total_chunks": len(all_rows),
                "skipped_files": skipped,
                "chunk_sec": args.chunk_sec,
                "min_rms_db": args.min_rms_db,
                "sample_rate": args.sr,
                "counts_per_class": counts,
            },
            f,
            indent=2,
        )
    log.info("Stats written to %s", stats_path)


if __name__ == "__main__":
    main()
