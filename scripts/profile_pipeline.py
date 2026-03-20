"""
profile_pipeline.py
--------------------
Wall-clock profiling for the TADProcessor pipeline.

Two modes
~~~~~~~~~
--mode timing   Fast benchmark: measures per-chunk latency over N chunks,
                reports mean / p95 / p99 / max and per-stage breakdown.
                Run this first — no extra tools needed.

--mode cprofile Runs cProfile over the benchmark loop and prints the top 30
                functions by cumulative time.  Good for finding which Python
                calls dominate.

--mode pyspy    Emits instructions for running py-spy record on this script.
                py-spy produces a flamegraph SVG — best for finding C-extension
                bottlenecks that cProfile misses (e.g. inside numpy/scipy).

Usage
~~~~~
    # Step 1 — timing breakdown (always run this first)
    python profile_pipeline.py --mode timing --chunks 200

    # Step 2 — cProfile (no install needed)
    python profile_pipeline.py --mode cprofile --chunks 500

    # Step 3 — flamegraph (requires: pip install py-spy)
    python profile_pipeline.py --mode pyspy

What to look for
~~~~~~~~~~~~~~~~
Per-chunk budget: 80ms.  Comfortable target: < 15ms (leaves 65ms headroom).

Likely hot paths (in order):
    1. FeatureExtractor._lpc_formants  — Levinson-Durbin + np.roots
    2. SoftMaskGenerator.apply         — scipy stft/istft
    3. FeatureExtractor._compute_mfcc  — mel filterbank matmul
    4. TrumpetScorer._run_inference    — onnxruntime (already C++)
    5. FrameManager.push               — numpy concat in loop

Rust candidates (only after profiling confirms):
    - FrameManager.push (ring buffer + windowing)
    - FeatureExtractor core (LPC, MFCC, pitch)
    - SoftMaskGenerator (STFT/ISTFT if scipy overhead is measurable)

Requirements
~~~~~~~~~~~~
    pip install numpy scipy onnxruntime  (already installed)
    pip install py-spy                   (only for --mode pyspy flamegraph)
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

SR     = 24_000
CHUNK  = 1_920   # 80ms @ 24kHz


def make_trumpet_signal(duration: float = 30.0) -> np.ndarray:
    """Additive synthesis trumpet proxy — 8 harmonics at Bb4 (465 Hz)."""
    n = int(SR * duration)
    t = np.arange(n) / SR
    signal = np.zeros(n, dtype=np.float32)
    for h in range(1, 9):
        signal += (0.5 * (0.7 ** (h - 1))) * np.sin(2 * np.pi * h * 465 * t)
    return signal


# ---------------------------------------------------------------------------
# Per-stage timer context manager
# ---------------------------------------------------------------------------


class StageTimer:
    """Accumulates wall-clock time per named stage across many calls."""

    def __init__(self):
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._current: str | None = None
        self._t0: float = 0.0

    def start(self, stage: str):
        self._current = stage
        self._t0 = time.perf_counter()

    def stop(self):
        if self._current is None:
            return
        elapsed = time.perf_counter() - self._t0
        self._totals[self._current] = self._totals.get(self._current, 0.0) + elapsed
        self._counts[self._current] = self._counts.get(self._current, 0) + 1
        self._current = None

    def report(self, n_chunks: int):
        print("\nPer-stage breakdown (mean ms per chunk):")
        print(f"  {'Stage':<30}  {'mean ms':>8}  {'total ms':>9}  {'calls':>6}")
        print("  " + "-" * 60)
        total_pipeline = sum(self._totals.values())
        for stage, total in sorted(self._totals.items(), key=lambda x: -x[1]):
            mean_ms = total * 1000 / n_chunks
            pct = 100 * total / total_pipeline if total_pipeline > 0 else 0
            print(
                f"  {stage:<30}  {mean_ms:>8.3f}  {total*1000:>9.1f}  "
                f"{self._counts[stage]:>6}  ({pct:.1f}%)"
            )
        print("  " + "-" * 60)
        print(f"  {'TOTAL PIPELINE':<30}  {total_pipeline*1000/n_chunks:>8.3f}")


# ---------------------------------------------------------------------------
# Instrumented pipeline (stage-by-stage timing without TADProcessor wrapper)
# ---------------------------------------------------------------------------


def run_instrumented(n_chunks: int, signal: np.ndarray) -> dict:
    """
    Run the full pipeline chunk by chunk with per-stage timing.
    Bypasses TADProcessor to time each component individually.
    """
    from trublib.config import TADConfig
    from trublib.feature_extractor import FeatureExtractor
    from trublib.frame_manager import FrameManager
    from trublib.normalizer import TwoStageNormalizer
    from trublib.soft_mask import SoftMaskGenerator
    from trublib.tad_state_machine import TADStateMachine
    from trublib.trumpet_scorer import TrumpetScorer

    cfg     = TADConfig()
    norm    = TwoStageNormalizer(target_rms=cfg.target_rms)
    fm      = FrameManager()
    ex      = FeatureExtractor(sr=SR)
    scorer  = TrumpetScorer()
    mask_gen = SoftMaskGenerator(sr=SR)
    sm      = TADStateMachine(
        onset_chunks=cfg.onset_chunks,
        trailing_chunks=cfg.trailing_chunks,
        threshold=cfg.threshold,
        lookback_frames=cfg.lookback_frames,
    )

    timer   = StageTimer()
    latencies = []

    chunks = [
        signal[i : i + CHUNK]
        for i in range(0, len(signal) - CHUNK, CHUNK)
    ][:n_chunks]

    # Warm up (model load, JIT, etc.)
    print(f"Warming up ({min(5, len(chunks))} chunks)...")
    for chunk in chunks[:5]:
        normalised, rms_db = norm.process(chunk)
        frames = fm.push(normalised)
        fvs = [ex.extract(f, rms_db_override=rms_db) for f in frames]
        scorer.score(fvs)

    fm.reset()
    ex.reset()
    sm.reset()
    print(f"Running {n_chunks} chunks...\n")

    for chunk in chunks:
        t_chunk_start = time.perf_counter()

        timer.start("1_normalizer")
        normalised, rms_db = norm.process(chunk)
        timer.stop()

        timer.start("2_frame_manager")
        frames = fm.push(normalised)
        timer.stop()

        timer.start("3_feature_extractor")
        fvs = [ex.extract(f, rms_db_override=rms_db) for f in frames]
        timer.stop()

        timer.start("4_trumpet_scorer")
        confidence = scorer.score(fvs)
        timer.stop()

        timer.start("5_soft_mask")
        masked = mask_gen.apply(normalised, confidence)
        timer.stop()

        timer.start("6_state_machine")
        sm.update(confidence, normalised, masked)
        timer.stop()

        latencies.append((time.perf_counter() - t_chunk_start) * 1000)

    latencies_arr = np.array(latencies)
    return {"latencies": latencies_arr, "timer": timer}


# ---------------------------------------------------------------------------
# Mode: timing
# ---------------------------------------------------------------------------


def mode_timing(n_chunks: int):
    signal = make_trumpet_signal(duration=max(30.0, n_chunks * 0.08 + 5))
    result = run_instrumented(n_chunks, signal)

    lat = result["latencies"]
    timer = result["timer"]

    print("=" * 60)
    print(f"  TADProcessor pipeline — {n_chunks} chunks @ 80ms each")
    print("=" * 60)
    print(f"  Mean latency:  {lat.mean():.3f} ms")
    print(f"  Median:        {np.median(lat):.3f} ms")
    print(f"  p95:           {np.percentile(lat, 95):.3f} ms")
    print(f"  p99:           {np.percentile(lat, 99):.3f} ms")
    print(f"  Max:           {lat.max():.3f} ms")
    print(f"  Min:           {lat.min():.3f} ms")
    print(f"  Std:           {lat.std():.3f} ms")
    print(f"  Budget (80ms): {'✅ OK' if lat.max() < 80 else '❌ EXCEEDED'}")
    print(f"  Headroom:      {80 - lat.mean():.1f} ms mean / {80 - lat.max():.1f} ms worst")
    print("=" * 60)

    timer.report(n_chunks)

    # Histogram buckets
    buckets = [0, 5, 10, 15, 20, 30, 40, 80, float("inf")]
    labels  = ["<5ms", "5-10ms", "10-15ms", "15-20ms", "20-30ms", "30-40ms", "40-80ms", ">80ms"]
    print("\nLatency distribution:")
    for i, label in enumerate(labels):
        lo, hi = buckets[i], buckets[i + 1]
        count = int(np.sum((lat >= lo) & (lat < hi)))
        bar = "█" * (count * 40 // n_chunks)
        pct = 100 * count / n_chunks
        print(f"  {label:>10}  {bar:<40}  {count:>5} ({pct:.1f}%)")

    print("\nRust recommendation:")
    stage_totals = {k: v for k, v in result["timer"]._totals.items()}
    sorted_stages = sorted(stage_totals.items(), key=lambda x: -x[1])
    top = sorted_stages[0]
    top_pct = 100 * top[1] / sum(stage_totals.values())
    mean_ms = top[1] * 1000 / n_chunks
    if lat.mean() < 15:
        print(f"  ✅ Mean latency {lat.mean():.1f}ms is well under budget.")
        print(f"     Pure Python + NumPy is sufficient. No Rust needed yet.")
    elif top_pct > 40:
        print(f"  ⚡ '{top[0]}' dominates at {top_pct:.0f}% ({mean_ms:.1f}ms/chunk).")
        print(f"     This is the primary Rust candidate.")
    else:
        print(f"  ℹ  Time is spread across stages. Profile with py-spy for")
        print(f"     a flamegraph before deciding on Rust targets.")


# ---------------------------------------------------------------------------
# Mode: cprofile
# ---------------------------------------------------------------------------


def mode_cprofile(n_chunks: int):
    from trublib.processor import TADProcessor
    from trublib.config import TADConfig

    signal = make_trumpet_signal(duration=max(30.0, n_chunks * 0.08 + 5))
    chunks = [
        signal[i : i + CHUNK]
        for i in range(0, len(signal) - CHUNK, CHUNK)
    ][:n_chunks]

    tad = TADProcessor(TADConfig())

    # Warm up
    for chunk in chunks[:5]:
        tad.process(chunk)

    print(f"Running cProfile over {n_chunks} chunks...")
    pr = cProfile.Profile()
    pr.enable()
    for chunk in chunks:
        tad.process(chunk)
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(40)
    print(buf.getvalue())

    # Also save to file
    out = Path("profile_output.prof")
    pr.dump_stats(str(out))
    print(f"Profile data saved to {out}")
    print("Visualise with: python -m snakeviz profile_output.prof")


# ---------------------------------------------------------------------------
# Mode: pyspy
# ---------------------------------------------------------------------------


def mode_pyspy():
    print("=" * 60)
    print("  py-spy flamegraph instructions")
    print("=" * 60)
    print()
    print("Install py-spy:")
    print("  pip install py-spy")
    print()
    print("Run the pipeline under py-spy (records a flamegraph SVG):")
    print()
    print("  # On Linux/macOS (may need sudo):")
    print(f"  py-spy record -o flamegraph.svg -- python {__file__} --mode timing --chunks 500")
    print()
    print("  # On Windows (run as Administrator):")
    print(f"  py-spy record -o flamegraph.svg -- python {__file__} --mode timing --chunks 500")
    print()
    print("Open flamegraph.svg in your browser.")
    print()
    print("What to look for in the flamegraph:")
    print("  - Wide bars at the top = most wall time")
    print("  - np.roots / np.linalg inside _lpc_formants → Rust candidate")
    print("  - scipy stft/istft → measure width vs total")
    print("  - onnxruntime → should be narrow (already C++)")
    print("  - numpy matmul in _compute_mfcc → unlikely to need Rust")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Profile the TADProcessor pipeline."
    )
    parser.add_argument(
        "--mode",
        choices=["timing", "cprofile", "pyspy"],
        default="timing",
        help="Profiling mode (default: timing)"
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=200,
        help="Number of 80ms chunks to process (default: 200)"
    )
    args = parser.parse_args()

    if args.mode == "timing":
        mode_timing(args.chunks)
    elif args.mode == "cprofile":
        mode_cprofile(args.chunks)
    elif args.mode == "pyspy":
        mode_pyspy()


if __name__ == "__main__":
    main()
