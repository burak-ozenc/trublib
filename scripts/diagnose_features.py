"""
diagnose_features.py
---------------------
Diagnoses TAD model behaviour on a WAV file by replaying the exact
inference pipeline that TADProcessor uses — one independent 80ms chunk
at a time, with persistent FrameManager and FeatureExtractor state.

Previous version had a critical grouping bug: it extracted all frames as a
flat list then rechunked by CHUNK//HOP=7.  Actual steady-state FrameManager
output is 8 frames per chunk, so the flat-list groupings drifted out of phase
with real chunk boundaries.  The "chunk at time T" in the old timeline was not
what the model actually saw at time T.  This version fixes that.

What it does
~~~~~~~~~~~~
1. Loads a WAV file (resamples to 24kHz if needed)
2. Processes EACH 80ms chunk independently through:
       TwoStageNormalizer → FrameManager → FeatureExtractor → TrumpetScorer
   This is byte-for-byte identical to TADProcessor.process().
3. Prints per-chunk confidence timeline with rms_db, flatness, f0
4. Flags chunks the model should detect but doesn't (and vice versa)
5. Optionally compares against training data distribution

Usage
~~~~~
    python diagnose_features.py --file path/to/trumpet.wav
    python diagnose_features.py --file path/to/trumpet.wav --features data/train.npz
    python diagnose_features.py --file path/to/any.wav --threshold 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SR    = 24_000
CHUNK = 1_920   # 80ms @ 24kHz


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_wav(path: Path) -> np.ndarray:
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        samples = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        samples = data.astype(np.float32) / 2_147_483_648.0
    else:
        samples = data.astype(np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sr != SR:
        try:
            import soxr
            samples = soxr.resample(samples, sr, SR, quality="HQ").astype(np.float32)
            print(f"  Resampled {sr} Hz → {SR} Hz")
        except ImportError:
            print(f"  Warning: file is {sr} Hz but soxr not installed. Install with: pip install soxr")
    return samples


# ---------------------------------------------------------------------------
# Inference-aligned per-chunk processing
# ---------------------------------------------------------------------------

def process_all_chunks(samples: np.ndarray, scorer) -> list[dict]:
    """
    Process every 80ms chunk exactly as TADProcessor.process() does:
      TwoStageNormalizer → FrameManager → FeatureExtractor → TrumpetScorer

    State (pending buffer, MFCC history, prev_mag) is maintained continuously
    across chunks within this call, matching real-time streaming behaviour.

    Returns a list of dicts with per-chunk diagnostics.
    """
    from trublib.feature_extractor import FeatureExtractor
    from trublib.frame_manager import FrameManager
    from trublib.normalizer import TwoStageNormalizer

    norm = TwoStageNormalizer()
    fm   = FrameManager()
    ex   = FeatureExtractor(sr=SR)

    results = []

    for start in range(0, len(samples) - CHUNK + 1, CHUNK):
        chunk      = samples[start : start + CHUNK]
        normalised, rms_db = norm.process(chunk)
        frames     = fm.push(normalised)
        fvs        = [ex.extract(f, rms_db_override=rms_db) for f in frames]

        if len(fvs) < 3:
            # Too few frames — not enough for reliable pooling, skip scoring
            results.append({
                "time":     start / SR,
                "rms_db":   rms_db,
                "n_frames": len(fvs),
                "score":    0.0,
                "flatness": float("nan"),
                "f0_hz":    float("nan"),
                "hnr_db":   float("nan"),
                "skipped":  True,
            })
            continue

        mat  = np.stack([fv.to_vector() for fv in fvs], axis=0)  # (n_frames, 53)
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)
        seg  = np.concatenate([mean, std])[np.newaxis, :]         # (1, 106)

        probs = scorer._run_inference(seg)
        score = float(probs[0, 1])

        # Key diagnostic features (raw, un-normalised)
        fv0 = fvs[len(fvs) // 2]  # middle frame as representative
        results.append({
            "time":     start / SR,
            "rms_db":   rms_db,
            "n_frames": len(fvs),
            "score":    score,
            "flatness": fv0.spectral_flatness,
            "f0_hz":    fv0.f0_hz,
            "hnr_db":   fv0.hnr_db,
            "skipped":  False,
        })

    return results


# ---------------------------------------------------------------------------
# Training-style score (for reference — shows distribution shift)
# ---------------------------------------------------------------------------

def score_full_file(samples: np.ndarray, scorer) -> float:
    """
    Pool mean+std over ALL frames in the file, score as one segment.
    This is how the OLD extract_features.py worked, and why it broke.
    After retraining, this score is expected to be WRONG (low) for trumpet
    because the new model was trained on 80ms windows, not full-file pools.
    """
    from trublib.feature_extractor import FeatureExtractor
    from trublib.frame_manager import FrameManager
    from trublib.normalizer import TwoStageNormalizer

    norm = TwoStageNormalizer()
    fm   = FrameManager()
    ex   = FeatureExtractor(sr=SR)
    all_vecs = []

    for start in range(0, len(samples) - CHUNK + 1, CHUNK):
        chunk = samples[start : start + CHUNK]
        normalised, rms_db = norm.process(chunk)
        for frame in fm.push(normalised):
            all_vecs.append(ex.extract(frame, rms_db_override=rms_db).to_vector())

    if not all_vecs:
        return 0.0

    mat  = np.array(all_vecs, dtype=np.float32)
    mean = mat.mean(axis=0)
    std  = mat.std(axis=0)
    seg  = np.concatenate([mean, std])[np.newaxis, :]
    probs = scorer._run_inference(seg)
    return float(probs[0, 1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose TAD model inference on a WAV file."
    )
    parser.add_argument("--file",      type=Path, required=True,
                        help="WAV file to diagnose")
    parser.add_argument("--features",  type=Path, default=None,
                        help="train.npz from extract_features.py "
                             "(optional — shows training distribution comparison)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Detection threshold (default: 0.6)")
    parser.add_argument("--verbose",   action="store_true",
                        help="Show all chunks, not just active/interesting ones")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)

    from trublib.trumpet_scorer import TrumpetScorer

    print(f"\nLoading {args.file.name}...")
    samples  = load_wav(args.file)
    duration = len(samples) / SR
    n_chunks = len(samples) // CHUNK
    print(f"Duration: {duration:.2f}s  |  Samples: {len(samples)}  |  Chunks: {n_chunks}")

    scorer = TrumpetScorer()

    # ── 1. Full-file pooled score (expected ~0.0 with retrained model) ─────
    print("\n" + "="*60)
    print("  FULL-FILE POOLED SCORE")
    print("  (Expected ~0.0 — new model trained on 80ms windows,")
    print("   not full-file pools.  Low score here is CORRECT.)")
    print("="*60)
    p_full = score_full_file(samples, scorer)
    note = "✅ correct (model trained on 80ms windows)" if p_full < 0.5 else "⚠️  unexpected high (old model?)"
    print(f"  P(trumpet) = {p_full:.6f}  {note}")

    # ── 2. Per-chunk inference (the real diagnostic) ─────────────────────
    print("\n" + "="*60)
    print("  PER-CHUNK INFERENCE  (exact TADProcessor replay)")
    print(f"  Threshold: {args.threshold}")
    print("="*60)

    results = process_all_chunks(samples, scorer)

    scores = [r["score"] for r in results if not r["skipped"]]
    n_above = sum(1 for p in scores if p >= args.threshold)
    n_total = len(scores)

    print(f"\n  Chunks scored: {n_total}")
    print(f"  Above threshold ({args.threshold}): {n_above} ({100*n_above/max(n_total,1):.1f}%)")
    print(f"  Max P(trumpet): {max(scores):.6f}" if scores else "  No valid chunks.")
    print(f"  Mean P(trumpet): {np.mean(scores):.6f}" if scores else "")

    # Timeline
    print(f"\n  {'Time':>6}  {'rms_dB':>7}  {'flatness':>9}  {'f0_Hz':>7}  {'P(trumpet)':>11}  {'frames':>7}  Bar")
    print("  " + "-"*72)

    prev_above = False
    for r in results:
        if r["skipped"]:
            continue

        p      = r["score"]
        above  = p >= args.threshold
        rms    = r["rms_db"]
        flat   = r["flatness"]
        f0     = r["f0_hz"]
        t      = r["time"]
        nf     = r["n_frames"]

        # Show all chunks in verbose mode; otherwise show above-threshold,
        # boundaries around them, and quiet/interesting ones
        near_active = prev_above or above
        is_interesting = (rms > -55 and p < 0.05 and not np.isnan(flat) and flat < 0.2)

        if args.verbose or near_active or is_interesting:
            bar  = "█" * int(p * 25)
            flag = " ← DETECTED" if above else (" ← missed (tonal)" if is_interesting else "")
            flat_s = f"{flat:.3f}" if not np.isnan(flat) else "  n/a"
            f0_s   = f"{f0:.0f}" if (not np.isnan(f0) and f0 > 0) else "   —"
            print(f"  {t:>5.2f}s  {rms:>7.1f}  {flat_s:>9}  {f0_s:>7}  {p:>11.6f}  {nf:>7}  {bar}{flag}")
        elif not args.verbose:
            # Print a "..." placeholder when skipping a run of uninteresting chunks
            pass

        prev_above = above

    if not args.verbose:
        print()
        print("  (Use --verbose to show all chunks)")

    # ── 3. What the state machine would do ────────────────────────────────
    print("\n" + "="*60)
    print("  STATE MACHINE SIMULATION  (onset=3, trailing=4)")
    print("="*60)

    onset_chunks   = 3
    trailing_chunks = 4
    state          = "SILENT"
    onset_count    = 0
    trailing_count = 0
    active_periods = []
    active_start   = None

    for r in results:
        if r["skipped"]:
            continue
        above = r["score"] >= args.threshold
        t     = r["time"]

        if state == "SILENT":
            if above:
                onset_count = 1
                state = "ONSET" if onset_chunks > 1 else "ACTIVE"
                if state == "ACTIVE":
                    active_start = t
        elif state == "ONSET":
            if above:
                onset_count += 1
                if onset_count >= onset_chunks:
                    state = "ACTIVE"
                    active_start = t
            else:
                state = "SILENT"
                onset_count = 0
        elif state == "ACTIVE":
            if not above:
                trailing_count = 1
                state = "TRAILING"
        elif state == "TRAILING":
            if above:
                state = "ACTIVE"
                trailing_count = 0
            else:
                trailing_count += 1
                if trailing_count >= trailing_chunks:
                    active_periods.append((active_start, t))
                    state = "SILENT"
                    trailing_count = 0
                    active_start = None

    if active_start is not None:
        active_periods.append((active_start, results[-1]["time"]))

    if active_periods:
        print(f"\n  Active periods detected: {len(active_periods)}")
        total_active = sum(e - s for s, e in active_periods)
        print(f"  Total active duration: {total_active:.2f}s of {duration:.2f}s")
        for i, (s, e) in enumerate(active_periods, 1):
            print(f"  {i:>3}. {s:.2f}s → {e:.2f}s  ({e-s:.2f}s)")
    else:
        print(f"\n  No active periods detected.")
        print(f"  (onset_chunks=3 requires 3 consecutive chunks ≥ {args.threshold})")
        # Find best consecutive run
        run_max = 0
        run_cur = 0
        for r in results:
            if r["score"] >= args.threshold:
                run_cur += 1
                run_max = max(run_max, run_cur)
            else:
                run_cur = 0
        print(f"  Longest consecutive run above threshold: {run_max} chunk(s)")
        if run_max > 0:
            print(f"  Tip: lower --threshold or reduce onset_chunks in TADConfig")

    # ── 4. Training comparison (if --features provided) ──────────────────
    if args.features and args.features.exists():
        print("\n" + "="*60)
        print("  TRAINING DISTRIBUTION COMPARISON")
        print("="*60)

        data     = np.load(args.features, allow_pickle=True)
        X, y     = data["X"], data["y"]
        X_trump  = X[y == 1]
        X_other  = X[y == 0]

        feature_names = [
                            "spectral_centroid", "spectral_flatness", "spectral_flux", "spectral_rolloff",
                            "f0_hz", "pitch_salience", "hnr_db", "inharmonicity", "odd_even_ratio",
                        ] + [f"mfcc_{i}" for i in range(1, 14)] + \
                        [f"delta_mfcc_{i}" for i in range(1, 14)] + \
                        [f"mfcc_var_{i}" for i in range(1, 14)] + \
                        ["lpc_formant_count", "has_f1", "has_f2", "has_f3", "rms_db"]

        # Build per-chunk mean vectors for this file
        from trublib.feature_extractor import FeatureExtractor
        from trublib.frame_manager import FrameManager
        from trublib.normalizer import TwoStageNormalizer

        norm2 = TwoStageNormalizer()
        fm2   = FrameManager()
        ex2   = FeatureExtractor(sr=SR)

        chunk_vecs = []
        for start in range(0, len(samples) - CHUNK + 1, CHUNK):
            chunk = samples[start : start + CHUNK]
            normalised, rms_db = norm2.process(chunk)
            frames = fm2.push(normalised)
            fvs = [ex2.extract(f, rms_db_override=rms_db) for f in frames]
            if len(fvs) >= 3:
                mat  = np.stack([fv.to_vector() for fv in fvs])
                mean = mat.mean(axis=0)
                std  = mat.std(axis=0)
                chunk_vecs.append(np.concatenate([mean, std]))

        if not chunk_vecs:
            print("  No valid chunks for comparison.")
        else:
            this_X = np.array(chunk_vecs, dtype=np.float32)

            key_features = [0, 1, 2, 4, 5, 6, 7, 52]
            print(f"\n  {'Feature':<22}  {'Train trumpet':>14}  {'Train other':>12}  "
                  f"{'This file':>10}  {'Status':>12}")
            print("  " + "-"*78)

            for i in key_features:
                name   = feature_names[i] if i < len(feature_names) else f"feat_{i}"
                tr_m   = float(X_trump[:, i].mean())
                tr_s   = float(X_trump[:, i].std())
                ot_m   = float(X_other[:, i].mean())
                this_m = float(this_X[:, i].mean())

                # How many std devs from the trumpet mean?
                z = abs(this_m - tr_m) / (tr_s + 1e-10)
                if z < 1.0:
                    status = "✅ in-dist"
                elif z < 2.5:
                    status = "⚠️  borderline"
                else:
                    status = "❌ outlier"

                print(f"  {name:<22}  {tr_m:>8.4f}±{tr_s:.4f}  {ot_m:>12.4f}  "
                      f"{this_m:>10.4f}  {status} ({z:.1f}σ)")

            # RMS distribution specifically
            rms_col = 52
            this_rms_vals = this_X[:, rms_col] * 80 - 80
            train_rms_vals = X_trump[:, rms_col] * 80 - 80
            print(f"\n  rms_db distribution (dBFS):")
            print(f"    Training trumpet:  mean={train_rms_vals.mean():.1f}  "
                  f"p5={np.percentile(train_rms_vals, 5):.1f}  "
                  f"p95={np.percentile(train_rms_vals, 95):.1f}")
            print(f"    This file:         mean={this_rms_vals.mean():.1f}  "
                  f"p5={np.percentile(this_rms_vals, 5):.1f}  "
                  f"p95={np.percentile(this_rms_vals, 95):.1f}")

            rms_gap = float(X_trump[:, rms_col].mean()) - float(this_X[:, rms_col].mean())
            if rms_gap > 0.1:
                rms_gap_db = rms_gap * 80
                print(f"\n  ⚠️  This file is {rms_gap_db:.1f} dB quieter than avg training trumpet.")
                print(f"     The model may under-score quiet playing.")
                print(f"     → Try: lower TADConfig threshold from 0.6 to "
                      f"{max(0.3, 0.6 - rms_gap_db/40):.2f}")


if __name__ == "__main__":
    main()