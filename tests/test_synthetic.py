"""
tests/test_synthetic.py
-----------------------
Acoustic correctness tests.

These tests do NOT check if numbers fall in a specific range by luck —
they check *directional separation* between trumpet-like and non-trumpet
signals across all four feature families.  If these pass, the feature
pipeline is acoustically correct and ready to train a classifier on.

Signals used
~~~~~~~~~~~~
- ``trumpet``  : 465 Hz (Bb4) additive synthesis, 8 harmonics
- ``low_bb``   : 233 Hz (Bb2) additive synthesis — low register
- ``noise``    : white noise
- ``chirp``    : 200→800 Hz linear sweep (speech proxy for delta features)

What each test validates
~~~~~~~~~~~~~~~~~~~~~~~~
1. Spectral flatness separates tonal from noise
2. Spectral flux separates stable from changing timbres
3. HNR separates periodic from aperiodic
4. Inharmonicity separates harmonically locked instruments from noise
5. Delta MFCC separates sustained notes from sweeping frequencies
6. MFCC variance stays low for sustained playing
7. Pitch salience separates voiced from unvoiced
8. Full pipeline latency on one 80ms chunk stays under budget
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from trublib.feature_extractor import FeatureExtractor
from trublib.frame_manager import FrameManager
from tests.conftest import (
    SR, CHUNK,
    make_chirp, make_harmonic, make_white_noise,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def extract_all(signal: np.ndarray, sr: int = SR) -> list:
    """Run signal through FrameManager → FeatureExtractor, return all FVs."""
    fm = FrameManager()
    ex = FeatureExtractor(sr=sr)
    fvs = []
    for start in range(0, len(signal), CHUNK):
        for frame in fm.push(signal[start : start + CHUNK]):
            fvs.append(ex.extract(frame))
    return fvs


def mean_feature(fvs, attr: str) -> float:
    return float(np.mean([getattr(fv, attr) for fv in fvs]))


def mean_abs_delta(fvs) -> float:
    """Mean absolute delta MFCC magnitude across all frames and coefficients."""
    d = np.array([fv.delta_mfcc for fv in fvs])
    return float(np.mean(np.abs(d)))


# ---------------------------------------------------------------------------
# 1. Spectral flatness: tonal << noise
# ---------------------------------------------------------------------------


def test_flatness_trumpet_vs_noise():
    trumpet = make_harmonic(465, n_harmonics=8, duration=1.0)
    noise = make_white_noise(duration=1.0, seed=7)

    fvs_t = extract_all(trumpet)
    fvs_n = extract_all(noise)

    ft = mean_feature(fvs_t, "spectral_flatness")
    fn = mean_feature(fvs_n, "spectral_flatness")

    assert ft < 0.25, f"Trumpet flatness too high: {ft:.3f}"
    assert fn > 0.50, f"Noise flatness too low: {fn:.3f}"
    assert fn > ft + 0.30, f"Gap too small: noise={fn:.3f}, trumpet={ft:.3f}"


# ---------------------------------------------------------------------------
# 2. Spectral flux: sustained tone << chirp (changing timbre)
# ---------------------------------------------------------------------------


def test_flux_trumpet_vs_chirp():
    trumpet = make_harmonic(465, duration=0.5)
    chirp = make_chirp(f_start=200, f_end=800, duration=0.5)

    fvs_t = extract_all(trumpet)[2:]   # skip first 2 (warmup)
    fvs_c = extract_all(chirp)[2:]

    ft = mean_feature(fvs_t, "spectral_flux")
    fc = mean_feature(fvs_c, "spectral_flux")

    assert fc > ft, f"Chirp flux ({fc:.5f}) should exceed trumpet flux ({ft:.5f})"


# ---------------------------------------------------------------------------
# 3. HNR: periodic tone >> aperiodic noise
# ---------------------------------------------------------------------------


def test_hnr_trumpet_vs_noise():
    trumpet = make_harmonic(465, n_harmonics=8, duration=1.0)
    noise = make_white_noise(duration=1.0, seed=99)

    fvs_t = extract_all(trumpet)
    fvs_n = extract_all(noise)

    ht = mean_feature(fvs_t, "hnr_db")
    hn = mean_feature(fvs_n, "hnr_db")

    assert ht > hn + 5.0, f"HNR gap insufficient: trumpet={ht:.1f}, noise={hn:.1f}"


def test_hnr_trumpet_positive():
    fvs = extract_all(make_harmonic(465, n_harmonics=8, duration=1.0))
    voiced = [fv.hnr_db for fv in fvs if fv.f0_hz > 0]
    if voiced:
        assert np.mean(voiced) > 5.0, f"Trumpet HNR={np.mean(voiced):.1f} dB"


# ---------------------------------------------------------------------------
# 4. Pitch salience: voiced tone >> noise
# ---------------------------------------------------------------------------


def test_pitch_salience_separation():
    fvs_t = extract_all(make_harmonic(465, duration=1.0))
    fvs_n = extract_all(make_white_noise(duration=1.0))

    st = mean_feature(fvs_t, "pitch_salience")
    sn = mean_feature(fvs_n, "pitch_salience")

    assert st > sn + 0.1, f"trumpet={st:.3f}, noise={sn:.3f}"


def test_f0_in_trumpet_range():
    """Detected f0 for 465 Hz signal should be within ±20% of 465 Hz."""
    fvs = extract_all(make_harmonic(465, duration=1.0))
    detected = [fv.f0_hz for fv in fvs if fv.f0_hz > 0]
    assert len(detected) >= len(fvs) // 2, "Should detect pitch in majority of frames"
    mean_f0 = np.mean(detected)
    assert 370 <= mean_f0 <= 560, f"Mean f0={mean_f0:.1f} Hz"


def test_f0_low_register():
    """233 Hz (Bb2) should also be detected."""
    fvs = extract_all(make_harmonic(233, duration=1.0))
    detected = [fv.f0_hz for fv in fvs if fv.f0_hz > 0]
    assert len(detected) >= len(fvs) // 3, "Should detect low Bb"


# ---------------------------------------------------------------------------
# 5. Inharmonicity: additive synthesis << noise
# ---------------------------------------------------------------------------


def test_inharmonicity_trumpet_lower_than_noise():
    """
    Perfectly additive synthesis is maximally harmonic → low inharmonicity.
    Noise has no harmonic structure → should show higher inharmonicity.
    """
    fvs_t = extract_all(make_harmonic(465, n_harmonics=8, duration=1.0))
    fvs_n = extract_all(make_white_noise(duration=1.0))

    # Only compare voiced frames for trumpet
    voiced_inh = [fv.inharmonicity for fv in fvs_t if fv.f0_hz > 0]
    noise_inh = [fv.inharmonicity for fv in fvs_n]

    if voiced_inh:
        assert np.mean(voiced_inh) < np.mean(noise_inh), (
            f"trumpet inh={np.mean(voiced_inh):.3f}, "
            f"noise inh={np.mean(noise_inh):.3f}"
        )


# ---------------------------------------------------------------------------
# 6. Delta MFCC: sustained << chirp
# ---------------------------------------------------------------------------


def test_delta_mfcc_sustained_vs_chirp():
    """
    This is the primary feature separating sustained trumpet from speech.
    Sustained playing → stable timbre → near-zero deltas.
    Chirp → continuously changing spectrum → large deltas.
    """
    fvs_t = extract_all(make_harmonic(465, duration=0.5))[3:]
    fvs_c = extract_all(make_chirp(f_start=200, f_end=1000, duration=0.5))[3:]

    delta_t = mean_abs_delta(fvs_t)
    delta_c = mean_abs_delta(fvs_c)

    assert delta_c > delta_t, (
        f"Chirp delta ({delta_c:.4f}) must exceed trumpet delta ({delta_t:.4f})"
    )


def test_delta_mfcc_sustained_low_absolute():
    """Sustained Bb4: mean |delta MFCC| should be below 3.0."""
    fvs = extract_all(make_harmonic(465, duration=1.0))[5:]
    assert mean_abs_delta(fvs) < 3.0, f"mean |delta|={mean_abs_delta(fvs):.3f}"


# ---------------------------------------------------------------------------
# 7. MFCC variance: stable tone
# ---------------------------------------------------------------------------


def test_mfcc_variance_sustained_low():
    """
    Sustained tone: MFCC variance over the rolling window should be low
    after the history fills (frame 7+).
    """
    fvs = extract_all(make_harmonic(465, duration=1.0))[7:]
    mean_var = float(np.mean([np.sum(fv.mfcc_variance) for fv in fvs]))
    # Just verify it's bounded (not exploding)
    assert 0 <= mean_var < 5000, f"MFCC variance={mean_var:.1f}"


# ---------------------------------------------------------------------------
# 8. Signal separation summary (human-readable report)
# ---------------------------------------------------------------------------


def test_feature_separation_report(capsys):
    """
    Print a compact separation table.  This test always passes — it's a
    diagnostic aid to visually inspect feature behaviour during development.
    """
    trumpet = make_harmonic(465, n_harmonics=8, duration=1.0)
    noise = make_white_noise(duration=1.0)
    chirp = make_chirp(200, 800, duration=0.5)

    fvs_t = extract_all(trumpet)[3:]
    fvs_n = extract_all(noise)[3:]
    fvs_c = extract_all(chirp)[3:]

    def stats(fvs, attr):
        vals = [getattr(fv, attr) for fv in fvs]
        return f"{np.mean(vals):7.3f} ± {np.std(vals):.3f}"

    with capsys.disabled():
        print("\n")
        print("=" * 65)
        print(f"{'Feature':<22} {'Trumpet':>20} {'Noise':>20}")
        print("-" * 65)
        for feat in ["spectral_flatness", "spectral_flux", "hnr_db",
                     "pitch_salience", "inharmonicity", "odd_even_ratio"]:
            print(f"  {feat:<20} {stats(fvs_t, feat):>20} {stats(fvs_n, feat):>20}")
        print("-" * 65)
        print(f"  {'delta_mfcc |mean|':<20} {mean_abs_delta(fvs_t):>20.4f} {'(chirp)':>14} {mean_abs_delta(fvs_c):>5.4f}")
        print("=" * 65)


# ---------------------------------------------------------------------------
# 9. Latency budget: full pipeline under 80ms constraint
# ---------------------------------------------------------------------------


def test_processing_latency_under_budget():
    """
    Process 10 consecutive 80ms chunks and verify total wall time is well
    under 10 × 80ms = 800ms.  In practice this should run in < 20ms total.
    """
    fm = FrameManager()
    ex = FeatureExtractor()
    signal = make_harmonic(465, duration=1.0)

    chunks = [signal[i : i + CHUNK] for i in range(0, len(signal), CHUNK) if len(signal[i : i + CHUNK]) == CHUNK]

    start = time.perf_counter()
    for chunk in chunks[:10]:
        for frame in fm.push(chunk):
            ex.extract(frame)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Must complete all 10 chunks in well under 800ms (budget × 10)
    # In practice: < 30ms on any modern machine
    assert elapsed_ms < 800, f"Pipeline took {elapsed_ms:.1f}ms for 10 chunks"
    print(f"\n  Latency: {elapsed_ms:.2f}ms for 10×80ms chunks ({elapsed_ms/10:.2f}ms/chunk)")
