"""
Shared fixtures and signal generators for trublib tests.

All signals are at 24 kHz (Moshi's native rate) and mono float32.
"""

from __future__ import annotations

import numpy as np
import pytest

SR = 24_000
CHUNK = 1_920       # 80 ms @ 24 kHz
FRAME = 512
HOP = 256


# ---------------------------------------------------------------------------
# Signal factories (raw functions, usable without pytest)
# ---------------------------------------------------------------------------


def make_sine(freq: float, duration: float = 0.5, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    """Pure sine wave — maximally tonal, minimum spectral flatness."""
    t = np.arange(int(sr * duration)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_harmonic(
    f0: float,
    n_harmonics: int = 8,
    duration: float = 0.5,
    sr: int = SR,
    amp: float = 0.5,
    decay: float = 0.7,
) -> np.ndarray:
    """
    Additive synthesis with n_harmonics partials (amplitudes decay as decay^h).
    Approximates a trumpet-like harmonic series (both odd and even present).
    """
    n_samples = int(sr * duration)
    t = np.arange(n_samples) / sr
    signal = np.zeros(n_samples, dtype=np.float64)
    for h in range(1, n_harmonics + 1):
        freq = h * f0
        if freq >= sr / 2:
            break
        signal += (amp * (decay ** (h - 1))) * np.sin(2 * np.pi * freq * t)
    return signal.astype(np.float32)


def make_white_noise(duration: float = 0.5, sr: int = SR, amp: float = 0.3, seed: int = 42) -> np.ndarray:
    """White noise — maximum spectral flatness, no pitch."""
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(sr * duration))).astype(np.float32)


def make_chirp(
    f_start: float = 200.0,
    f_end: float = 800.0,
    duration: float = 0.5,
    sr: int = SR,
    amp: float = 0.5,
) -> np.ndarray:
    """
    Linear frequency sweep.  Useful to produce non-zero delta MFCCs
    (continuous timbre change = speech proxy).
    """
    n = int(sr * duration)
    t = np.arange(n) / sr
    instantaneous_freq = f_start + (f_end - f_start) * t / duration
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sr
    return (amp * np.sin(phase)).astype(np.float32)


def make_sustained_tone(
    f0: float = 465.0,   # Bb4 — middle of trumpet range
    duration: float = 1.0,
    sr: int = SR,
) -> np.ndarray:
    """Trumpet-proxy: harmonic, stable, long duration."""
    return make_harmonic(f0, n_harmonics=8, duration=duration, sr=sr)


def make_speech_proxy(duration: float = 0.5, sr: int = SR) -> np.ndarray:
    """
    Crude speech proxy: noise filtered by a time-varying resonance (formant-like).
    Not real speech, but tests formant detection without needing a speech corpus.
    """
    from scipy.signal import lfilter, butter

    rng = np.random.default_rng(0)
    noise = rng.standard_normal(int(sr * duration))

    # Sweep a bandpass filter through F1-like frequencies (400–900 Hz)
    n = len(noise)
    out = np.zeros(n)
    segment = n // 10
    for i in range(10):
        fc = 400 + i * 50  # 400 → 850 Hz
        b, a = butter(2, [fc / (sr / 2) * 0.8, fc / (sr / 2) * 1.2], btype="band")
        out[i * segment : (i + 1) * segment] = lfilter(
            b, a, noise[i * segment : (i + 1) * segment]
        )
    # Normalise
    mx = np.max(np.abs(out)) + 1e-10
    return (0.4 * out / mx).astype(np.float32)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sr():
    return SR


@pytest.fixture
def sustained_bb4():
    """Sustained Bb4 (465 Hz) with 8 harmonics — trumpet proxy."""
    return make_sustained_tone(f0=465.0, duration=1.0)


@pytest.fixture
def sustained_low_bb():
    """Sustained Bb2 (233 Hz) — low register trumpet."""
    return make_sustained_tone(f0=233.0, duration=1.0)


@pytest.fixture
def white_noise():
    return make_white_noise(duration=0.5)


@pytest.fixture
def chirp():
    return make_chirp(f_start=200, f_end=800, duration=0.5)
