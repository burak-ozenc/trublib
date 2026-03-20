"""
trublib.feature_extractor
-------------------------
Computes the full trumpet-discriminating feature vector from a single
Hann-windowed 512-sample frame.

Feature families
~~~~~~~~~~~~~~~~
1. Spectral   — centroid, flatness (Wiener entropy), flux, rolloff
2. Harmonic   — f0, pitch salience, HNR, inharmonicity, odd/even ratio
3. Cepstral   — MFCCs 1–13, delta-MFCCs, MFCC variance (rolling window)
4. Formant    — LPC-derived speech-formant detection (F1/F2/F3)

All implementations are intentionally in pure NumPy/SciPy so that
measured hot paths can be surgically ported to Rust (PyO3/Maturin) without
changing the Python API.  Do **not** introduce PyTorch or librosa here.

Key design decisions
~~~~~~~~~~~~~~~~~~~~
- FFT over the 512-sample Hann-windowed frame (= 46.9 Hz/bin @ 24 kHz).
  The 1024-point STFT used for *masking* is separate and lives in the full
  pipeline (not yet implemented).
- Mel filterbank built once in __init__, stored as a float32 matrix.
- Delta-MFCCs use a causal rolling buffer (no future frames).
- LPC order 12 with Levinson-Durbin; speech-formant detection via pole
  bandwidth and frequency range checks.
- All methods return float32 values; no hidden allocations after __init__.

Not thread-safe.  One instance per stream.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.fft import dct as scipy_dct


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class FeatureVector:
    """
    All features extracted from one 512-sample windowed frame.

    Call :meth:`to_vector` to obtain the flat float32 array fed to the
    ONNX classifier (53 dimensions).
    """

    # ---- Spectral (4) ----
    spectral_centroid: float
    """Weighted mean frequency of the magnitude spectrum (Hz)."""

    spectral_flatness: float
    """
    Wiener entropy = geometric_mean(mag) / arithmetic_mean(mag).
    Trumpet: 0.01–0.10 (tonal).  Noise: 0.70–1.00.
    """

    spectral_flux: float
    """
    L2 distance between current and previous magnitude spectrum, normalised
    by spectrum length.  Near-zero for sustained trumpet notes.
    """

    spectral_rolloff: float
    """Frequency (Hz) below which 85 % of spectral energy lies."""

    # ---- Harmonic (5) ----
    f0_hz: float
    """Estimated fundamental frequency (Hz).  0.0 if no pitch detected."""

    pitch_salience: float
    """
    Normalised autocorrelation peak height [0, 1].  High for periodic
    signals (sustained trumpet), low for noise and speech transitions.
    """

    hnr_db: float
    """
    Harmonic-to-Noise Ratio in dB, clipped to [−10, 40].
    Trumpet during sustained notes: 20–35 dB.
    Noise / aperiodic signal: negative values.
    """

    inharmonicity: float
    """
    Mean relative deviation of upper harmonics from ideal f0 multiples.
    Trumpet: near 0.0 (bore physics forces harmonic alignment).
    Higher values indicate inharmonic / noisy sources.
    """

    odd_even_ratio: float
    """
    Ratio of odd-harmonic energy to even-harmonic energy.
    Trumpet (open conical resonator approximation): ~1.0 (both present).
    Clarinet (closed cylindrical): >> 1.0.  Good clarinet separator.
    """

    # ---- Cepstral (39) ----
    mfcc: np.ndarray
    """Mel-Frequency Cepstral Coefficients 1–13 (shape 13,)."""

    delta_mfcc: np.ndarray
    """First-order causal delta of MFCCs (shape 13,).  ~0 for sustained tone."""

    mfcc_variance: np.ndarray
    """Variance of MFCCs over the last 7 frames (shape 13,)."""

    # ---- Formant / LPC (4) ----
    lpc_formant_count: int
    """Number of speech-like formants detected (0–3)."""

    has_f1_formant: bool
    """Narrow-bandwidth pole in F1 range 300–900 Hz."""

    has_f2_formant: bool
    """Narrow-bandwidth pole in F2 range 850–2500 Hz."""

    has_f3_formant: bool
    """Narrow-bandwidth pole in F3 range 2000–3500 Hz."""

    # ---- Energy (1) ----
    rms_db: float
    """
    Frame RMS in dB (floor −80 dB).  In production, the TwoStageNormalizer
    provides the pre-normalisation chunk RMS via rms_db_override so the
    classifier sees actual playing loudness rather than the normalised frame.
    """

    # ------------------------------------------------------------------

    def to_vector(self) -> np.ndarray:
        """
        Flatten to a 1-D float32 array (53 dims) for the ONNX classifier.

        Values are soft-normalised to roughly [0, 1] or [-1, 1] ranges
        based on expected acoustic bounds.  These constants are intentional
        and must match the training-time normalisation.
        """
        v = np.empty(53, dtype=np.float32)

        # Spectral
        v[0] = self.spectral_centroid / 12_000.0       # 0 → Nyquist
        v[1] = float(self.spectral_flatness)            # already [0, 1]
        v[2] = float(self.spectral_flux)                # already small
        v[3] = self.spectral_rolloff / 12_000.0

        # Harmonic
        v[4] = self.f0_hz / 2_000.0
        v[5] = float(self.pitch_salience)               # [0, 1]
        v[6] = (self.hnr_db + 10.0) / 50.0             # [-10, 40] → [0, 1]
        v[7] = float(np.clip(self.inharmonicity, 0, 1))
        v[8] = float(np.clip(self.odd_even_ratio / 4.0, 0, 2))

        # Cepstral
        v[9:22] = self.mfcc / 80.0
        v[22:35] = self.delta_mfcc / 20.0
        v[35:48] = np.sqrt(np.clip(self.mfcc_variance, 0, None)) / 20.0

        # Formant
        v[48] = self.lpc_formant_count / 3.0
        v[49] = float(self.has_f1_formant)
        v[50] = float(self.has_f2_formant)
        v[51] = float(self.has_f3_formant)

        # Energy
        v[52] = (self.rms_db + 80.0) / 80.0            # [-80, 0] → [0, 1]

        return v

    @property
    def feature_dim(self) -> int:
        return 53


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """
    Stateful feature extractor.  Call :meth:`extract` once per :class:`Frame`.

    State
    -----
    - ``_prev_mag``      : previous frame's magnitude spectrum (spectral flux)
    - ``_mfcc_history``  : rolling deque of last 7 MFCC vectors (delta/variance)

    Parameters
    ----------
    sr : int
        Sample rate.  Must match the FrameManager's output (24 000 Hz).
    rms_db_override : float | None
        If provided, replaces the per-frame RMS calculation with the
        pre-normalisation chunk RMS from TwoStageNormalizer.  Pass this
        in production so the classifier sees actual playing loudness.
    """

    # --- FFT / mel constants ---
    _N_FFT: int = 512        # frame size; 46.9 Hz/bin @ 24 kHz
    _N_BINS: int = 257       # N_FFT//2 + 1
    _N_MELS: int = 40
    _N_MFCC: int = 13
    _MEL_FMIN: float = 80.0
    _MEL_FMAX: float = 8_000.0

    # --- Pitch detection range ---
    _F0_MIN: float = 100.0   # covers pedal tones below standard Bb2 (233 Hz)
    _F0_MAX: float = 1_200.0 # above high C6 (1047 Hz) for safety margin

    # --- LPC ---
    _LPC_ORDER: int = 12     # sr/1000 + 2 = 26, but 12 avoids overfitting
    _MAX_FORMANT_BW: float = 400.0  # Hz — narrower = more speech-like

    # --- Rolling MFCC window ---
    _MFCC_WINDOW: int = 7

    def __init__(self, sr: int = 24_000) -> None:
        self._sr = sr
        self._bin_freqs = np.linspace(0.0, sr / 2.0, self._N_BINS, dtype=np.float32)

        # Pitch lag bounds in samples
        self._lag_min = max(1, int(sr / self._F0_MAX))   # ≈ 20
        self._lag_max = int(sr / self._F0_MIN)            # = 240

        # Pre-build mel filterbank — computed once, never reallocated
        self._mel_fb: np.ndarray = self._build_mel_filterbank(
            sr, self._N_FFT, self._N_MELS, self._MEL_FMIN, self._MEL_FMAX
        )

        # Rolling state
        self._prev_mag: Optional[np.ndarray] = None
        self._mfcc_history: deque[np.ndarray] = deque(maxlen=self._MFCC_WINDOW)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        frame,                          # Frame NamedTuple from FrameManager
        rms_db_override: Optional[float] = None,
    ) -> FeatureVector:
        """
        Extract all features from a single windowed :class:`Frame`.

        Parameters
        ----------
        frame : Frame
            Output of ``FrameManager.push()``.
        rms_db_override : float, optional
            Pre-normalisation RMS dB from ``TwoStageNormalizer.process()``.
            When supplied, replaces the in-frame RMS so the classifier sees
            the actual playing loudness (not the normalised level).
        """
        w = frame.windowed  # Hann-windowed, float32, shape (512,)

        # --- FFT magnitude spectrum ---
        mag = np.abs(np.fft.rfft(w, n=self._N_FFT)).astype(np.float32)  # (257,)
        eps = 1e-10

        # === 1. Spectral features ===
        centroid = self._spectral_centroid(mag, eps)
        flatness = self._spectral_flatness(mag, eps)
        flux = self._spectral_flux(mag, eps)
        rolloff = self._spectral_rolloff(mag)

        # === 2. Harmonic features ===
        f0, salience = self._detect_pitch(w)
        hnr = self._compute_hnr(w, f0, eps)
        inharmonicity = self._compute_inharmonicity(mag, f0, eps)
        odd_even = self._compute_odd_even_ratio(mag, f0, eps)

        # === 3. Cepstral features ===
        mfcc = self._compute_mfcc(mag, eps)
        self._mfcc_history.append(mfcc)
        delta = self._compute_delta_mfcc()
        variance = self._compute_mfcc_variance()

        # === 4. LPC formants ===
        fcount, has_f1, has_f2, has_f3 = self._lpc_formants(w)

        # === Energy ===
        if rms_db_override is not None:
            rms_db = float(rms_db_override)
        else:
            rms = float(np.sqrt(np.mean(w ** 2)))
            rms_db = float(20.0 * np.log10(max(rms, 1e-10)))
        rms_db = max(rms_db, -80.0)

        # Update flux state for next call
        self._prev_mag = mag

        return FeatureVector(
            spectral_centroid=centroid,
            spectral_flatness=flatness,
            spectral_flux=flux,
            spectral_rolloff=rolloff,
            f0_hz=f0,
            pitch_salience=salience,
            hnr_db=hnr,
            inharmonicity=inharmonicity,
            odd_even_ratio=odd_even,
            mfcc=mfcc,
            delta_mfcc=delta,
            mfcc_variance=variance,
            lpc_formant_count=fcount,
            has_f1_formant=has_f1,
            has_f2_formant=has_f2,
            has_f3_formant=has_f3,
            rms_db=rms_db,
        )

    def reset(self) -> None:
        """Clear rolling state (e.g. on stream restart)."""
        self._prev_mag = None
        self._mfcc_history.clear()

    # ------------------------------------------------------------------
    # 1. Spectral
    # ------------------------------------------------------------------

    def _spectral_centroid(self, mag: np.ndarray, eps: float) -> float:
        total = float(np.sum(mag)) + eps
        return float(np.dot(self._bin_freqs, mag) / total)

    def _spectral_flatness(self, mag: np.ndarray, eps: float) -> float:
        """
        Wiener entropy.  We work in log-space for numerical stability:
        geometric_mean = exp(mean(log(mag))).
        """
        log_mag = np.log(mag + eps)
        geometric_mean = float(np.exp(np.mean(log_mag)))
        arithmetic_mean = float(np.mean(mag)) + eps
        return float(np.clip(geometric_mean / arithmetic_mean, 0.0, 1.0))

    def _spectral_flux(self, mag: np.ndarray, eps: float) -> float:
        """
        Normalised L2 distance from the previous frame's spectrum.
        Returns 0.0 for the very first frame (no previous frame available).
        """
        if self._prev_mag is None:
            return 0.0
        diff = mag - self._prev_mag
        return float(np.sqrt(np.dot(diff, diff)) / (len(mag) + eps))

    def _spectral_rolloff(self, mag: np.ndarray, rolloff_fraction: float = 0.85) -> float:
        """Frequency below which *rolloff_fraction* of spectral energy lies."""
        power = mag ** 2
        cumsum = np.cumsum(power)
        threshold = rolloff_fraction * cumsum[-1]
        idx = int(np.searchsorted(cumsum, threshold))
        idx = min(idx, len(self._bin_freqs) - 1)
        return float(self._bin_freqs[idx])

    # ------------------------------------------------------------------
    # 2. Harmonic
    # ------------------------------------------------------------------

    def _detect_pitch(self, windowed: np.ndarray) -> tuple[float, float]:
        """
        Autocorrelation-based pitch detection (YIN-inspired, simplified).

        Returns
        -------
        f0_hz : float  — 0.0 if no confident pitch found
        salience : float — normalised peak height [0, 1]
        """
        # Normalised autocorrelation via FFT for efficiency
        n = len(windowed)
        fft_size = 1 << (2 * n - 1).bit_length()   # next power of 2
        fft = np.fft.rfft(windowed, n=fft_size)
        ac_full = np.fft.irfft(fft * np.conj(fft))[:n].real
        ac = ac_full / (ac_full[0] + 1e-10)        # normalise to [0, 1]

        lo, hi = self._lag_min, min(self._lag_max, n - 1)
        peak_rel = int(np.argmax(ac[lo : hi + 1]))
        lag = peak_rel + lo
        salience = float(np.clip(ac[lag], 0.0, 1.0))

        # Require a minimum salience to avoid reporting pitch in noise
        if salience < 0.25:
            return 0.0, salience

        f0 = float(self._sr) / float(lag)
        return f0, salience

    def _compute_hnr(self, windowed: np.ndarray, f0: float, eps: float) -> float:
        """
        HNR via normalised autocorrelation at the pitch period.

        HNR = 10 * log10(r[T] / (1 − r[T]))  where r is normalised.
        Result clipped to [−10, 40] dB.
        """
        if f0 <= 0.0:
            return -10.0

        lag = int(round(self._sr / f0))
        if lag >= len(windowed):
            return -10.0

        n = len(windowed)
        fft_size = 1 << (2 * n - 1).bit_length()
        fft = np.fft.rfft(windowed, n=fft_size)
        ac = np.fft.irfft(fft * np.conj(fft))[:n].real
        ac = ac / (ac[0] + eps)

        r = float(ac[lag])
        denom = max(1.0 - r, eps)
        hnr_db = 10.0 * np.log10(max(r / denom, eps))
        return float(np.clip(hnr_db, -10.0, 40.0))

    def _compute_inharmonicity(
        self, mag: np.ndarray, f0: float, eps: float
    ) -> float:
        """
        Mean relative deviation of detected harmonic peaks from ideal f0 multiples.

        Trumpet: near 0 (bore physics enforces harmonic alignment).
        High values (>0.1) indicate inharmonic or non-pitched sources.
        """
        if f0 <= 0.0:
            return 0.5   # neutral — no pitch

        deviations: list[float] = []
        for h in range(2, 12):
            ideal_hz = h * f0
            if ideal_hz >= self._sr / 2.0:
                break
            ideal_bin = int(round(ideal_hz * self._N_FFT / self._sr))
            search = max(2, ideal_bin // 20)   # ±5 % of harmonic frequency
            lo = max(0, ideal_bin - search)
            hi = min(len(mag) - 1, ideal_bin + search)
            actual_bin = lo + int(np.argmax(mag[lo : hi + 1]))
            actual_hz = actual_bin * self._sr / self._N_FFT
            deviation = abs(actual_hz - ideal_hz) / (ideal_hz + eps)
            deviations.append(deviation)

        if not deviations:
            return 0.5
        return float(np.clip(np.mean(deviations), 0.0, 1.0))

    def _compute_odd_even_ratio(
        self, mag: np.ndarray, f0: float, eps: float
    ) -> float:
        """
        Ratio of odd-harmonic energy to even-harmonic energy.

        Trumpet (open conical resonator): both present → ratio near 1.
        Clarinet (closed cylinder): odd dominate → ratio >> 1.
        Returns 1.0 when no pitch is detected (neutral).
        """
        if f0 <= 0.0:
            return 1.0

        odd_e = 0.0
        even_e = 0.0
        for h in range(1, 14):
            hz = h * f0
            if hz >= self._sr / 2.0:
                break
            b = int(round(hz * self._N_FFT / self._sr))
            lo = max(0, b - 1)
            hi = min(len(mag) - 1, b + 1)
            energy = float(np.sum(mag[lo : hi + 1] ** 2))
            if h % 2 == 1:
                odd_e += energy
            else:
                even_e += energy

        return float(odd_e / (even_e + eps))

    # ------------------------------------------------------------------
    # 3. Cepstral
    # ------------------------------------------------------------------

    def _compute_mfcc(self, mag: np.ndarray, eps: float) -> np.ndarray:
        """
        MFCCs 1–13 from the power spectrum via triangular mel filterbank + DCT.

        We return coefficients starting at index 1 (skip c0 — it's just log
        energy and is redundant with rms_db).
        """
        power = mag ** 2
        mel_e = self._mel_fb @ power                      # (40,)
        log_mel = np.log(mel_e + eps)                     # (40,)
        cepstrum = scipy_dct(log_mel, type=2, norm="ortho")
        return cepstrum[1 : self._N_MFCC + 1].astype(np.float32)   # (13,)

    def _compute_delta_mfcc(self) -> np.ndarray:
        """
        Causal first-order delta MFCCs.

        Uses the centred difference over ±1 frame (Δ = (t − t−2) / 2).
        Returns zeros for the first two frames.

        Key property: near-zero for sustained trumpet notes (stable timbre),
        continuously non-zero for speech (vocal tract always moving).
        """
        if len(self._mfcc_history) < 3:
            return np.zeros(self._N_MFCC, dtype=np.float32)
        h = list(self._mfcc_history)
        return ((h[-1] - h[-3]) / 2.0).astype(np.float32)

    def _compute_mfcc_variance(self) -> np.ndarray:
        """Per-coefficient variance over the rolling MFCC window (max 7 frames)."""
        if len(self._mfcc_history) < 2:
            return np.zeros(self._N_MFCC, dtype=np.float32)
        hist = np.array(self._mfcc_history, dtype=np.float32)
        return hist.var(axis=0)

    # ------------------------------------------------------------------
    # 4. Formant / LPC
    # ------------------------------------------------------------------

    def _lpc_formants(
        self, windowed: np.ndarray
    ) -> tuple[int, bool, bool, bool]:
        """
        Fit an LPC model, extract resonant poles, check for speech formants.

        Speech has narrow-bandwidth poles in F1 (300–900 Hz), F2 (850–2500 Hz),
        F3 (2000–3500 Hz) because the vocal tract resonates at those frequencies.

        Trumpet physically *cannot* produce formant structure (no vocal tract),
        so this is the strongest speech-rejection feature.

        Edge case: Harmon/wah-wah mutes can produce resonance in 500–2000 Hz.
        The TAD state machine handles this by downweighting LPC rejection when
        harmonic features are strongly trumpet-positive (``muted_mode=True``).

        Returns
        -------
        (formant_count, has_f1, has_f2, has_f3)
        """
        # Pre-emphasis (6 dB/octave tilt) to flatten spectrum for LPC
        pre: np.ndarray = np.empty_like(windowed)
        pre[0] = windowed[0]
        pre[1:] = windowed[1:] - 0.97 * windowed[:-1]

        # Autocorrelation vector r[0..order]
        n = len(pre)
        ac_full = np.correlate(pre, pre, mode="full")
        r = ac_full[n - 1 : n + self._LPC_ORDER].astype(np.float64)

        if r[0] < 1e-10:
            return 0, False, False, False

        # Levinson-Durbin → LPC polynomial a[0..order] with a[0]=1
        a = self._levinson_durbin(r, self._LPC_ORDER)

        # Roots of the polynomial in the z-plane
        roots = np.roots(a)

        # Keep poles inside the unit circle with positive imaginary part
        mask = (np.abs(roots) < 1.0 - 1e-6) & (np.imag(roots) >= 0.0)
        roots = roots[mask]

        if len(roots) == 0:
            return 0, False, False, False

        # Convert to frequency (Hz) and bandwidth (Hz)
        angles = np.angle(roots)
        freqs = angles * self._sr / (2.0 * np.pi)
        bandwidths = -np.log(np.abs(roots) + 1e-10) * self._sr / np.pi

        # A pole is "speech-like" if its bandwidth is narrow
        bw = self._MAX_FORMANT_BW

        has_f1 = bool(
            np.any((freqs >= 300) & (freqs <= 900) & (bandwidths < bw))
        )
        has_f2 = bool(
            np.any((freqs >= 850) & (freqs <= 2_500) & (bandwidths < bw))
        )
        has_f3 = bool(
            np.any((freqs >= 2_000) & (freqs <= 3_500) & (bandwidths < bw))
        )

        return int(has_f1) + int(has_f2) + int(has_f3), has_f1, has_f2, has_f3

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
        """
        Levinson-Durbin recursion for real autocorrelation sequences.

        Parameters
        ----------
        r : np.ndarray
            Autocorrelation vector [r[0], r[1], ..., r[order]], length order+1.
        order : int
            LPC order p.

        Returns
        -------
        a : np.ndarray  shape (order+1,)
            LPC polynomial [1, a1, a2, ..., a_p].  Pass directly to
            ``np.roots()`` to obtain the z-plane poles.
        """
        a = np.zeros(order + 1, dtype=np.float64)
        a[0] = 1.0
        e = float(r[0])

        if e < 1e-10:
            return a

        for m in range(1, order + 1):
            # Reflection coefficient k_m
            # k_m = -(r[m] + sum_{j=1}^{m-1} a[j] * r[m-j]) / e
            if m > 1:
                # a[1:m] · r[m-1 : 0 : -1]  =  sum a[j]*r[m-j] for j=1..m-1
                dot = float(np.dot(a[1:m], r[m - 1 : 0 : -1]))
            else:
                dot = 0.0

            k = -(r[m] + dot) / e
            k = float(np.clip(k, -1.0 + 1e-12, 1.0 - 1e-12))

            # Update polynomial in-place (vectorised)
            # new_a[j] = a[j] + k * a[m-j]  for j = 1..m-1
            # new_a[m] = k
            prev = a[1:m].copy()
            a[1:m] = prev + k * prev[::-1]
            a[m] = k

            e *= 1.0 - k * k
            if e < 1e-10:
                break

        return a

    @staticmethod
    def _build_mel_filterbank(
        sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
    ) -> np.ndarray:
        """
        Build a triangular mel filterbank matrix of shape (n_mels, n_fft//2+1).

        Filters are normalised so each row sums to 1.0 (area normalisation),
        making the output invariant to filter spacing density.
        """

        def hz_to_mel(hz: float) -> float:
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(m: float) -> float:
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        n_bins = n_fft // 2 + 1
        bin_freqs = np.linspace(0.0, sr / 2.0, n_bins)

        # n_mels+2 evenly spaced mel points covering [fmin, fmax]
        mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        hz_points = np.array([mel_to_hz(m) for m in mel_points])

        fb = np.zeros((n_mels, n_bins), dtype=np.float32)
        for i in range(n_mels):
            left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
            width_l = center - left + 1e-10
            width_r = right - center + 1e-10
            # Rising slope
            mask_l = (bin_freqs > left) & (bin_freqs <= center)
            fb[i, mask_l] = (bin_freqs[mask_l] - left) / width_l
            # Falling slope
            mask_r = (bin_freqs > center) & (bin_freqs < right)
            fb[i, mask_r] = (right - bin_freqs[mask_r]) / width_r

        # Area normalisation
        row_sums = fb.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        fb /= row_sums

        return fb
