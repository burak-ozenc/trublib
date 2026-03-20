"""
trublib.soft_mask
-----------------
Applies P(trumpet) as a per-bin gain multiplier in the STFT domain,
then reconstructs clean audio via ISTFT.

Why a soft mask instead of a hard gate?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A hard gate (multiply by 0 or 1) introduces clicks and discontinuities
at state transitions.  A soft mask scales each STFT bin by P(trumpet),
so the gain ramps smoothly between 0 and 1 as confidence changes.

P(trumpet) = 0.0  → all bins zeroed (silence)
P(trumpet) = 0.5  → all bins at half amplitude
P(trumpet) = 1.0  → all bins pass through unchanged

The mask is uniform across all frequency bins.  A future version could
apply a frequency-dependent mask (e.g. stronger suppression of low-energy
bins), but uniform masking is sufficient and avoids spectral colouring.

STFT parameters (fixed — must match training and the architecture doc)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_fft      = 1024     42.7ms window, 23.4 Hz/bin @ 24kHz
    hop_length = 256      10.7ms step
    window     = Hann
    center     = False    streaming-safe, no lookahead, no padding

Note: the feature extractor uses a 512-point FFT on individual frames.
This is a separate 1024-point STFT used only for masking and reconstruction.

Overlap-add reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~
With center=False and 50% overlap (hop=256, n_fft=1024 would be 50% if
hop=512; here hop=256 is 25% of n_fft), we use scipy.signal.istft which
handles the overlap-add correctly.

Not thread-safe.  One instance per audio stream.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import istft, stft


class SoftMaskGenerator:
    """
    Applies a uniform per-bin soft mask to audio using STFT/ISTFT.

    Parameters
    ----------
    sr : int
        Sample rate.  Must be 24000 (Moshi's native rate).
    n_fft : int
        FFT size.  Fixed at 1024 per architecture specification.
    hop_length : int
        STFT hop size in samples.  Fixed at 256.
    min_gain : float
        Minimum gain applied even when P(trumpet) = 0.0.
        Default 0.0 (full silence).  Set to e.g. 0.05 if you want a
        faint background bleed for naturalness.
    """

    N_FFT      : int = 1024
    HOP_LENGTH : int = 256

    def __init__(
            self,
            sr: int = 24_000,
            n_fft: int = 1024,
            hop_length: int = 256,
            min_gain: float = 0.0,
    ) -> None:
        if n_fft != self.N_FFT or hop_length != self.HOP_LENGTH:
            raise ValueError(
                f"STFT parameters are fixed at n_fft={self.N_FFT}, "
                f"hop_length={self.HOP_LENGTH} per the architecture spec. "
                f"Got n_fft={n_fft}, hop_length={hop_length}."
            )
        self._sr         = sr
        self._n_fft      = n_fft
        self._hop_length = hop_length
        self._min_gain   = float(min_gain)
        self._window     = "hann"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, audio: np.ndarray, confidence: float) -> np.ndarray:
        """
        Apply a uniform soft mask of *confidence* to *audio*.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 mono audio chunk (typically 1920 samples / 80ms).
        confidence : float
            P(trumpet) in [0.0, 1.0] from TrumpetScorer.

        Returns
        -------
        np.ndarray
            Masked audio, same shape and dtype as input.

        Implementation note — why time-domain, not STFT
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        The current mask is uniform across **all** frequency bins.
        Multiplying every STFT bin by a scalar k is mathematically
        identical to multiplying the time-domain signal by k:

            ISTFT(k · STFT(x)) == k · x

        Running STFT→multiply→ISTFT is therefore unnecessary, and
        scipy's ISTFT with ``center=False / boundary=None`` on a
        1920-sample chunk only reconstructs 1792 samples — the last
        128 samples (5.3 ms) are attenuated or silent.  Concatenating
        these truncated chunks produces audible 5 ms dropouts every
        80 ms.

        Time-domain multiplication avoids all of this with no loss of
        correctness.  The STFT path (below, currently unreachable) is
        preserved for future per-bin frequency-dependent masking.
        """
        audio = np.asarray(audio, dtype=np.float32)
        gain = float(np.clip(confidence, self._min_gain, 1.0))

        # ── Time-domain path (uniform gain) ─────────────────────────────
        # Covers all three TAD states:
        #   ACTIVE   → gain = 1.0  → identity copy, no processing cost
        #   TRAILING → 0 < gain < 1 → linear attenuation, no STFT artifacts
        #   SILENT/ONSET → gain = 0.0 → silence, no processing cost
        if gain >= 1.0:
            return audio.copy()
        if gain <= 0.0:
            return np.zeros_like(audio)
        return (audio * gain).astype(np.float32)

        # ── STFT path (frequency-dependent masking, not yet used) ────────
        # Kept for future per-bin masking (e.g. harmonic-only pass).
        # Do not remove — this is the correct implementation once the
        # mask becomes frequency-dependent.
        # pylint: disable=unreachable
        _, _, Zxx = stft(
            audio,
            fs=self._sr,
            window=self._window,
            nperseg=self._n_fft,
            noverlap=self._n_fft - self._hop_length,
            boundary=None,
            padded=False,
        )
        Zxx_masked = Zxx * gain
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="NOLA condition")
            _, reconstructed = istft(
                Zxx_masked,
                fs=self._sr,
                window=self._window,
                nperseg=self._n_fft,
                noverlap=self._n_fft - self._hop_length,
                boundary=None,
            )
        out = self._match_length(reconstructed, len(audio))
        return out.astype(np.float32)

    def apply_with_fade(
            self,
            audio: np.ndarray,
            gain_start: float,
            gain_end: float,
    ) -> np.ndarray:
        """
        Apply a linearly interpolated gain ramp across the chunk.

        Used during TRAILING to fade out smoothly across a single chunk
        rather than applying a constant gain for the whole chunk.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 mono audio chunk.
        gain_start : float
            Gain at the first sample (0.0–1.0).
        gain_end : float
            Gain at the last sample (0.0–1.0).
        """
        audio = np.asarray(audio, dtype=np.float32)
        ramp = np.linspace(gain_start, gain_end, len(audio), dtype=np.float32)
        return (audio * ramp).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _match_length(signal: np.ndarray, target_len: int) -> np.ndarray:
        """
        Trim or zero-pad *signal* to exactly *target_len* samples.

        ISTFT output length can differ from input by a few samples due to
        window boundary effects.  This makes the output deterministic.
        """
        n = len(signal)
        if n >= target_len:
            return signal[:target_len]
        # Zero-pad
        out = np.zeros(target_len, dtype=np.float32)
        out[:n] = signal
        return out