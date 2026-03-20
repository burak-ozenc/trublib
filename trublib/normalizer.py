"""
trublib.normalizer
------------------
Two-stage chunk normaliser.

Stage 1 — RMS normalise to *target_rms* (eliminates mic/distance variance).
Stage 2 — capture raw RMS in dB *before* normalising and return it as an
           explicit feature passed to FeatureExtractor.rms_db_override.

Why two stages?  If we normalise first, the loudness information is gone.
A whispering beginner and a fortissimo advanced player would look identical.
Capturing it first preserves the dynamic range cue for the classifier.
"""

from __future__ import annotations

import numpy as np


class TwoStageNormalizer:
    """
    Normalise a mono audio chunk and report its pre-normalisation loudness.

    Parameters
    ----------
    target_rms : float
        Desired RMS of the output signal (default 0.1).
    silence_floor_db : float
        Any input quieter than this (dB) is returned as-is with no
        gain applied.  Prevents extreme gain on silence / breath.
    """

    def __init__(
        self,
        target_rms: float = 0.1,
        silence_floor_db: float = -60.0,
    ) -> None:
        self._target_rms = float(target_rms)
        self._silence_floor = float(silence_floor_db)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, samples: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Normalise *samples* and return the raw loudness.

        Parameters
        ----------
        samples : np.ndarray
            1-D float32 audio chunk at any amplitude.

        Returns
        -------
        normalised : np.ndarray
            Stage-1 RMS-normalised copy of *samples*.
        rms_db : float
            Raw RMS in dB (floor-clamped to *silence_floor_db*).
            Pass this value to ``FeatureExtractor`` so the classifier
            sees the actual playing loudness.
        """
        samples = np.asarray(samples, dtype=np.float32)

        rms = float(np.sqrt(np.mean(samples ** 2)))
        rms_db = float(20.0 * np.log10(max(rms, 1e-10)))
        rms_db = max(rms_db, self._silence_floor)

        # Below silence floor — return unchanged to avoid explosive gain
        if rms_db <= self._silence_floor:
            return samples.copy(), rms_db

        scale = self._target_rms / rms
        return (samples * scale).astype(np.float32), rms_db
