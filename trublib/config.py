"""
trublib.config
--------------
Public configuration surface for the TAD pipeline.  All parameters live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TADConfig:
    """
    Configuration for the Trumpet Activity Detection pipeline.

    Parameters
    ----------
    input_sample_rate : int
        Sample rate of the incoming audio stream.  Internally resampled to
        24 kHz (Moshi's native rate) when this differs from 24 000.
    threshold : float
        P(trumpet) cutoff in [0.0, 1.0].  1.0 = only near-certain trumpet
        passes; 0.0 = everything passes.  Also used as the ONSET trigger.
        Default 0.6 suits typical playing conditions.
    onset_chunks : int
        Number of consecutive chunks above *threshold* required to enter
        ACTIVE state (hysteresis guard, "hard to start").  Default 3 → 240 ms.
    trailing_chunks : int
        Number of consecutive chunks below *threshold* before ACTIVE →
        SILENT transition (hold-off, "slow to stop").  Default 4 → 320 ms.
    lookback_frames : int
        Depth of the retroactive ring buffer.  On ACTIVE entry, the last
        *lookback_frames* are flushed with graduated alpha masks so MERT
        sees the note attack.
    muted_mode : bool
        When True, LPC formant rejection is downweighted during frames where
        HNR and inharmonicity are strongly trumpet-positive.  Required for
        Harmon / wah-wah mutes that produce speech-like resonance peaks.
    target_rms : float
        Stage-1 normalisation target.  Input chunks are scaled to this RMS
        before feature extraction, eliminating mic/distance variance.
    """

    input_sample_rate: int = 44100
    threshold: float = 0.6
    onset_chunks: int = 3
    trailing_chunks: int = 4
    lookback_frames: int = 3
    muted_mode: bool = False
    target_rms: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.onset_chunks < 1:
            raise ValueError("onset_chunks must be >= 1")
        if self.trailing_chunks < 1:
            raise ValueError("trailing_chunks must be >= 1")
        if self.lookback_frames < 0:
            raise ValueError("lookback_frames must be >= 0")
        if self.target_rms <= 0:
            raise ValueError("target_rms must be > 0")
