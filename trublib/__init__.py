"""
trublib — Trumpet Activity Detection for real-time audio streams.

Milestone 1 exports
-------------------
FrameManager, Frame         — sliding-window frame extraction
FeatureExtractor, FeatureVector — full 4-family feature pipeline
TwoStageNormalizer          — RMS normalisation + loudness capture
TADConfig                   — public configuration surface

Not yet implemented (v0.1)
--------------------------
TrumpetScorer, TADStateMachine, SoftMaskGenerator — coming in next milestones.
"""

from trublib.config import TADConfig
from trublib.feature_extractor import FeatureExtractor, FeatureVector
from trublib.frame_manager import Frame, FrameManager
from trublib.normalizer import TwoStageNormalizer
from trublib.trumpet_scorer import TrumpetScorer
from trublib.tad_state_machine import TADState, TADStateMachine, TADResult
from trublib.soft_mask import SoftMaskGenerator
from trublib.processor import TADProcessor

__all__ = [
    "TADConfig",
    "FeatureExtractor",
    "FeatureVector",
    "Frame",
    "FrameManager",
    "TwoStageNormalizer",
    "TrumpetScorer",
    "TADState",
    "TADStateMachine",
    "TADResult",
    "SoftMaskGenerator",
    "TADProcessor",
]
