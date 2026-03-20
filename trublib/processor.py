"""
trublib.processor
-----------------
Top-level TAD pipeline.  This is the single entry point for TRUB.AI.

    tad = TADProcessor(TADConfig())
    result = tad.process(chunk)     # chunk: np.ndarray[N] at input_sample_rate

Pipeline (per 80ms chunk)
~~~~~~~~~~~~~~~~~~~~~~~~~~
    [raw chunk @ input_sr]
           ↓
    [Resampler]            input_sr → 24kHz  (soxr, high quality)
           ↓
    [Mono enforcement]     stereo → mono mix-down
           ↓
    [TwoStageNormalizer]   Stage 1: RMS → target_rms
                           Stage 2: capture rms_db before normalising
           ↓
    [FrameManager]         ring buffer → Hann-windowed 512-sample frames
           ↓
    [FeatureExtractor]     53-dim vector per frame (4 feature families)
           ↓
    [TrumpetScorer]        stats-pool frames → 106-dim → ONNX → P(trumpet)
           ↓
    [SoftMaskGenerator]    STFT → P(trumpet) × bins → ISTFT
           ↓
    [TADStateMachine]      confidence gate + hysteresis + retroactive flush
           ↓
    [TADResult]            .masked_audio / .state / .is_trumpet
                           .confidence / .flush

Latency contract: complete processing within 80ms so Moshi never waits.
Typical wall time: 5–15ms per chunk on modern hardware (pure Python/NumPy).

Not thread-safe.  One instance per audio stream.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from trublib.config import TADConfig
from trublib.feature_extractor import FeatureExtractor
from trublib.frame_manager import FrameManager
from trublib.normalizer import TwoStageNormalizer
from trublib.soft_mask import SoftMaskGenerator
from trublib.tad_state_machine import TADResult, TADStateMachine
from trublib.trumpet_scorer import TrumpetScorer

log = logging.getLogger(__name__)

_INTERNAL_SR = 24_000   # Moshi's native rate — all processing happens here
_CHUNK_SAMPLES = 1_920  # 80ms @ 24kHz


class TADProcessor:
    """
    Full Trumpet Activity Detection pipeline.

    Parameters
    ----------
    config : TADConfig
        All tunable parameters.  See :class:`~trublib.config.TADConfig`.
    model_path : Path | None
        Explicit path to a custom .onnx model.  When None, the bundled
        model inside the trublib package is used.

    Examples
    --------
    ::

        from trublib import TADProcessor, TADConfig

        tad = TADProcessor(TADConfig(input_sample_rate=44100, threshold=0.6))

        for chunk in mic_stream:                # 1920 samples @ 44100 Hz
            result = tad.process(chunk)

            if result.flush:
                for frame in result.flush:
                    moshi.send(frame)           # retroactive note attack

            if result.is_trumpet:
                moshi.send(result.masked_audio)

            print(result.state, result.confidence)
    """

    def __init__(
            self,
            config: Optional[TADConfig] = None,
            model_path: Optional[Path] = None,
    ) -> None:
        self._config = config or TADConfig()

        # Resampler — built lazily on first call if input_sr != internal_sr
        self._resampler = None
        self._needs_resample = self._config.input_sample_rate != _INTERNAL_SR
        if self._needs_resample:
            self._resampler = _build_resampler(
                self._config.input_sample_rate, _INTERNAL_SR
            )

        self._normalizer = TwoStageNormalizer(
            target_rms=self._config.target_rms
        )
        self._frame_manager = FrameManager()
        self._extractor = FeatureExtractor(sr=_INTERNAL_SR)
        self._scorer = TrumpetScorer(model_path=model_path)
        self._mask_gen = SoftMaskGenerator(sr=_INTERNAL_SR)
        self._state_machine = TADStateMachine(
            onset_chunks=self._config.onset_chunks,
            trailing_chunks=self._config.trailing_chunks,
            threshold=self._config.threshold,
            lookback_frames=self._config.lookback_frames,
            muted_mode=self._config.muted_mode,
        )

        log.info(
            "TADProcessor ready — sr=%d→%d, threshold=%.2f, "
            "onset=%d, trailing=%d",
            self._config.input_sample_rate, _INTERNAL_SR,
            self._config.threshold,
            self._config.onset_chunks,
            self._config.trailing_chunks,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, chunk: np.ndarray) -> TADResult:
        """
        Process one chunk of raw microphone audio.

        Parameters
        ----------
        chunk : np.ndarray
            1-D or 2-D (stereo) float32 audio at ``config.input_sample_rate``.
            Typically 1920 samples for an 80ms chunk at 24kHz, or proportionally
            more at higher input sample rates.

        Returns
        -------
        TADResult
            See :class:`~trublib.tad_state_machine.TADResult`.
        """
        chunk = np.asarray(chunk, dtype=np.float32)

        # Step 1 — Mono enforcement
        if chunk.ndim == 2:
            chunk = chunk.mean(axis=1)
        elif chunk.ndim != 1:
            raise ValueError(
                f"chunk must be 1-D (mono) or 2-D (stereo), got shape {chunk.shape}"
            )

        # Step 2 — Resample to 24kHz if needed
        if self._needs_resample:
            chunk = self._resampler(chunk)

        # Step 3 — Two-stage normalisation
        # normalised  : used ONLY for feature extraction (Steps 4–5)
        # chunk       : resampled but un-normalised — used for audio output
        #
        # Why separate?  TwoStageNormalizer amplifies every chunk to
        # target_rms=0.1 for level-invariant feature extraction.  Passing
        # normalised through the gate as output audio amplifies quiet
        # inter-note passages (breath, room tone) to -20 dBFS, creating
        # audible white-noise bursts during ACTIVE state.  The output should
        # preserve the original dynamics — only gating, not level changes.
        normalised, rms_db = self._normalizer.process(chunk)

        # Step 4 — Frame extraction + feature extraction
        frames = self._frame_manager.push(normalised)
        feature_vectors = [
            self._extractor.extract(f, rms_db_override=rms_db)
            for f in frames
        ]

        # Step 5 — Score: P(trumpet) for this chunk
        confidence = self._scorer.score(feature_vectors)

        # Step 6 — State machine (decides ACTIVE/ONSET/TRAILING/SILENT)
        # We pass chunk (un-normalised) as the raw_audio stored in the ring
        # buffer so that retroactive flush frames carry original dynamics.
        sm_result = self._state_machine.update(
            confidence=confidence,
            raw_audio=chunk,
            masked_audio=chunk,  # placeholder, overwritten below
        )

        # Step 7 — Gate: apply mask gain to the original un-normalised chunk.
        #   ACTIVE   → gain=1.0  → chunk passes through unchanged
        #   TRAILING → fade from 1.0 toward 0 over trailing_chunks
        #   ONSET/SILENT → gain=0.0 → silence
        state = sm_result.state
        if state.value == "active":
            mask_gain = 1.0
        elif state.value == "trailing":
            tc = self._state_machine._trailing_count
            total = self._config.trailing_chunks
            mask_gain = max(0.0, 1.0 - (tc / total))
        else:
            mask_gain = 0.0

        masked_audio = self._mask_gen.apply(chunk, mask_gain)

        # Rebuild result with correct masked_audio
        from trublib.tad_state_machine import TADResult
        result = TADResult(
            masked_audio=masked_audio,
            state=sm_result.state,
            is_trumpet=sm_result.is_trumpet,
            confidence=confidence,
            flush=sm_result.flush,
        )

        return result

    def reset(self) -> None:
        """
        Reset all internal state.

        Call this when the audio stream is interrupted or restarted —
        clears the ring buffer, frame queue, MFCC history, and state machine.
        """
        self._frame_manager.reset()
        self._extractor.reset()
        self._scorer.reset()
        self._state_machine.reset()
        if self._needs_resample:
            self._resampler = _build_resampler(
                self._config.input_sample_rate, _INTERNAL_SR
            )
        log.debug("TADProcessor reset")

    @property
    def config(self) -> TADConfig:
        return self._config

    @property
    def state(self):
        return self._state_machine.state


# ---------------------------------------------------------------------------
# Resampler factory
# ---------------------------------------------------------------------------


def _build_resampler(in_sr: int, out_sr: int):
    """
    Build a stateful resampler function using soxr (preferred) or resampy.

    soxr is faster and has better phase accuracy — preferred for real-time.
    resampy is a fallback for environments where soxr is not available.

    Returns a callable: (chunk: np.ndarray) -> np.ndarray
    """
    try:
        import soxr

        def _resample_soxr(chunk: np.ndarray) -> np.ndarray:
            return soxr.resample(
                chunk, in_sr, out_sr, quality="HQ"
            ).astype(np.float32)

        log.debug("Resampler: soxr (HQ) %d→%d", in_sr, out_sr)
        return _resample_soxr

    except ImportError:
        pass

    try:
        import resampy

        def _resample_resampy(chunk: np.ndarray) -> np.ndarray:
            return resampy.resample(
                chunk, in_sr, out_sr, filter="kaiser_best"
            ).astype(np.float32)

        log.debug("Resampler: resampy (kaiser_best) %d→%d", in_sr, out_sr)
        return _resample_resampy

    except ImportError:
        pass

    raise ImportError(
        "No resampler found.  Install soxr (recommended): pip install soxr\n"
        "Or install resampy as a fallback: pip install resampy"
    )