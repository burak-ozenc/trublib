"""
trublib.frame_manager
---------------------
Ingests variable-length audio chunks from the microphone and emits
fixed-size, overlapping, Hann-windowed frames for feature extraction.

Design contract
~~~~~~~~~~~~~~~
- Frame size  : 512 samples  (21.3 ms @ 24 kHz)
- Hop size    : 256 samples  (10.7 ms — 50 % overlap)
- Window      : Hann
- Typical in  : 1920 samples per 80 ms chunk  → ~7 frames out
- center=False : no lookahead, no padding — streaming-safe

Not thread-safe.  Use one instance per audio stream.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Frame(NamedTuple):
    """A single analysis frame extracted from the audio stream."""

    windowed: np.ndarray
    """Hann-windowed float32 samples of length FRAME_SIZE.

    Feed this to :class:`~trublib.feature_extractor.FeatureExtractor`.
    """

    raw: np.ndarray
    """Un-windowed float32 samples of length FRAME_SIZE.

    Stored in the retroactive ring buffer; flushed on ACTIVE entry so
    MERT sees the note attack before classification completes.
    """

    index: int
    """Monotonically increasing frame counter (0-based, resets on :meth:`reset`)."""


class FrameManager:
    """
    Sliding-window frame extractor with a simple pending-sample queue.

    Usage
    -----
    ::

        fm = FrameManager()
        for chunk in mic_stream:           # each chunk: 1920 samples @ 24 kHz
            for frame in fm.push(chunk):   # ~7 Frame objects
                features = extractor.extract(frame)

    The internal buffer grows to at most ``FRAME_SIZE + len(chunk) - 1``
    samples between calls, which is negligible.
    """

    FRAME_SIZE: int = 512
    """Analysis window length in samples (21.3 ms @ 24 kHz)."""

    HOP_SIZE: int = 256
    """Step between successive frames in samples (10.7 ms)."""

    def __init__(self) -> None:
        self._window: np.ndarray = np.hanning(self.FRAME_SIZE).astype(np.float32)
        self._pending: np.ndarray = np.empty(0, dtype=np.float32)
        self._frame_index: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, samples: np.ndarray) -> list[Frame]:
        """
        Accept a chunk of mono float32 samples (any positive length).

        Appends *samples* to the internal pending buffer, then emits as
        many complete :class:`Frame` objects as possible.

        Returns an empty list if fewer than ``FRAME_SIZE`` samples have
        accumulated in total since the last frame was emitted.

        Parameters
        ----------
        samples : np.ndarray
            1-D array of mono float32 samples at 24 kHz (already resampled
            and mono-mixed by the pipeline layer above).
        """
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError(
                f"FrameManager expects 1-D mono audio, got shape {samples.shape}. "
                "Mix stereo to mono before calling push()."
            )

        self._pending = np.concatenate((self._pending, samples))

        frames: list[Frame] = []
        while len(self._pending) >= self.FRAME_SIZE:
            raw = self._pending[: self.FRAME_SIZE].copy()
            frames.append(
                Frame(
                    windowed=(raw * self._window),
                    raw=raw,
                    index=self._frame_index,
                )
            )
            self._frame_index += 1
            self._pending = self._pending[self.HOP_SIZE :]

        return frames

    def flush(self) -> list[Frame]:
        """
        Drain the pending buffer by zero-padding to a full frame.

        Call at end-of-stream or when the upstream has gone silent long
        enough that the ring buffer should be emptied.  Returns at most
        one frame (the zero-padded remainder).  Returns an empty list if
        the buffer is already empty.
        """
        if len(self._pending) == 0:
            return []

        raw = np.zeros(self.FRAME_SIZE, dtype=np.float32)
        raw[: len(self._pending)] = self._pending
        frame = Frame(
            windowed=(raw * self._window),
            raw=raw,
            index=self._frame_index,
        )
        self._frame_index += 1
        self._pending = np.empty(0, dtype=np.float32)
        return [frame]

    def reset(self) -> None:
        """Clear all internal state (e.g. on stream restart)."""
        self._pending = np.empty(0, dtype=np.float32)
        self._frame_index = 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def pending_samples(self) -> int:
        """Samples waiting in the buffer (useful for latency estimation)."""
        return len(self._pending)

    @property
    def frame_index(self) -> int:
        """Total frames emitted since construction or last :meth:`reset`."""
        return self._frame_index

    @property
    def frames_per_chunk(self) -> float:
        """
        Expected frames for a standard 1920-sample (80 ms) chunk at steady
        state.  At 50 % overlap: (1920 + 256) / 256 ≈ 8.5, settling to ~7
        once the overlap region fills.
        """
        return (1920 - self.FRAME_SIZE) / self.HOP_SIZE + 1
