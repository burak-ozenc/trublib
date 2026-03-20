"""
tests/test_frame_manager.py
---------------------------
Unit tests for FrameManager.  Validates mechanical correctness:
windowing, frame count, overlap, reset, flush.
"""

from __future__ import annotations

import numpy as np
import pytest

from trublib.frame_manager import Frame, FrameManager
from tests.conftest import CHUNK, FRAME, HOP, SR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def feed_chunks(fm: FrameManager, signal: np.ndarray, chunk_size: int = CHUNK) -> list[Frame]:
    frames = []
    for start in range(0, len(signal), chunk_size):
        frames.extend(fm.push(signal[start : start + chunk_size]))
    return frames


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFrameManagerBasic:
    def test_frame_size(self):
        """Every emitted frame is exactly FRAME_SIZE samples."""
        fm = FrameManager()
        signal = np.random.randn(SR).astype(np.float32)
        for frame in feed_chunks(fm, signal):
            assert frame.windowed.shape == (FRAME,)
            assert frame.raw.shape == (FRAME,)

    def test_frame_dtype(self):
        fm = FrameManager()
        frames = fm.push(np.ones(CHUNK, dtype=np.float32))
        assert frames[0].windowed.dtype == np.float32
        assert frames[0].raw.dtype == np.float32

    def test_frame_index_monotonic(self):
        fm = FrameManager()
        signal = np.random.randn(SR).astype(np.float32)
        frames = feed_chunks(fm, signal)
        indices = [f.index for f in frames]
        assert indices == list(range(len(indices)))

    def test_reset_clears_index(self):
        fm = FrameManager()
        fm.push(np.zeros(CHUNK, dtype=np.float32))
        fm.reset()
        assert fm.frame_index == 0
        assert fm.pending_samples == 0

    def test_stereo_raises(self):
        fm = FrameManager()
        with pytest.raises(ValueError, match="1-D"):
            fm.push(np.zeros((CHUNK, 2), dtype=np.float32))


class TestFrameCount:
    def test_steady_state_chunk_yields_eight_frames(self):
        """
        At steady state (after a priming chunk), a 1920-sample chunk should
        yield 8 frames.

        Derivation:
          chunk 1: 1920 samples → floor((1920 - 512) / 256) + 1 = 6 frames
                   leftover = 1920 - 6*256 = 384 samples
          chunk 2: 384 + 1920 = 2304 samples
                   frames = floor((2304 - 512) / 256) + 1 = 8
        """
        fm = FrameManager()
        fm.push(np.zeros(CHUNK, dtype=np.float32))  # prime — fills overlap
        frames = fm.push(np.zeros(CHUNK, dtype=np.float32))
        assert len(frames) == 8

    def test_total_frames_for_one_second(self):
        """One second of audio at 1920-sample chunks → ~93 frames."""
        fm = FrameManager()
        signal = np.random.randn(SR).astype(np.float32)
        frames = feed_chunks(fm, signal, chunk_size=CHUNK)
        # Expected: (24000 - 512) / 256 + 1 ≈ 92.6 → 92 complete frames
        assert len(frames) >= 90

    def test_small_chunks_accumulate(self):
        """Push 100 samples 20 times → get frames only after enough accumulate."""
        fm = FrameManager()
        all_frames = []
        for _ in range(20):
            all_frames.extend(fm.push(np.zeros(100, dtype=np.float32)))
        # 100 * 20 = 2000 samples total, should yield at least 5 frames
        assert len(all_frames) >= 5


class TestHannWindow:
    def test_windowed_edges_near_zero(self):
        """Hann window → first and last samples should be near zero."""
        fm = FrameManager()
        signal = np.ones(CHUNK * 2, dtype=np.float32)
        frames = feed_chunks(fm, signal)
        for frame in frames[:3]:
            assert abs(frame.windowed[0]) < 0.01
            assert abs(frame.windowed[-1]) < 0.01

    def test_windowed_peak_near_center(self):
        """Hann window peak is at the centre of the frame."""
        fm = FrameManager()
        signal = np.ones(CHUNK * 2, dtype=np.float32)
        frames = feed_chunks(fm, signal)
        for frame in frames[:3]:
            peak = int(np.argmax(frame.windowed))
            assert FRAME // 2 - 5 <= peak <= FRAME // 2 + 5

    def test_raw_unaffected_by_window(self):
        """raw must be the unmodified samples (no windowing applied)."""
        fm = FrameManager()
        signal = np.ones(CHUNK * 2, dtype=np.float32)
        frames = feed_chunks(fm, signal)
        for frame in frames[:5]:
            np.testing.assert_array_equal(frame.raw, np.ones(FRAME, dtype=np.float32))

    def test_window_does_not_modify_original(self):
        """push() must not mutate the input array."""
        fm = FrameManager()
        chunk = np.ones(CHUNK, dtype=np.float32)
        original = chunk.copy()
        fm.push(chunk)
        np.testing.assert_array_equal(chunk, original)


class TestOverlap:
    def test_consecutive_frames_share_samples(self):
        """
        Frame i+1 should share HOP_SIZE fewer new samples than frame i.
        Verified by checking that frame.raw[HOP:] of frame i equals frame.raw[:-HOP] of frame i+1.
        """
        fm = FrameManager()
        signal = np.arange(CHUNK * 2, dtype=np.float32)
        frames = feed_chunks(fm, signal)
        # Check first few consecutive pairs
        for i in range(min(3, len(frames) - 1)):
            np.testing.assert_array_equal(
                frames[i].raw[HOP:],
                frames[i + 1].raw[:-HOP],
            )


class TestFlush:
    def test_flush_empty_buffer_returns_empty(self):
        fm = FrameManager()
        assert fm.flush() == []

    def test_flush_after_push_drains_remainder(self):
        fm = FrameManager()
        # Push exactly FRAME samples — 1 full frame emitted, buffer empty
        fm.push(np.zeros(FRAME, dtype=np.float32))
        # Push a partial chunk
        fm.push(np.zeros(100, dtype=np.float32))
        flushed = fm.flush()
        assert len(flushed) == 1
        assert flushed[0].windowed.shape == (FRAME,)
        # Buffer should now be empty
        assert fm.pending_samples == 0
        assert fm.flush() == []

    def test_flush_zero_pads_correctly(self):
        fm = FrameManager()
        fm.push(np.ones(200, dtype=np.float32))
        frame = fm.flush()[0]
        # First 200 samples should be 1, rest should be 0 (before windowing)
        np.testing.assert_array_equal(frame.raw[:200], np.ones(200))
        np.testing.assert_array_equal(frame.raw[200:], np.zeros(FRAME - 200))
