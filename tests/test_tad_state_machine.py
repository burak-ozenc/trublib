"""
tests/test_tad_state_machine.py
--------------------------------
Unit tests for TADStateMachine.

Coverage
~~~~~~~~
- All valid state transitions
- Asymmetric hysteresis (hard to start, slow to stop)
- TRAILING → ACTIVE recovery (no re-onset required)
- Retroactive flush: structure, alpha graduation, ordering
- Fade-out during TRAILING (no clicks)
- reset() clears all state
- TADResult fields on every transition
- Edge cases: confidence exactly at threshold, single-frame signals
"""

from __future__ import annotations

import numpy as np
import pytest

from trublib.tad_state_machine import TADResult, TADState, TADStateMachine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHUNK = 1_920
FRAME = 512
SR = 24_000

HIGH = 0.9    # above default threshold 0.6
LOW  = 0.1    # below threshold
AT   = 0.6    # exactly at threshold (should be treated as above: >= 0.6)


def make_audio(val: float = 0.5, n: int = CHUNK) -> np.ndarray:
    """Constant-valued audio chunk for testing."""
    return np.full(n, val, dtype=np.float32)


def make_machine(**kwargs) -> TADStateMachine:
    """Default machine: onset=3, trailing=4, threshold=0.6, lookback=3."""
    defaults = dict(onset_chunks=3, trailing_chunks=4, threshold=0.6, lookback_frames=3)
    defaults.update(kwargs)
    return TADStateMachine(**defaults)


def feed(machine: TADStateMachine, confidence: float, n_chunks: int = 1) -> TADResult:
    """Feed n identical chunks to the machine, return the last result."""
    result = None
    for _ in range(n_chunks):
        result = machine.update(
            confidence=confidence,
            raw_audio=make_audio(confidence),     # use confidence as signal value
            masked_audio=make_audio(confidence),
        )
    return result


def reach_active(machine: TADStateMachine) -> TADResult:
    """Drive machine to ACTIVE state and return the ACTIVE entry result."""
    return feed(machine, HIGH, n_chunks=machine._onset_chunks)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_starts_silent(self):
        m = make_machine()
        assert m.state == TADState.SILENT

    def test_first_result_silent(self):
        m = make_machine()
        r = feed(m, LOW)
        assert r.state == TADState.SILENT
        assert r.is_trumpet is False
        assert r.flush is None
        assert np.all(r.masked_audio == 0)


# ---------------------------------------------------------------------------
# SILENT → ONSET
# ---------------------------------------------------------------------------


class TestSilentToOnset:
    def test_high_confidence_enters_onset(self):
        m = make_machine()
        r = feed(m, HIGH)
        assert r.state == TADState.ONSET

    def test_at_threshold_enters_onset(self):
        """Threshold is inclusive (>= not >)."""
        m = make_machine()
        r = feed(m, AT)
        assert r.state == TADState.ONSET

    def test_onset_not_is_trumpet(self):
        m = make_machine()
        r = feed(m, HIGH)
        assert r.is_trumpet is False

    def test_onset_no_flush(self):
        m = make_machine()
        r = feed(m, HIGH)
        assert r.flush is None

    def test_onset_silent_output(self):
        """Output is muted during ONSET — not yet confirmed."""
        m = make_machine()
        r = feed(m, HIGH)
        assert np.all(r.masked_audio == 0)

    def test_low_confidence_stays_silent(self):
        m = make_machine()
        r = feed(m, LOW)
        assert r.state == TADState.SILENT


# ---------------------------------------------------------------------------
# ONSET → SILENT (evidence collapses)
# ---------------------------------------------------------------------------


class TestOnsetCollapse:
    def test_confidence_drop_resets_to_silent(self):
        m = make_machine()
        feed(m, HIGH)          # enter ONSET
        r = feed(m, LOW)       # evidence collapses
        assert r.state == TADState.SILENT

    def test_onset_count_resets_on_collapse(self):
        m = make_machine()
        feed(m, HIGH)
        feed(m, LOW)
        # Should need full onset_chunks again to reach ACTIVE
        feed(m, HIGH, n_chunks=2)
        assert m.state == TADState.ONSET   # not yet ACTIVE

    def test_partial_onset_does_not_activate(self):
        """onset_chunks - 1 high chunks followed by 1 low → stays SILENT."""
        m = make_machine(onset_chunks=3)
        feed(m, HIGH, n_chunks=2)   # 2 of 3 required
        r = feed(m, LOW)
        assert r.state == TADState.SILENT


# ---------------------------------------------------------------------------
# ONSET → ACTIVE
# ---------------------------------------------------------------------------


class TestOnsetToActive:
    def test_reaches_active_after_onset_chunks(self):
        m = make_machine(onset_chunks=3)
        r = reach_active(m)
        assert r.state == TADState.ACTIVE

    def test_active_is_trumpet(self):
        m = make_machine()
        r = reach_active(m)
        assert r.is_trumpet is True

    def test_active_returns_masked_audio(self):
        """masked_audio should be passed through during ACTIVE."""
        m = make_machine()
        r = reach_active(m)
        assert not np.all(r.masked_audio == 0)

    def test_flush_emitted_on_active_entry(self):
        """Flush must be a non-None list on ACTIVE entry."""
        m = make_machine()
        r = reach_active(m)
        assert r.flush is not None
        assert isinstance(r.flush, list)

    def test_flush_emitted_only_once(self):
        """Subsequent ACTIVE chunks must not re-emit flush."""
        m = make_machine()
        reach_active(m)
        r = feed(m, HIGH)    # second ACTIVE chunk
        assert r.flush is None

    def test_onset_count_resets_on_active(self):
        m = make_machine()
        reach_active(m)
        assert m._onset_count == 0


# ---------------------------------------------------------------------------
# Retroactive flush
# ---------------------------------------------------------------------------


class TestRetroactiveFlush:
    def test_flush_frames_are_float32(self):
        m = make_machine(lookback_frames=3)
        r = reach_active(m)
        for frame in r.flush:
            assert frame.dtype == np.float32

    def test_flush_not_empty(self):
        m = make_machine(lookback_frames=3)
        r = reach_active(m)
        assert len(r.flush) > 0

    def test_flush_discards_fully_attenuated_frames(self):
        """
        Frames with alpha=0 are discarded, not returned as silence.
        With lookback=3 and _FLUSH_ALPHAS=[0,0,0.3,0.6],
        the oldest (alpha=0) frame should not appear in flush.
        """
        m = make_machine(lookback_frames=3)
        r = reach_active(m)
        # With lookback=3 frames and alphas tail = [0.0, 0.3, 0.6]
        # Only alpha > 0 frames are returned: 2 frames max
        assert len(r.flush) <= 2

    def test_flush_alpha_increases_toward_newest(self):
        """
        Each successive flush frame should have higher or equal amplitude
        than the previous (graduated alpha means newer = stronger).
        """
        m = make_machine(lookback_frames=3)
        # Fill ring with identical audio so RMS difference = alpha difference
        for _ in range(10):
            m.update(HIGH, make_audio(1.0), make_audio(1.0))
        # Force one more to tip into ACTIVE
        m._state = TADState.SILENT
        m._onset_count = 0
        m.reset()

        # Fresh machine, feed identical audio
        m2 = make_machine(lookback_frames=3)
        for i in range(m2._onset_chunks):
            r = m2.update(HIGH, make_audio(1.0), make_audio(1.0))

        if r.flush and len(r.flush) >= 2:
            rms = [float(np.sqrt(np.mean(f**2))) for f in r.flush]
            for i in range(len(rms) - 1):
                assert rms[i] <= rms[i+1] + 1e-6, (
                    f"Flush frame {i} RMS ({rms[i]:.4f}) > frame {i+1} ({rms[i+1]:.4f})"
                )

    def test_flush_none_in_silent(self):
        m = make_machine()
        r = feed(m, LOW)
        assert r.flush is None

    def test_flush_none_in_onset(self):
        m = make_machine()
        r = feed(m, HIGH)
        assert r.flush is None

    def test_flush_none_in_trailing(self):
        m = make_machine()
        reach_active(m)
        r = feed(m, LOW)    # enter TRAILING
        assert r.flush is None


# ---------------------------------------------------------------------------
# ACTIVE → TRAILING
# ---------------------------------------------------------------------------


class TestActiveToTrailing:
    def test_one_low_chunk_enters_trailing(self):
        m = make_machine()
        reach_active(m)
        r = feed(m, LOW)
        assert r.state == TADState.TRAILING

    def test_trailing_not_is_trumpet(self):
        m = make_machine()
        reach_active(m)
        r = feed(m, LOW)
        assert r.is_trumpet is False

    def test_trailing_still_outputs_audio(self):
        """TRAILING should fade, not immediately silence."""
        m = make_machine()
        reach_active(m)
        r = feed(m, LOW)
        # First TRAILING chunk has some audio (gain > 0)
        assert not np.all(r.masked_audio == 0)

    def test_trailing_audio_fades(self):
        """Each successive TRAILING chunk should have lower RMS."""
        m = make_machine(trailing_chunks=4)
        reach_active(m)

        rms_values = []
        for _ in range(4):
            r = m.update(LOW, make_audio(1.0), make_audio(1.0))
            rms_values.append(float(np.sqrt(np.mean(r.masked_audio**2))))

        for i in range(len(rms_values) - 1):
            assert rms_values[i] >= rms_values[i+1], (
                f"RMS not fading: chunk {i}={rms_values[i]:.4f}, "
                f"chunk {i+1}={rms_values[i+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# TRAILING → ACTIVE (recovery)
# ---------------------------------------------------------------------------


class TestTrailingRecovery:
    def test_confidence_recovery_enters_active(self):
        """Confidence recovering during TRAILING → snap to ACTIVE."""
        m = make_machine()
        reach_active(m)
        feed(m, LOW)          # enter TRAILING
        r = feed(m, HIGH)     # recover
        assert r.state == TADState.ACTIVE

    def test_recovery_is_trumpet(self):
        m = make_machine()
        reach_active(m)
        feed(m, LOW)
        r = feed(m, HIGH)
        assert r.is_trumpet is True

    def test_recovery_no_flush(self):
        """Recovery into ACTIVE should NOT re-emit flush."""
        m = make_machine()
        reach_active(m)
        feed(m, LOW)
        r = feed(m, HIGH)
        assert r.flush is None

    def test_recovery_resets_trailing_count(self):
        m = make_machine()
        reach_active(m)
        feed(m, LOW)
        feed(m, HIGH)
        assert m._trailing_count == 0

    def test_recovery_after_multiple_trailing_chunks(self):
        """Recovery should work even after several TRAILING chunks."""
        m = make_machine(trailing_chunks=5)
        reach_active(m)
        feed(m, LOW, n_chunks=3)    # 3 trailing chunks (not expired)
        r = feed(m, HIGH)
        assert r.state == TADState.ACTIVE


# ---------------------------------------------------------------------------
# TRAILING → SILENT (expiry)
# ---------------------------------------------------------------------------


class TestTrailingExpiry:
    def test_trailing_expires_to_silent(self):
        m = make_machine(trailing_chunks=4)
        reach_active(m)
        r = feed(m, LOW, n_chunks=4)
        assert r.state == TADState.SILENT

    def test_trailing_count_resets_on_expiry(self):
        m = make_machine(trailing_chunks=4)
        reach_active(m)
        feed(m, LOW, n_chunks=4)
        assert m._trailing_count == 0

    def test_after_expiry_needs_full_onset_again(self):
        """After going SILENT via TRAILING, full onset_chunks needed again."""
        m = make_machine(onset_chunks=3, trailing_chunks=4)
        reach_active(m)
        feed(m, LOW, n_chunks=4)    # expire to SILENT
        feed(m, HIGH, n_chunks=2)   # only 2 of 3 onset chunks
        assert m.state == TADState.ONSET


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_from_active(self):
        m = make_machine()
        reach_active(m)
        m.reset()
        assert m.state == TADState.SILENT

    def test_reset_from_trailing(self):
        m = make_machine()
        reach_active(m)
        feed(m, LOW)
        m.reset()
        assert m.state == TADState.SILENT

    def test_reset_clears_onset_count(self):
        m = make_machine()
        feed(m, HIGH, n_chunks=2)
        m.reset()
        assert m._onset_count == 0

    def test_reset_clears_ring_buffer(self):
        m = make_machine()
        feed(m, HIGH, n_chunks=5)
        m.reset()
        assert len(m._ring) == 0

    def test_reset_clears_trailing_count(self):
        m = make_machine()
        reach_active(m)
        feed(m, LOW, n_chunks=2)
        m.reset()
        assert m._trailing_count == 0

    def test_full_cycle_after_reset(self):
        """After reset, the full onset → active cycle should work again."""
        m = make_machine()
        reach_active(m)
        m.reset()
        r = reach_active(m)
        assert r.state == TADState.ACTIVE


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_onset_chunks_one(self):
        """onset_chunks=1 → single high chunk enters ACTIVE immediately."""
        m = make_machine(onset_chunks=1)
        r = feed(m, HIGH)
        assert r.state == TADState.ACTIVE
        assert r.flush is not None

    def test_trailing_chunks_one(self):
        """trailing_chunks=1 → single low chunk exits to SILENT immediately."""
        m = make_machine(trailing_chunks=1)
        reach_active(m)
        r = feed(m, LOW)
        assert r.state == TADState.SILENT

    def test_confidence_exactly_at_threshold(self):
        """threshold=0.6, confidence=0.6 → treated as above (>=)."""
        m = make_machine(threshold=0.6)
        r = feed(m, 0.6)
        assert r.state == TADState.ONSET

    def test_zero_confidence_never_activates(self):
        m = make_machine()
        for _ in range(100):
            r = feed(m, 0.0)
        assert r.state == TADState.SILENT

    def test_full_confidence_activates_and_stays(self):
        m = make_machine()
        reach_active(m)
        for _ in range(50):
            r = feed(m, 1.0)
        assert r.state == TADState.ACTIVE

    def test_result_confidence_matches_input(self):
        """TADResult.confidence must echo the input confidence exactly."""
        m = make_machine()
        r = feed(m, 0.73)
        assert r.confidence == pytest.approx(0.73)

    def test_masked_audio_shape_preserved(self):
        """Output audio must always have the same shape as input."""
        m = make_machine()
        chunk = make_audio(n=CHUNK)
        r = m.update(HIGH, chunk, chunk)
        assert r.masked_audio.shape == chunk.shape

    def test_alternating_never_activates(self):
        """
        Alternating high/low should never reach ACTIVE with onset_chunks=3.
        High, Low, High, Low, ... → ONSET collapses every other chunk.
        """
        m = make_machine(onset_chunks=3)
        for _ in range(20):
            feed(m, HIGH)
            feed(m, LOW)
        assert m.state in (TADState.SILENT, TADState.ONSET)
        assert m.state != TADState.ACTIVE
