"""
trublib.tad_state_machine
--------------------------
Implements the four-state TAD state machine with asymmetric hysteresis,
retroactive ring buffer flush, and soft mask gain fade during TRAILING.

States
~~~~~~

    SILENT → ONSET → ACTIVE → TRAILING → SILENT
                  ↘ (confidence drops before onset_chunks) → SILENT
                                  TRAILING → ACTIVE (confidence recovers)

SILENT
    Output muted.  Raw frames written to ring buffer silently.
    Nothing sent to Moshi.

ONSET
    P(trumpet) ≥ threshold for first chunk.  Evidence accumulates over
    onset_chunks consecutive chunks.  If confidence drops before
    onset_chunks reached → back to SILENT.  "Hard to start."

ACTIVE
    onset_chunks consecutive chunks confirmed.  Retroactive flush triggered
    once on entry: ring buffer frames sent back to caller with graduated
    alpha masks so MERT sees the note attack.  Clean masked audio streamed
    to Moshi.  is_trumpet = True.

TRAILING
    P drops below threshold.  Hold-off for trailing_chunks before going
    SILENT.  Soft mask gain fades toward 0 during hold-off.  If P recovers
    during TRAILING → snap back to ACTIVE without re-entering ONSET.
    "Slow to stop."

Asymmetry
~~~~~~~~~
    onset_chunks  (default 3) chunks to enter ACTIVE  — hard to start
    1 chunk below threshold to enter TRAILING           — immediate
    trailing_chunks (default 4) chunks to exit          — slow to stop

Retroactive buffer
~~~~~~~~~~~~~~~~~~
    On ACTIVE entry, the last lookback_frames raw frames are flushed with
    graduated alpha so MERT receives the note attack:

        t-N .. t-2   alpha = 0.0  (discarded — too uncertain)
        t-1          alpha = 0.3  (soft)
        t            alpha = 0.6  (mid)
        t+1 onwards  alpha = 1.0  (live, full)

    The flush is emitted via TADResult.flush as a list of np.ndarray.
    Caller pattern:

        result = machine.update(confidence, raw_frame)
        if result.flush:
            for frame in result.flush:
                send_to_moshi(frame)
        if result.is_trumpet:
            send_to_moshi(result.masked_audio)

Not thread-safe.  One instance per audio stream.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class TADState(str, Enum):
    """The four TAD states, serialisable as plain strings."""
    SILENT   = "silent"
    ONSET    = "onset"
    ACTIVE   = "active"
    TRAILING = "trailing"


@dataclass
class TADResult:
    """
    Output of :meth:`TADStateMachine.update` for one 80ms chunk.

    Caller should:
    1. If ``flush`` is not None, send each frame in order to Moshi first.
    2. If ``is_trumpet`` is True, send ``masked_audio`` to Moshi.
    3. Always read ``state``, ``confidence`` for UI / diagnostics.
    """

    masked_audio: np.ndarray
    """
    Clean audio with soft mask applied (or silence if not in ACTIVE).
    Shape matches the input chunk passed to :meth:`TADStateMachine.update`.
    """

    state: TADState
    """Current state after processing this chunk."""

    is_trumpet: bool
    """True only when state is ACTIVE."""

    confidence: float
    """Raw P(trumpet) passed into this update call."""

    flush: Optional[list[np.ndarray]]
    """
    Retroactive frames emitted on ACTIVE entry only.  None at all other times.
    Each array is a raw frame (512 samples) multiplied by its graduated alpha.
    Send these to Moshi *before* masked_audio on the ACTIVE entry chunk.
    """


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class TADStateMachine:
    """
    Confidence gate + hysteresis state machine for Trumpet Activity Detection.

    Parameters
    ----------
    onset_chunks : int
        Consecutive chunks above threshold required to enter ACTIVE.
    trailing_chunks : int
        Consecutive chunks below threshold required to exit ACTIVE.
    threshold : float
        P(trumpet) cutoff in [0, 1].
    lookback_frames : int
        Number of raw frames kept in the retroactive ring buffer.
    muted_mode : bool
        When True, LPC-based formant rejection is softer (handled upstream
        in FeatureExtractor; the state machine itself is unaffected).
    """

    # Graduated alpha for retroactive flush frames
    # Index 0 = oldest (t - lookback_frames + 1), last = newest (t)
    _FLUSH_ALPHAS = [0.0, 0.0, 0.3, 0.6]   # last 4 frames; pad with 0 if more

    def __init__(
        self,
        onset_chunks: int = 3,
        trailing_chunks: int = 4,
        threshold: float = 0.6,
        lookback_frames: int = 3,
        muted_mode: bool = False,
    ) -> None:
        self._onset_chunks    = onset_chunks
        self._trailing_chunks = trailing_chunks
        self._threshold       = threshold
        self._lookback        = lookback_frames
        self._muted_mode      = muted_mode

        # Ring buffer: stores raw frames for retroactive flush
        self._ring: deque[np.ndarray] = deque(maxlen=lookback_frames)

        # State
        self._state          = TADState.SILENT
        self._onset_count    = 0   # consecutive chunks above threshold in ONSET
        self._trailing_count = 0   # consecutive chunks below threshold in TRAILING

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        confidence: float,
        raw_audio: np.ndarray,
        masked_audio: np.ndarray,
    ) -> TADResult:
        """
        Process one 80ms chunk and return a :class:`TADResult`.

        Parameters
        ----------
        confidence : float
            P(trumpet) for this chunk from :class:`TrumpetScorer`.
        raw_audio : np.ndarray
            Original (un-masked) audio chunk — stored in the ring buffer
            for retroactive flush.
        masked_audio : np.ndarray
            Soft-masked audio from the mask generator.  Returned as-is
            during ACTIVE; silenced during all other states.
        """
        above = confidence >= self._threshold

        flush: Optional[list[np.ndarray]] = None
        out_audio = np.zeros_like(raw_audio)

        # Store raw audio in ring buffer regardless of state
        self._ring.append(raw_audio.copy())

        # ------------------------------------------------------------------
        # State transitions
        # ------------------------------------------------------------------

        if self._state == TADState.SILENT:
            if above:
                self._onset_count = 1
                if self._onset_count >= self._onset_chunks:
                    # onset_chunks=1: activate immediately
                    self._state = TADState.ACTIVE
                    flush = self._build_flush()
                    out_audio = masked_audio
                else:
                    self._state = TADState.ONSET
            # output stays silent unless we just activated above

        elif self._state == TADState.ONSET:
            if above:
                self._onset_count += 1
                if self._onset_count >= self._onset_chunks:
                    # Confirmed — enter ACTIVE, emit retroactive flush
                    self._state = TADState.ACTIVE
                    self._onset_count = 0
                    flush = self._build_flush()
                    out_audio = masked_audio
            else:
                # Evidence collapsed before threshold — back to SILENT
                self._state = TADState.SILENT
                self._onset_count = 0

        elif self._state == TADState.ACTIVE:
            if above:
                out_audio = masked_audio
            else:
                # First chunk below threshold
                self._trailing_count = 1
                if self._trailing_count >= self._trailing_chunks:
                    # trailing_chunks=1: expire immediately
                    self._state = TADState.SILENT
                    self._trailing_count = 0
                    out_audio = self._fade_audio(masked_audio, self._trailing_chunks)
                else:
                    self._state = TADState.TRAILING
                    out_audio = self._fade_audio(masked_audio, self._trailing_count)

        elif self._state == TADState.TRAILING:
            if above:
                # Confidence recovered — snap back to ACTIVE, no re-onset
                self._state = TADState.ACTIVE
                self._trailing_count = 0
                out_audio = masked_audio
            else:
                self._trailing_count += 1
                if self._trailing_count >= self._trailing_chunks:
                    # Hold-off expired — go SILENT
                    self._state = TADState.SILENT
                    expired_at = self._trailing_count   # capture before reset
                    self._trailing_count = 0
                    # Final chunk: fully faded (near silence)
                    out_audio = self._fade_audio(masked_audio, expired_at)
                else:
                    # Still in hold-off — fade toward silence
                    out_audio = self._fade_audio(masked_audio, self._trailing_count)

        return TADResult(
            masked_audio=out_audio,
            state=self._state,
            is_trumpet=(self._state == TADState.ACTIVE),
            confidence=float(confidence),
            flush=flush,
        )

    def reset(self) -> None:
        """Clear all internal state (e.g. on stream restart)."""
        self._state          = TADState.SILENT
        self._onset_count    = 0
        self._trailing_count = 0
        self._ring.clear()

    @property
    def state(self) -> TADState:
        """Current state (read-only)."""
        return self._state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_flush(self) -> list[np.ndarray]:
        """
        Build the retroactive flush packet from the ring buffer.

        Applies graduated alpha to each stored frame:
        - Oldest frames (very uncertain): alpha = 0.0 → zeroed out
        - Last 2 frames before confirmation: alpha = 0.3, 0.6
        - Frames are returned newest-last so Moshi receives them in order.

        Returns an empty list if the ring buffer has no frames.
        """
        frames = list(self._ring)   # oldest → newest
        if not frames:
            return []

        n = len(frames)
        # Build alpha sequence aligned to the tail of the frame list
        # _FLUSH_ALPHAS[-n:] handles the case where n < len(_FLUSH_ALPHAS)
        alphas_tail = self._FLUSH_ALPHAS[-n:] if n <= len(self._FLUSH_ALPHAS) \
                      else [0.0] * (n - len(self._FLUSH_ALPHAS)) + self._FLUSH_ALPHAS

        result = []
        for frame, alpha in zip(frames, alphas_tail):
            if alpha > 0.0:
                result.append((frame * alpha).astype(np.float32))
            # alpha == 0.0 → discard (don't send to Moshi)

        return result

    def _fade_audio(self, audio: np.ndarray, trailing_count: int) -> np.ndarray:
        """
        Linearly fade the soft-masked audio toward silence during TRAILING.

        At trailing_count=1: gain = (trailing_chunks - 1) / trailing_chunks
        At trailing_count=trailing_chunks: gain ≈ 0

        This prevents clicks/pops at the end of a note.
        """
        gain = max(
            0.0,
            1.0 - (trailing_count / self._trailing_chunks)
        )
        return (audio * gain).astype(np.float32)
