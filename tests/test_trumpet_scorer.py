"""
tests/test_trumpet_scorer.py
-----------------------------
Unit tests for TrumpetScorer.

These tests use the trained ONNX model bundled in trublib/models/.
If the model is not present, all tests are skipped with a clear message.

Tests cover:
- Model loads without error
- score() returns float in [0.0, 1.0]
- score() returns 0.0 for empty input
- score_raw() accepts a numpy matrix
- Trumpet-like signal scores higher than noise
- score() is reproducible (deterministic)
- model_input_dim matches feature vector dimension
- Custom model_path works
"""

from __future__ import annotations

import numpy as np
import pytest

from trublib.feature_extractor import FeatureExtractor
from trublib.frame_manager import FrameManager
from tests.conftest import make_harmonic, make_white_noise, SR, CHUNK


# ---------------------------------------------------------------------------
# Skip guard — model must be present
# ---------------------------------------------------------------------------

def model_is_available() -> bool:
    try:
        from trublib.trumpet_scorer import TrumpetScorer
        TrumpetScorer()
        return True
    except FileNotFoundError:
        return False
    except ImportError:
        return False

requires_model = pytest.mark.skipif(
    not model_is_available(),
    reason=(
        "trumpet_scorer_v1.onnx not found in trublib/models/. "
        "Copy your trained model there and reinstall the package."
    )
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_feature_vectors(signal: np.ndarray):
    """
    Extract all frames from the signal — no early cutoff.

    The model was trained on stats-pooled 5-second chunks (~468 frames).
    Cutting to 10 frames produces near-zero std values across the board,
    putting the feature vector far outside the training distribution and
    causing the model to output 0.0 for everything.

    Use a full-length signal (>= 1 second) when calling this in tests.
    """
    fm = FrameManager()
    ex = FeatureExtractor(sr=SR)
    fvs = []
    for start in range(0, len(signal), CHUNK):
        chunk = signal[start:start + CHUNK]
        for frame in fm.push(chunk):
            fvs.append(ex.extract(frame))
    return fvs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_model
class TestTrumpetScorerLoading:
    def test_loads_bundled_model(self):
        from trublib.trumpet_scorer import TrumpetScorer
        scorer = TrumpetScorer()
        assert scorer is not None

    def test_model_input_dim(self):
        from trublib.trumpet_scorer import TrumpetScorer
        scorer = TrumpetScorer()
        assert scorer.model_input_dim == 106

    def test_custom_path_not_found_raises(self, tmp_path):
        from trublib.trumpet_scorer import TrumpetScorer
        with pytest.raises(FileNotFoundError):
            TrumpetScorer(model_path=tmp_path / "nonexistent.onnx")


@requires_model
class TestTrumpetScorerScore:
    def test_score_returns_float(self):
        from trublib.trumpet_scorer import TrumpetScorer
        scorer = TrumpetScorer()
        fvs = get_feature_vectors(make_harmonic(465, duration=5.0))
        result = scorer.score(fvs)
        assert isinstance(result, float)

    def test_score_in_unit_range(self):
        from trublib.trumpet_scorer import TrumpetScorer
        scorer = TrumpetScorer()
        for signal in [make_harmonic(465), make_white_noise()]:
            fvs = get_feature_vectors(signal)
            p = scorer.score(fvs)
            assert 0.0 <= p <= 1.0, f"P(trumpet)={p} out of range"

    def test_score_empty_input_returns_zero(self):
        from trublib.trumpet_scorer import TrumpetScorer
        scorer = TrumpetScorer()
        assert scorer.score([]) == 0.0

    def test_score_reproducible(self):
        from trublib.trumpet_scorer import TrumpetScorer
        scorer = TrumpetScorer()
        fvs = get_feature_vectors(make_harmonic(465, duration=5.0))
        p1 = scorer.score(fvs)
        p2 = scorer.score(fvs)
        assert p1 == p2

    def test_trumpet_scores_higher_than_noise(self):
        """Core acoustic test: model must separate trumpet from noise."""
        from trublib.trumpet_scorer import TrumpetScorer
        scorer = TrumpetScorer()

        fvs_t = get_feature_vectors(make_harmonic(465, n_harmonics=8, duration=5.0))
        fvs_n = get_feature_vectors(make_white_noise(duration=5.0))

        p_trumpet = scorer.score(fvs_t)
        p_noise = scorer.score(fvs_n)

        assert p_trumpet > p_noise, (
            f"Trumpet P={p_trumpet:.4f} should exceed noise P={p_noise:.4f}"
        )

    def test_trumpet_score_above_threshold(self):
        """Trumpet signal should clear the default 0.6 threshold."""
        from trublib.trumpet_scorer import TrumpetScorer
        scorer = TrumpetScorer()
        fvs = get_feature_vectors(make_harmonic(465, n_harmonics=8, duration=5.0))
        p = scorer.score(fvs)
        assert p >= 0.6, f"Trumpet P(trumpet)={p:.4f} below default threshold 0.6"

    def test_noise_score_below_threshold(self):
        """White noise should not clear the default 0.6 threshold."""
        from trublib.trumpet_scorer import TrumpetScorer
        scorer = TrumpetScorer()
        fvs = get_feature_vectors(make_white_noise(duration=5.0))
        p = scorer.score(fvs)
        assert p < 0.6, f"Noise P(trumpet)={p:.4f} above threshold 0.6"


@requires_model
class TestTrumpetScorerRaw:
    def test_score_raw_shape(self):
        from trublib.trumpet_scorer import TrumpetScorer
        scorer = TrumpetScorer()
        mat = np.random.randn(20, 106).astype(np.float32)
        result = scorer.score_raw(mat)
        assert result.shape == (20,)
        assert result.dtype == np.float32

    def test_score_raw_range(self):
        from trublib.trumpet_scorer import TrumpetScorer
        scorer = TrumpetScorer()
        mat = np.random.randn(50, 106).astype(np.float32)
        result = scorer.score_raw(mat)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_score_raw_wrong_dim_raises(self):
        from trublib.trumpet_scorer import TrumpetScorer
        scorer = TrumpetScorer()
        mat = np.random.randn(10, 53).astype(np.float32)  # wrong: 53 not 106
        with pytest.raises(ValueError, match="Expected 106 features"):
            scorer.score_raw(mat)

    def test_score_raw_1d_input_accepted(self):
        """Single sample as 1D array should be accepted (auto-reshaped)."""
        from trublib.trumpet_scorer import TrumpetScorer
        scorer = TrumpetScorer()
        vec = np.random.randn(106).astype(np.float32)
        result = scorer.score_raw(vec)
        assert result.shape == (1,)
