"""
trublib.trumpet_scorer
-----------------------
Loads the ONNX classifier and scores a batch of feature vectors,
returning P(trumpet) per frame.

Design decisions
~~~~~~~~~~~~~~~~
- Model loaded once at construction via importlib.resources — no hardcoded paths
- Input: list of FeatureVector objects (one per frame from FeatureExtractor)
- Output: float in [0.0, 1.0] — median P(trumpet) across all input frames
- Median (not mean) — resists single-frame noise spikes
- Thread safety: NOT thread-safe. One instance per audio stream.
- No PyTorch dependency — onnxruntime is compiled C++, stays fast

Runtime flow (per 80ms chunk)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    frames = frame_manager.push(chunk)          # ~7 Frame objects
    fvs    = [extractor.extract(f) for f in frames]
    score  = scorer.score(fvs)                  # float P(trumpet)
    is_trumpet = score >= config.threshold
"""

from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from trublib.feature_extractor import FeatureVector

log = logging.getLogger(__name__)

# Model filename — versioned so future upgrades ship alongside the old model
_MODEL_FILENAME = "trumpet_scorer_v1.onnx"


class TrumpetScorer:
    """
    Wraps the ONNX trumpet classifier with a clean, minimal interface.

    Parameters
    ----------
    model_path : Path | None
        Explicit path to a .onnx file.  When None (default), the model
        bundled inside the trublib package is used.  Pass an explicit path
        to use a custom or fine-tuned model.

    Examples
    --------
    ::

        scorer = TrumpetScorer()
        frames = frame_manager.push(chunk)
        fvs = [extractor.extract(f) for f in frames]
        p = scorer.score(fvs)          # float, e.g. 0.91
        is_trumpet = p >= 0.6
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self._session = self._load_session(model_path)
        self._input_name: str = self._session.get_inputs()[0].name
        self._n_features: int = self._session.get_inputs()[0].shape[1]
        log.debug(
            "TrumpetScorer loaded — input: %s (%d dims)",
            self._input_name, self._n_features,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, feature_vectors: Sequence[FeatureVector]) -> float:
        """
        Score a batch of frames from one 80ms chunk.

        Parameters
        ----------
        feature_vectors : Sequence[FeatureVector]
            Output of ``FeatureExtractor.extract()`` for each frame in
            the chunk.  Typically 6–8 vectors per 80ms chunk.

        Returns
        -------
        float
            P(trumpet) for this chunk in [0.0, 1.0], derived from
            stats-pooling (mean+std) across all input frames — matching
            the aggregation used during training.
            Returns 0.0 for an empty input (silent / no frames).
        """
        if not feature_vectors:
            return 0.0

        # Stack per-frame 53-dim vectors into (n_frames, 53)
        mat = np.stack(
            [fv.to_vector() for fv in feature_vectors],
            axis=0,
        ).astype(np.float32)

        # Apply the same stats pooling used during training:
        # mean + std across frames → single 106-dim segment vector
        mean = mat.mean(axis=0)                              # (53,)
        std  = mat.std(axis=0)                               # (53,)
        segment = np.concatenate([mean, std])[np.newaxis, :] # (1, 106)

        probs = self._run_inference(segment)  # (1, 2)
        return float(probs[0, 1])

    def score_raw(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Score a pre-built (N, 106) float32 matrix and return P(trumpet)
        per row as a float32 array.

        Useful for batch evaluation and training diagnostics.

        Parameters
        ----------
        feature_matrix : np.ndarray
            Shape (N, n_features), dtype float32.

        Returns
        -------
        np.ndarray
            Shape (N,), dtype float32 — P(trumpet) per row.
        """
        mat = np.asarray(feature_matrix, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat[np.newaxis, :]
        if mat.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {mat.shape[1]}. "
                "Make sure you are using the same FeatureExtractor version "
                "that was used to train this model."
            )
        probs = self._run_inference(mat)
        return probs[:, 1].astype(np.float32)

    @property
    def model_input_dim(self) -> int:
        """Expected number of features per sample (106 for v1)."""
        return self._n_features

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_inference(self, mat: np.ndarray) -> np.ndarray:
        """
        Run ONNX inference on a (N, n_features) float32 matrix.

        Returns (N, 2) float32 — [P(non_trumpet), P(trumpet)] per row.
        """
        outputs = self._session.run(None, {self._input_name: mat})
        # outputs[0] = labels (int64), outputs[1] = probabilities (float32, N×2)
        return outputs[1].astype(np.float32)

    @staticmethod
    def _load_session(model_path: Path | None):
        """
        Load an onnxruntime InferenceSession.

        Tries importlib.resources first (bundled model), falls back to
        explicit path if provided.
        """
        try:
            import onnxruntime as ort
        except Exception as e:
            raise ImportError(
                f"Failed to import onnxruntime: {type(e).__name__}: {e}. "
                "Install it with: pip install onnxruntime"
            ) from e

        if model_path is not None:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            log.info("TrumpetScorer: loading custom model from %s", path)
            return ort.InferenceSession(str(path))

        # Load bundled model via importlib.resources (Python 3.9+)
        # This works correctly whether trublib is installed as a wheel,
        # editable install, or zip-imported — never relies on __file__.
        try:
            pkg = importlib.resources.files("trublib") / "models" / _MODEL_FILENAME
            with importlib.resources.as_file(pkg) as onnx_path:
                if not onnx_path.exists():
                    raise FileNotFoundError(
                        f"Bundled model '{_MODEL_FILENAME}' not found inside the "
                        "trublib package.  Copy your trained .onnx file to "
                        "trublib/models/ and reinstall, or pass an explicit "
                        "model_path to TrumpetScorer()."
                    )
                log.info("TrumpetScorer: loading bundled model %s", onnx_path)
                return ort.InferenceSession(str(onnx_path))
        except (TypeError, AttributeError):
            # Python < 3.9 fallback using pkg_resources
            try:
                import pkg_resources
                onnx_bytes = pkg_resources.resource_string(
                    "trublib", f"models/{_MODEL_FILENAME}"
                )
                return ort.InferenceSession(onnx_bytes)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load bundled model '{_MODEL_FILENAME}'. "
                    f"Original error: {exc}"
                ) from exc
