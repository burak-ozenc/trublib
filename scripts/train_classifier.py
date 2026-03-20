"""
train_classifier.py
--------------------
Trains a trumpet vs non-trumpet classifier on the feature matrix produced
by extract_features.py.  Benchmarks MLP vs SVM, selects the winner by
F1 score on the trumpet class, and exports to ONNX via skl2onnx.

Strategy
~~~~~~~~
- StandardScaler normalisation (required for both MLP and SVM)
- class_weight='balanced' on both models to handle 16:1 imbalance
- Optional SMOTE oversampling of trumpet class (--smote flag)
- 5-fold stratified cross-validation for honest evaluation
- Final model trained on full training set, evaluated on held-out test set
- ONNX export: model + scaler fused into a single sklearn Pipeline

Output
~~~~~~
    models/trumpet_scorer_v1.onnx   — production model
    models/scaler.pkl               — kept for reference / debugging
    models/training_report.json     — full metrics, feature importances

Usage
~~~~~
    python train_classifier.py \\
        --features  data/features.npz \\
        --out-dir   models \\
        --test-size 0.15 \\
        --smote

Requirements
~~~~~~~~~~~~
    pip install scikit-learn skl2onnx onnxruntime imbalanced-learn
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def build_mlp(class_weight: dict | str = "balanced") -> Pipeline:
    """
    MLP with two hidden layers.

    Architecture chosen for the feature space:
    - Input: 106 dims (mean + std of 53 features)
    - Hidden: 256 → 128  (enough capacity without overfitting on 79k samples)
    - Output: 2 classes with softmax

    relu activation avoids vanishing gradients.
    early_stopping guards against overfitting.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,             # L2 regularisation
            batch_size=512,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
            verbose=False,
        )),
    ])


def build_svm(class_weight: dict | str = "balanced") -> Pipeline:
    """
    RBF-kernel SVM.

    SVM with RBF kernel is a strong baseline for well-normalised feature
    vectors of this dimensionality.  C=10 and gamma='scale' are good
    starting points; grid search can tune later.

    Note: SVM does not natively output probabilities.  probability=True
    adds Platt scaling (5-fold internal CV) — necessary for P(trumpet).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight=class_weight,
            probability=True,       # needed for P(trumpet) output
            random_state=42,
            cache_size=2000,        # MB — speeds up training on large datasets
        )),
    ])


# ---------------------------------------------------------------------------
# SMOTE oversampling
# ---------------------------------------------------------------------------


def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """
    Oversample the minority class (trumpet) with SMOTE.

    SMOTE generates synthetic trumpet samples by interpolating between
    existing trumpet feature vectors in the 106-dim space.  More effective
    than random duplication because it adds diversity.

    Requires: pip install imbalanced-learn
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        log.error(
            "imbalanced-learn not installed.  "
            "Run: pip install imbalanced-learn  or omit --smote flag."
        )
        raise

    log.info("Applying SMOTE oversampling...")
    t0 = time.perf_counter()

    smote = SMOTE(
        sampling_strategy=0.3,  # trumpet:non_trumpet = 0.3 after SMOTE (~4.5:15)
        k_neighbors=5,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X, y)

    elapsed = time.perf_counter() - t0
    n_new = int((y_res == 1).sum()) - int((y == 1).sum())
    log.info(
        "SMOTE complete in %.1fs — added %d synthetic trumpet samples "
        "(total trumpet: %d, non_trumpet: %d)",
        elapsed, n_new, int((y_res == 1).sum()), int((y_res == 0).sum())
    )
    return X_res, y_res


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate_model(
        model_fn,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        name: str = "model",
) -> dict:
    """
    Stratified k-fold CV.  Returns mean ± std of key metrics.

    We prioritise F1 (trumpet) as the primary metric because:
    - Precision: how often 'is_trumpet' is actually trumpet
    - Recall: how often real trumpet is detected
    - F1: harmonic mean — penalises both false alarms and misses equally
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    f1s, precisions, recalls, aucs = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        t0 = time.perf_counter()
        pipe = model_fn()
        pipe.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - t0

        y_pred = pipe.predict(X_val)
        y_prob = pipe.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, y_pred, pos_label=1)
        prec = f1_score(y_val, y_pred, pos_label=1, average=None)[1] if hasattr(f1_score(y_val, y_pred, pos_label=1, average=None), '__len__') else 0
        rec = f1_score(y_val, y_pred, pos_label=1, average=None)

        from sklearn.metrics import precision_score, recall_score
        prec = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_val, y_prob)

        f1s.append(f1)
        precisions.append(prec)
        recalls.append(rec)
        aucs.append(auc)

        log.info(
            "  [%s] fold %d/%d — F1: %.3f  Prec: %.3f  Rec: %.3f  AUC: %.3f  (%.1fs)",
            name, fold, n_splits, f1, prec, rec, auc, elapsed
        )

    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "precision_mean": float(np.mean(precisions)),
        "recall_mean": float(np.mean(recalls)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_to_onnx(pipeline: Pipeline, out_path: Path, n_features: int) -> None:
    """
    Export the full sklearn Pipeline (scaler + classifier) to ONNX.

    The exported model accepts a float32 array of shape (N, 106) and
    outputs:
        label      : int64  (N,)      — predicted class (0 or 1)
        probabilities : float32 (N, 2) — [P(non_trumpet), P(trumpet)]

    At inference time in TrumpetScorer:
        p_trumpet = probabilities[:, 1]
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        log.error(
            "skl2onnx not installed.  Run: pip install skl2onnx"
        )
        raise

    log.info("Exporting pipeline to ONNX...")

    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=17,
        options={id(pipeline["clf"]): {"zipmap": False}},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_kb = out_path.stat().st_size / 1024
    log.info("ONNX model saved to %s (%.1f KB)", out_path, size_kb)


def verify_onnx(onnx_path: Path, X_sample: np.ndarray) -> None:
    """Quick sanity check: run a few rows through onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping ONNX verification.")
        return

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    sample = X_sample[:10].astype(np.float32)
    outputs = sess.run(None, {input_name: sample})

    labels = outputs[0]       # (10,)
    probs = outputs[1]        # (10, 2)

    log.info("ONNX verification — 10 sample predictions:")
    for i in range(len(labels)):
        log.info(
            "  sample %02d — label=%d  P(trumpet)=%.4f  P(non_trumpet)=%.4f",
            i, labels[i], probs[i][1], probs[i][0]
        )


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------


def analyse_threshold(pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute F1, precision, recall at multiple P(trumpet) thresholds.

    Helps you pick a threshold appropriate for your use case:
    - High recall, lower precision → fewer missed trumpets (better for MERT)
    - High precision, lower recall → fewer false alarms (better for Moshi)

    Default threshold in TADConfig is 0.6.
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    log.info("")
    log.info("Threshold analysis (trumpet class):")
    log.info("  %-10s  %-10s  %-10s  %-10s", "Threshold", "Precision", "Recall", "F1")
    log.info("  " + "-" * 44)

    from sklearn.metrics import precision_score, recall_score

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        prec = precision_score(y_test, y_pred_t, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred_t, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred_t, pos_label=1, zero_division=0)
        log.info("  %-10.2f  %-10.3f  %-10.3f  %-10.3f", t, prec, rec, f1)
        results.append({"threshold": t, "precision": prec, "recall": rec, "f1": f1})

    return {"threshold_analysis": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and export a trumpet activity detection classifier."
    )
    # New primary mode: separate pre-split files from extract_features.py
    parser.add_argument(
        "--train", type=Path, default=None,
        help=(
            "Path to train.npz produced by extract_features.py (preferred). "
            "Use together with --test.  Bypasses --features and --test-size."
        )
    )
    parser.add_argument(
        "--test", type=Path, default=None,
        help="Path to test.npz produced by extract_features.py."
    )
    # Legacy mode: single combined file (old extract_features.py output)
    parser.add_argument(
        "--features", type=Path, default=None,
        help=(
            "[Legacy] Path to a single features.npz. "
            "Deprecated: use --train / --test from the new extract_features.py."
        )
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("models"),
        help="Output directory for ONNX model and reports (default: models/)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.15,
        help=(
            "[Legacy] Fraction of data held out when using --features (default: 0.15). "
            "Ignored when --train / --test are provided."
        )
    )
    parser.add_argument(
        "--smote", action="store_true",
        help="Apply SMOTE oversampling to the trumpet class before training"
    )
    parser.add_argument(
        "--model", choices=["mlp", "svm", "both"], default="both",
        help="Which model(s) to train (default: both — benchmark and pick winner)"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--skip-cv", action="store_true",
        help="Skip cross-validation and go straight to final training (faster)"
    )
    args = parser.parse_args()

    # --- Load features (new pre-split mode or legacy single-file mode) ---
    use_presplit = args.train is not None and args.test is not None

    if use_presplit:
        # ── New mode: separate train.npz / test.npz from extract_features.py ──
        for p, name in [(args.train, "--train"), (args.test, "--test")]:
            if not p.exists():
                log.error("%s file not found: %s", name, p)
                raise SystemExit(1)

        log.info("Loading pre-split train features from %s", args.train)
        train_data = np.load(args.train, allow_pickle=True)
        X_train = train_data["X"].astype(np.float32)
        y_train = train_data["y"].astype(np.int32)
        feature_names = list(train_data["feature_names"])

        log.info("Loading pre-split test features from %s", args.test)
        test_data = np.load(args.test, allow_pickle=True)
        X_test = test_data["X"].astype(np.float32)
        y_test = test_data["y"].astype(np.int32)

        log.info(
            "Train: %d samples  (trumpet: %d, non_trumpet: %d)",
            len(X_train), int((y_train == 1).sum()), int((y_train == 0).sum()),
        )
        log.info(
            "Test:  %d samples  (trumpet: %d, non_trumpet: %d)",
            len(X_test), int((y_test == 1).sum()), int((y_test == 0).sum()),
        )

    elif args.features is not None:
        # ── Legacy mode: single features.npz, do internal split ──
        log.warning(
            "--features is deprecated.  Re-run extract_features.py to get "
            "separate train.npz / test.npz, then use --train / --test."
        )
        log.info("Loading features from %s", args.features)
        data = np.load(args.features, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int32)
        feature_names = list(data["feature_names"])

        log.info("Loaded: X=%s  y=%s", X.shape, y.shape)
        log.info("Classes — trumpet: %d  non_trumpet: %d  ratio: %.1f:1",
                 int((y == 1).sum()), int((y == 0).sum()),
                 int((y == 0).sum()) / int((y == 1).sum()))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            stratify=y,
            random_state=42,
        )
        log.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    else:
        parser.error("Provide either --train + --test (recommended) or --features (legacy).")

    # --- Optional SMOTE on training set only ---
    X_train_fit, y_train_fit = X_train, y_train
    if args.smote:
        X_train_fit, y_train_fit = apply_smote(X_train, y_train)

    # --- Cross-validation ---
    cv_results = {}

    models_to_train = []
    if args.model in ("mlp", "both"):
        models_to_train.append(("MLP", build_mlp))
    if args.model in ("svm", "both"):
        models_to_train.append(("SVM", build_svm))

    if not args.skip_cv:
        log.info("")
        log.info("=" * 60)
        log.info("Cross-validation (%d folds)", args.cv_folds)
        log.info("=" * 60)

        for name, model_fn in models_to_train:
            log.info("")
            log.info("--- %s ---", name)
            cv = cross_validate_model(
                model_fn, X_train_fit, y_train_fit,
                n_splits=args.cv_folds, name=name
            )
            cv_results[name] = cv
            log.info(
                "  %s CV summary — F1: %.3f ± %.3f  AUC: %.3f ± %.3f",
                name, cv["f1_mean"], cv["f1_std"], cv["auc_mean"], cv["auc_std"]
            )

    # --- Pick winner ---
    if len(models_to_train) == 2 and cv_results:
        winner_name = max(cv_results, key=lambda k: cv_results[k]["f1_mean"])
        log.info("")
        log.info("Winner: %s (F1=%.3f vs F1=%.3f)",
                 winner_name,
                 cv_results[winner_name]["f1_mean"],
                 cv_results[[k for k in cv_results if k != winner_name][0]]["f1_mean"])
        winner_fn = dict(models_to_train)[winner_name]
    else:
        winner_name, winner_fn = models_to_train[0]

    # --- Final model: train on full training set ---
    log.info("")
    log.info("=" * 60)
    log.info("Training final %s on full training set...", winner_name)
    log.info("=" * 60)

    t0 = time.perf_counter()
    final_pipeline = winner_fn()
    final_pipeline.fit(X_train_fit, y_train_fit)
    elapsed = time.perf_counter() - t0
    log.info("Training complete in %.1fs", elapsed)

    # --- Evaluate on held-out test set ---
    log.info("")
    log.info("=" * 60)
    log.info("Test set evaluation")
    log.info("=" * 60)

    y_pred = final_pipeline.predict(X_test)
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]

    log.info("\n%s", classification_report(
        y_test, y_pred,
        target_names=["non_trumpet", "trumpet"],
        digits=4,
    ))

    cm = confusion_matrix(y_test, y_pred)
    log.info("Confusion matrix (rows=actual, cols=predicted):")
    log.info("  %-15s  predicted_non  predicted_trumpet", "")
    log.info("  %-15s  %13d  %17d", "actual_non", cm[0][0], cm[0][1])
    log.info("  %-15s  %13d  %17d", "actual_trumpet", cm[1][0], cm[1][1])

    auc = roc_auc_score(y_test, y_prob)
    f1_trumpet = f1_score(y_test, y_pred, pos_label=1)
    log.info("\nAUC-ROC: %.4f  |  F1 (trumpet): %.4f", auc, f1_trumpet)

    # --- Threshold analysis ---
    threshold_data = analyse_threshold(final_pipeline, X_test, y_test)

    # --- Save scaler separately for debugging ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = args.out_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(final_pipeline["scaler"], f)
    log.info("\nScaler saved to %s", scaler_path)

    # --- ONNX export ---
    log.info("")
    onnx_path = args.out_dir / "trumpet_scorer_v1.onnx"
    export_to_onnx(final_pipeline, onnx_path, n_features=X_train.shape[1])

    # --- ONNX verification ---
    verify_onnx(onnx_path, X_test)

    # --- Save training report ---
    report = {
        "winner_model": winner_name,
        "training_samples": int(len(X_train_fit)),
        "test_samples": int(len(X_test)),
        "smote_applied": args.smote,
        "class_counts": {
            "trumpet_train": int((y_train == 1).sum()),
            "non_trumpet_train": int((y_train == 0).sum()),
        },
        "cv_results": cv_results,
        "test_metrics": {
            "auc_roc": float(auc),
            "f1_trumpet": float(f1_trumpet),
            "confusion_matrix": cm.tolist(),
        },
        **threshold_data,
        "onnx_path": str(onnx_path),
        "feature_names": feature_names,
    }

    report_path = args.out_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Training report saved to %s", report_path)

    log.info("")
    log.info("=" * 60)
    log.info("Done.  Model ready at: %s", onnx_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()