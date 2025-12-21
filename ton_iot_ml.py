# ton_iot_ml.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from ton_iot import (
    load_ton_iot,
    preprocess_ton_iot,
    to_binary_labels,
    ton_preprocess_fit_transform,
    ton_preprocess_transform,
    metrics_binary,
    save_results,
    stratified_holdout,
)

from params import (
    N_TRIALS,
    RANDOM_STATE,
    SMOTE_K_NEIGHBORS,
    GINI_N_ESTIMATORS,
    TOP_K_FEATURES,
    VAL_FRAC,
    VAR_THRESHOLD,
    TONIOT_DATA_PCT,
    TEST_FRAC,
)
from utils import TON_IOT_PATH, RESULTS_DIR, ML_MODEL_LIST, logger


@dataclass(frozen=True)
class MLKnobs:
    random_state: int
    smote_k_neighbors: int
    gini_n_estimators: int
    top_k_features: int


def _metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return metrics_binary(y_true, y_pred)


def _get_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return p.reshape(-1)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return np.asarray(s).reshape(-1)
    y = model.predict(X)
    return np.asarray(y).reshape(-1).astype(np.float64)


def _best_threshold_f1(
    y_true: np.ndarray, scores: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    y_true_i = y_true.astype(np.int64)
    scores = scores.astype(np.float64)

    uniq = np.unique(scores)
    if uniq.size == 1:
        thr = float(uniq[0])
        y_pred = (scores >= thr).astype(np.int64)
        return thr, _metrics_from_preds(y_true_i, y_pred)

    mids = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = np.concatenate(([uniq[0] - 1e-12], mids, [uniq[-1] + 1e-12]))

    best_thr = float(candidates[0])
    best = {"f1": -1.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

    for thr in candidates:
        y_pred = (scores >= thr).astype(np.int64)
        m = _metrics_from_preds(y_true_i, y_pred)
        if m["f1"] > best["f1"]:
            best_thr = float(thr)
            best = m

    return best_thr, best


def top_k_by_gini(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: List[str],
    knobs: MLKnobs,
) -> List[int]:
    rf = RandomForestClassifier(
        n_estimators=knobs.gini_n_estimators,
        n_jobs=-1,
        random_state=knobs.random_state,
        class_weight="balanced",
        bootstrap=True,
    )
    rf.fit(X, y)
    idx = np.argsort(rf.feature_importances_)[::-1][
        : min(knobs.top_k_features, len(feature_names))
    ]
    return idx.tolist()


def build_ml_views(
    *,
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_val_raw: pd.DataFrame,
    y_val: np.ndarray,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    knobs: MLKnobs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_tr_np, X_val_np, art = ton_preprocess_fit_transform(
        X_train_raw, X_val_raw, var_threshold=VAR_THRESHOLD
    )
    X_te_np = ton_preprocess_transform(X_test_raw, art=art)

    y_tr = y_train.astype(np.float32, copy=False)
    y_val_out = y_val.astype(np.float32, copy=False)
    y_te_out = y_test.astype(np.float32, copy=False)

    X_tr = X_tr_np
    if len(np.unique(y_tr)) > 1:
        sm = SMOTE(random_state=knobs.random_state, k_neighbors=knobs.smote_k_neighbors)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

    X_val_out = X_val_np
    X_te_out = X_te_np

    feat_names = (
        art.feature_names
        if art.feature_names
        else [f"f{i}" for i in range(X_tr.shape[1])]
    )
    if len(np.unique(y_tr)) > 1 and X_tr.shape[1] > 1:
        keep = top_k_by_gini(X_tr, y_tr, feature_names=feat_names, knobs=knobs)
        X_tr = X_tr[:, keep]
        X_val_out = X_val_out[:, keep]
        X_te_out = X_te_out[:, keep]

    return (
        X_tr.astype(np.float32, copy=False),
        y_tr.astype(np.float32, copy=False),
        X_val_out.astype(np.float32, copy=False),
        y_val_out,
        X_te_out.astype(np.float32, copy=False),
        y_te_out,
    )


def run_ton_iot_ml(
    *,
    model_class,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_trials: int,
    results_dir,
    logger,
) -> Dict[str, Any]:
    t0 = time.time()

    if len(np.unique(y_tr)) < 2:
        logger.warning("Training has only 1 class after preprocessing. Skipping.")
        return save_results(
            model_name=model_class.__name__,
            best_value=0.0,
            best_metrics={"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0},
            best_params={},
            n_trials=n_trials,
            total_time=time.time() - t0,
            results_dir=results_dir,
            logger=logger,
        )

    model = model_class()

    if hasattr(model, "fit"):
        model.fit(X_tr, y_tr)
    else:
        _ = model.train_model(X_tr, y_tr, X_val, y_val)

    # Tune threshold on VAL
    val_scores = _get_scores(model, X_val)
    thr, val_metrics = _best_threshold_f1(y_val, val_scores)

    # Evaluate TEST with that threshold
    te_scores = _get_scores(model, X_te)
    te_pred = (te_scores >= thr).astype(np.int64)
    test_metrics = _metrics_from_preds(y_te, te_pred)

    return save_results(
        model_name=model_class.__name__,
        best_value=test_metrics["f1"],
        best_metrics=test_metrics,
        best_params={"threshold": thr, "val_metrics_at_threshold": val_metrics},
        n_trials=n_trials,
        total_time=time.time() - t0,
        results_dir=results_dir,
        logger=logger,
    )


if __name__ == "__main__":
    df = preprocess_ton_iot(
        load_ton_iot(
            ton_iot_path=TON_IOT_PATH,
            label_col="label",
            pct=TONIOT_DATA_PCT,
            random_state=RANDOM_STATE,
            logger=logger,
        ),
        logger=logger,
    )

    target_col = "label"
    if target_col not in df.columns:
        raise ValueError("Expected 'label' column not found after preprocessing.")

    y_all = to_binary_labels(df[target_col])
    X_all = df.drop(columns=[target_col])

    # Create TEST split first, then VAL from remaining pool (CICIDS-style: keep a real test set)
    X_pool, y_pool, X_test_raw, y_test = stratified_holdout(
        X_all, y_all, val_frac=TEST_FRAC, random_state=RANDOM_STATE
    )
    X_train_raw, y_train, X_val_raw, y_val = stratified_holdout(
        X_pool, y_pool, val_frac=VAL_FRAC, random_state=RANDOM_STATE
    )

    logger.info(
        "TON split (ML): "
        f"pool={len(X_pool)} -> train={len(X_train_raw)} val={len(X_val_raw)} "
        f"| test={len(X_test_raw)} "
        f"| pool_pos={float(y_pool.mean()):.4f} train_pos={float(y_train.mean()):.4f} "
        f"val_pos={float(y_val.mean()):.4f} test_pos={float(y_test.mean()):.4f}"
    )

    knobs = MLKnobs(
        random_state=RANDOM_STATE,
        smote_k_neighbors=SMOTE_K_NEIGHBORS,
        gini_n_estimators=GINI_N_ESTIMATORS,
        top_k_features=TOP_K_FEATURES,
    )

    X_tr, y_tr, X_val, y_val_out, X_te, y_te_out = build_ml_views(
        X_train_raw=X_train_raw,
        y_train=y_train,
        X_val_raw=X_val_raw,
        y_val=y_val,
        X_test_raw=X_test_raw,
        y_test=y_test,
        knobs=knobs,
    )

    for m in ML_MODEL_LIST:
        if not (hasattr(m, "train_model") or hasattr(m, "fit")):
            continue
        run_ton_iot_ml(
            model_class=m,
            X_tr=X_tr,
            y_tr=y_tr,
            X_val=X_val,
            y_val=y_val_out,
            X_te=X_te,
            y_te=y_te_out,
            n_trials=N_TRIALS,
            results_dir=RESULTS_DIR,
            logger=logger,
        )
