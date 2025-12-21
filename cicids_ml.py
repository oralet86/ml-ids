# CICIDS_ml.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from cicids2017 import (
    load_cicids2017,
    preprocess_cicids2017,
    numeric_preprocess_fit_transform,
    numeric_preprocess_transform,
    save_results,
    stratified_holdout,
)

from params import (
    N_TRIALS,
    RANDOM_STATE,
    SMOTE_K_NEIGHBORS,
    GINI_N_ESTIMATORS,
    TOP_K_FEATURES,
    CICIDS_DATA_PCT,
    VAL_FRAC,
)
from utils import CICIDS2017_PATH, RESULTS_DIR, ML_MODEL_LIST, logger


@dataclass(frozen=True)
class MLKnobs:
    random_state: int
    smote_k_neighbors: int
    gini_n_estimators: int
    top_k_features: int


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


def _metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec) / max(1e-12, prec + rec)
    return {
        "f1": float(f1),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
    }


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
    X_train_raw,
    y_train: np.ndarray,
    X_val_raw,
    y_val: np.ndarray,
    X_test_raw,
    y_test: np.ndarray,
    knobs: MLKnobs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_tr_num, X_val_num, names, imp, sc = numeric_preprocess_fit_transform(
        X_train_raw, X_val_raw
    )
    X_te_num = numeric_preprocess_transform(X_test_raw, names=names, imp=imp, sc=sc)

    y_tr = y_train.astype(np.float32, copy=False)
    y_val_out = y_val.astype(np.float32, copy=False)
    y_te_out = y_test.astype(np.float32, copy=False)

    X_tr = X_tr_num
    if len(np.unique(y_tr)) > 1:
        sm = SMOTE(random_state=knobs.random_state, k_neighbors=knobs.smote_k_neighbors)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

    X_val_out = X_val_num
    X_te_out = X_te_num

    if len(np.unique(y_tr)) > 1 and X_tr.shape[1] > 1:
        keep = top_k_by_gini(X_tr, y_tr, feature_names=names, knobs=knobs)
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


def run_cicids2017_ml(
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

    # Train on TRAIN only (no test leakage)
    if hasattr(model, "fit"):
        model.fit(X_tr, y_tr)
    else:
        # Fallback: if a wrapper lacks fit(), treat the provided "val" as the eval set.
        # This preserves the real test split for final evaluation.
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
    df = preprocess_cicids2017(
        load_cicids2017(
            cicids_dir=CICIDS2017_PATH,
            label_col="Label",
            pct=CICIDS_DATA_PCT,
            random_state=RANDOM_STATE,
            logger=logger,
        ),
        logger=logger,
    )

    day = df["__day"].astype("string").str.strip().str.lower()

    train_pool_df = df.loc[day.isin(("monday", "tuesday", "wednesday"))].copy()
    test_df = df.loc[day.isin(("thursday", "friday"))].copy()

    if train_pool_df.empty:
        raise ValueError("Train pool (Mon-Wed) is empty.")
    if test_df.empty:
        raise ValueError("Test (Thu-Fri) is empty.")

    y_pool = (
        (train_pool_df["Label"].astype("string").str.strip() != "BENIGN")
        .astype(np.int64)
        .to_numpy()
    )
    y_test = (
        (test_df["Label"].astype("string").str.strip() != "BENIGN")
        .astype(np.int64)
        .to_numpy()
    )

    X_pool = train_pool_df.drop(columns=["Label", "__day"])
    X_test_raw = test_df.drop(columns=["Label", "__day"])

    X_train_raw, y_train, X_val_raw, y_val = stratified_holdout(
        X_pool, y_pool, val_frac=VAL_FRAC, random_state=RANDOM_STATE
    )

    logger.info(
        "Custom split (ML): "
        f"train_pool(Mon-Wed)={len(train_pool_df)} -> train={len(X_train_raw)} val={len(X_val_raw)} "
        f"| test(Thu-Fri)={len(X_test_raw)} "
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
        run_cicids2017_ml(
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
