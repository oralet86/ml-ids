# unsw_nb15.py
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _find_first_csv(root: Path, pattern: str) -> Path:
    """Return first match under root; raise FileNotFoundError if none."""
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{pattern}' under: {root}\n"
            "Make sure the files are inside datasets/unsw_nb15/."
        )
    return matches[0]


def get_unsw_nb15(
    *,
    unsw_dir: os.PathLike | str,
    pct: float,
    random_state: int,
    logger: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load UNSW-NB15 training and testing CSVs separately.

    Looks for:
      - UNSW_NB15_training-set.csv  -> returned as TRAIN POOL
      - UNSW_NB15_testing-set.csv   -> returned as TEST

    If pct < 1.0, stratified sampling by 'label' is applied to the TRAIN POOL ONLY
    (so the official test set remains untouched).
    """
    root = Path(unsw_dir)
    train_file = _find_first_csv(root, "UNSW_NB15_training-set.csv")
    test_file = _find_first_csv(root, "UNSW_NB15_testing-set.csv")

    logger.info(f"Found training dataset in {train_file}")
    logger.info(f"Found testing dataset in {test_file}")

    t0 = time.time()
    df_train = pd.read_csv(train_file, engine="pyarrow", dtype_backend="pyarrow")
    df_test = pd.read_csv(test_file, engine="pyarrow", dtype_backend="pyarrow")

    df_train.columns = df_train.columns.astype(str).str.strip()
    df_test.columns = df_test.columns.astype(str).str.strip()

    if "label" not in df_train.columns:
        raise ValueError(
            "Expected 'label' column was not found in UNSW-NB15 TRAIN data."
        )
    if "label" not in df_test.columns:
        raise ValueError(
            "Expected 'label' column was not found in UNSW-NB15 TEST data."
        )

    df_train["label"] = df_train["label"].astype("object")
    df_test["label"] = df_test["label"].astype("object")

    # Apply sampling ONLY to training set (keep test set intact)
    if pct < 1.0:
        total_len = len(df_train)
        target_n = max(1, int(round(total_len * float(pct))))

        out = (
            df_train.groupby("label", group_keys=False)
            .sample(frac=float(pct), replace=False, random_state=int(random_state))
            .reset_index(drop=True)
        )

        if len(out) > target_n:
            out = out.sample(n=target_n, random_state=int(random_state)).reset_index(
                drop=True
            )

        df_train = out
        logger.info(
            f"Returning stratified sample on TRAIN 'label': "
            f"{len(df_train)}/{total_len} rows (pct={pct})"
        )

    logger.info(
        f"Loaded datasets in {time.time() - t0:.2f}s | "
        f"train={len(df_train)} test={len(df_test)}"
    )
    return df_train, df_test


def preprocess_unsw_nb15(df: pd.DataFrame, *, logger: Any) -> pd.DataFrame:
    """
    UNSW-specific hygiene-only preprocessing (kept from your original):
    - normalize column names
    - inf/-inf -> NA
    - placeholder strings -> NA for categorical-like columns
    - drop duplicate rows
    - drop 'id' if present
    - ensure 'label' exists and is int64
    """
    logger.info("Starting preprocessing of UNSW-NB15 dataset...")
    t0 = time.time()

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    df = df.replace([float("inf"), float("-inf")], np.nan)
    df = df.infer_objects(copy=False)

    placeholder_values = {
        "-",
        " -",
        "- ",
        "None",
        "none",
        "NULL",
        "null",
        "NaN",
        "nan",
        "",
    }
    obj_cols = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    for c in obj_cols:
        s = df[c].astype("string")
        df[c] = s.where(~s.isin(list(placeholder_values)), np.nan)

    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df)} duplicate rows")

    if "id" in df.columns:
        df = df.drop(columns=["id"])
        logger.info("Dropped columns: ['id']")

    if "label" not in df.columns:
        raise ValueError("Target column 'label' not found in UNSW-NB15 dataset.")
    df["label"] = df["label"].astype("int64")

    logger.info(f"Completed preprocessing in {time.time() - t0:.2f}s")
    return df


def _to_sklearn_nan(
    df: pd.DataFrame, *, num_cols: List[str], cat_cols: List[str]
) -> pd.DataFrame:
    """
    Convert Arrow / pandas nullable dtypes into sklearn-friendly numpy/object dtypes.

    - Numeric columns -> float64 with np.nan
    - Categorical columns -> object with None for missing

    Prevents: TypeError: boolean value of NA is ambiguous
    """
    out = df.copy()

    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("object")
            out[c] = out[c].where(pd.notna(out[c]), None)

    return out


def _infer_num_cat_cols(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    return num_cols, cat_cols


def get_unsw_preprocessor(
    *, num_cols: List[str], cat_cols: List[str]
) -> ColumnTransformer:
    """
    UNSW-specific preprocessing pipeline (kept as the distinctive part):
    - numeric: median impute + StandardScaler
    - categorical: most_frequent impute + OneHotEncoder(handle_unknown='ignore')
    """
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
    )


def preprocess_fit_transform(
    X_train_raw: pd.DataFrame, X_other_raw: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[str], ColumnTransformer]:
    """
    Fit UNSW preprocessor on TRAIN only, transform TRAIN + OTHER (val or test).
    Returns:
      X_tr_np, X_other_np, feature_names, fitted_preprocessor
    """
    num_cols, cat_cols = _infer_num_cat_cols(X_train_raw)

    X_tr = _to_sklearn_nan(X_train_raw, num_cols=num_cols, cat_cols=cat_cols)
    X_ot = _to_sklearn_nan(X_other_raw, num_cols=num_cols, cat_cols=cat_cols)

    pre = get_unsw_preprocessor(num_cols=num_cols, cat_cols=cat_cols)
    X_tr_np = pre.fit_transform(X_tr)
    X_ot_np = pre.transform(X_ot)

    # feature names (best-effort, used mostly for debugging; ML top-k uses indices)
    names: List[str] = []
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(int(X_tr_np.shape[1]))]

    return (
        np.asarray(X_tr_np, dtype=np.float32),
        np.asarray(X_ot_np, dtype=np.float32),
        names,
        pre,
    )


def preprocess_transform(
    X_raw: pd.DataFrame,
    *,
    fitted_preprocessor: ColumnTransformer,
    train_num_cols: List[str],
    train_cat_cols: List[str],
) -> np.ndarray:
    """
    Transform-only path using a train-fitted preprocessor.
    You must pass the TRAIN column lists (num/cat) so missing columns are handled consistently.
    """
    X_in = _to_sklearn_nan(X_raw, num_cols=train_num_cols, cat_cols=train_cat_cols)
    X_np = fitted_preprocessor.transform(X_in)
    return np.asarray(X_np, dtype=np.float32)


def stratified_split(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    CICIDS-style stratified split into train/test (no chronological constraint).
    """
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError(f"test_size must be in (0,1). Got {test_size}")

    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    if n < 2:
        raise ValueError("Not enough samples to split.")

    rng = np.random.default_rng(int(random_state))

    idx_pos = np.flatnonzero(y == 1)
    idx_neg = np.flatnonzero(y == 0)

    n_pos_te = int(round(len(idx_pos) * float(test_size)))
    n_neg_te = int(round(len(idx_neg) * float(test_size)))

    if len(idx_pos) > 0:
        n_pos_te = max(1, min(len(idx_pos) - 1 if len(idx_pos) > 1 else 1, n_pos_te))
    if len(idx_neg) > 0:
        n_neg_te = max(1, min(len(idx_neg) - 1 if len(idx_neg) > 1 else 1, n_neg_te))

    te_pos = (
        rng.choice(idx_pos, size=n_pos_te, replace=False)
        if len(idx_pos)
        else np.array([], dtype=np.int64)
    )
    te_neg = (
        rng.choice(idx_neg, size=n_neg_te, replace=False)
        if len(idx_neg)
        else np.array([], dtype=np.int64)
    )

    te_idx = np.concatenate([te_pos, te_neg])
    rng.shuffle(te_idx)

    mask = np.ones(n, dtype=bool)
    mask[te_idx] = False
    tr_idx = np.flatnonzero(mask)

    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    X_te = X.iloc[te_idx].reset_index(drop=True)
    y_te = y[te_idx]
    return X_tr, y_tr, X_te, y_te


def stratified_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    val_frac: float,
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Same helper you used for CICIDS: stratified validation holdout from a train pool.
    """
    if not (0.0 < float(val_frac) < 1.0):
        raise ValueError(f"val_frac must be in (0,1). Got {val_frac}")

    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    if n < 2:
        raise ValueError("Not enough samples to create holdout.")

    rng = np.random.default_rng(int(random_state))

    idx_pos = np.flatnonzero(y == 1)
    idx_neg = np.flatnonzero(y == 0)

    n_pos_val = int(round(len(idx_pos) * float(val_frac)))
    n_neg_val = int(round(len(idx_neg) * float(val_frac)))

    if len(idx_pos) > 0:
        n_pos_val = max(1, min(len(idx_pos) - 1 if len(idx_pos) > 1 else 1, n_pos_val))
    if len(idx_neg) > 0:
        n_neg_val = max(1, min(len(idx_neg) - 1 if len(idx_neg) > 1 else 1, n_neg_val))

    val_pos = (
        rng.choice(idx_pos, size=n_pos_val, replace=False)
        if len(idx_pos)
        else np.array([], dtype=np.int64)
    )
    val_neg = (
        rng.choice(idx_neg, size=n_neg_val, replace=False)
        if len(idx_neg)
        else np.array([], dtype=np.int64)
    )

    val_idx = np.concatenate([val_pos, val_neg])
    rng.shuffle(val_idx)

    mask = np.ones(n, dtype=bool)
    mask[val_idx] = False
    tr_idx = np.flatnonzero(mask)

    X_tr2 = X.iloc[tr_idx].reset_index(drop=True)
    y_tr2 = y[tr_idx]
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_val = y[val_idx]
    return X_tr2, y_tr2, X_val, y_val


def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Same metrics helper as CICIDS."""
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


def save_results(
    *,
    model_name: str,
    best_value: float,
    best_metrics: Dict[str, float],
    best_params: Dict[str, Any],
    n_trials: int,
    total_time: float,
    results_dir: os.PathLike | str,
    logger: Any,
) -> Dict[str, Any]:
    """
    CICIDS-style result writer (log file per model run).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(str(results_dir), exist_ok=True)

    log_path = os.path.join(str(results_dir), f"{model_name}_{ts}.log")
    lines = [
        f"Model: {model_name}",
        "Dataset: UNSW_NB15",
        "Split: Train/Val/Test via stratified split + stratified holdout",
        f"Date: {datetime.now().isoformat()}",
        f"Trials: {n_trials}",
        "-" * 60,
        f"Total Time: {total_time:.2f}s",
        "-" * 60,
        f"Test F1: {best_value:.6f}",
        f"Test Accuracy:  {best_metrics.get('accuracy', 0.0):.6f}",
        f"Test Precision: {best_metrics.get('precision', 0.0):.6f}",
        f"Test Recall:    {best_metrics.get('recall', 0.0):.6f}",
        f"Params:\n{json.dumps(best_params, indent=4)}",
    ]
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved log to {log_path}")
    return best_params
