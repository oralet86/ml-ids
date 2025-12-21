# ton_iot.py
from __future__ import annotations
from dataclasses import dataclass
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# TON_IoT dataset loading

_TON_DROP_COLS: List[str] = [
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "ts",
    "date",
    "time",
    "timestamp",
    "http_user_agent",
    "ssl_subject",
    "ssl_issuer",
    "type",
    "attack_cat",
]


def stratified_sample(
    df: pd.DataFrame, *, label_col: str, pct: float, random_state: int
) -> pd.DataFrame:
    """
    Stratified sampling within a single dataframe by label_col.
    Matches your prior behavior:
      groupby(label).sample(frac=pct), then hard-cap to target_n.
    """
    if pct >= 1.0:
        return df
    if label_col not in df.columns:
        raise ValueError(
            f"Expected '{label_col}' column was not found in TON_IoT data."
        )

    df = df.copy()
    df[label_col] = df[label_col].astype("object")

    total_len = len(df)
    target_n = max(1, int(round(total_len * pct)))

    out = (
        df.groupby(label_col, group_keys=False)
        .sample(frac=pct, replace=False, random_state=random_state)
        .reset_index(drop=True)
    )
    if len(out) > target_n:
        out = out.sample(n=target_n, random_state=random_state).reset_index(drop=True)
    return out


def load_ton_iot(
    *,
    ton_iot_path,  # Path-like root (e.g., TON_IOT_PATH from utils)
    filename: str = "train_test_network.csv",
    label_col: str = "label",
    pct: float = 1.0,
    random_state: int = 0,
    logger: Any,
) -> pd.DataFrame:
    """
    Load TON_IoT 'train_test_network.csv' under ton_iot_path.

    If pct < 1.0, returns a stratified sample on label_col to preserve class ratios.
    """
    file = next(ton_iot_path.rglob(filename), None)
    if file is None:
        raise FileNotFoundError(f"{filename} not found under: {ton_iot_path}")

    logger.info(f"Found dataset in {file}")
    t0 = time.time()

    df = pd.read_csv(file, engine="pyarrow", dtype_backend="pyarrow")
    df.columns = df.columns.astype(str).str.strip()

    if label_col not in df.columns:
        raise ValueError(
            f"Expected '{label_col}' column was not found in TON_IoT data."
        )

    if pct < 1.0:
        n0 = len(df)
        df = stratified_sample(
            df, label_col=label_col, pct=pct, random_state=random_state
        )
        logger.info(
            f"Returning stratified sample on '{label_col}': {len(df)}/{n0} rows (pct={pct})"
        )

    logger.info(f"Loaded dataset in {time.time() - t0:.2f}s")
    return df


def preprocess_ton_iot(df: pd.DataFrame, *, logger: Any) -> pd.DataFrame:
    """
    Keep ONLY TON_IoT-specific preprocessing from the original file:
      - drop duplicate rows
      - drop known identifier/temporal/text columns if present
    """
    t0 = time.time()
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    n0 = len(df)
    df = df.drop_duplicates()
    logger.info(f"TON clean: removed {n0 - len(df)} duplicate rows")

    existing_cols = [c for c in _TON_DROP_COLS if c in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)

    logger.info(
        f"TON clean: dropped {len(existing_cols)} cols, final_rows={len(df)} in {time.time() - t0:.2f}s"
    )
    return df


# Labels / metrics / splitting


def to_binary_labels(
    labels: pd.Series, *, benign_names: Tuple[str, ...] = ("normal", "benign")
) -> np.ndarray:
    """
    Robust conversion to binary labels:

    - If labels are numeric and {0,1} -> use them directly.
    - Else if any label matches benign_names (case-insensitive) -> benign=0, other=1.
    - Else if exactly 2 unique classes -> map most frequent class to 0, the other to 1.
    - Else (multi-class without a known benign name) -> map most frequent class to 0, others to 1.

    Returns int64 vector with values in {0,1}.
    """
    s = labels

    # Numeric 0/1 fast path
    if pd.api.types.is_numeric_dtype(s):
        u = pd.unique(s.dropna())
        u_set = set(map(int, u.tolist())) if len(u) else set()
        if u_set.issubset({0, 1}) and len(u_set) > 0:
            return s.fillna(0).astype(np.int64).to_numpy()

    s_str = s.astype("string").str.strip()
    s_low = s_str.str.lower()

    benign_set = {b.lower() for b in benign_names}
    has_known_benign = bool(s_low.isin(list(benign_set)).any())

    if has_known_benign:
        y = (~s_low.isin(list(benign_set))).astype(np.int64).to_numpy()
        return y

    # No known benign string. Use frequency heuristic.
    vc = s_low.value_counts(dropna=False)
    if vc.empty:
        return np.zeros(len(s), dtype=np.int64)

    benign_class = vc.index[0]  # most frequent
    y = (s_low != benign_class).astype(np.int64).to_numpy()
    return y


def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Shared TP/TN/FP/FN metrics (same as CICIDS behavior)."""
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


def stratified_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    val_frac: float,
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Create a stratified validation holdout from (X, y).
    Returns: X_train2, y_train2, X_val, y_val
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

    n_pos_val = int(round(len(idx_pos) * val_frac))
    n_neg_val = int(round(len(idx_neg) * val_frac))

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


# -----------------------------
# TON preprocessing transforms
# -----------------------------


@dataclass
class TonPreprocessArtifacts:
    """
    Fitted preprocessing objects for TON_IoT mixed feature types.
    """

    pipeline: ColumnTransformer
    var_filter: VarianceThreshold
    feature_names: List[str]


def get_pipeline(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    TON-specific mixed-type preprocessing (kept from your original):
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
        ]
    )


def ton_preprocess_fit_transform(
    X_train_raw: pd.DataFrame,
    X_other_raw: pd.DataFrame,
    *,
    var_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, TonPreprocessArtifacts]:
    """
    Fit on X_train_raw, transform both X_train_raw and X_other_raw.

    Keeps TON-specific steps from the original:
      - ColumnTransformer mixed preprocessing
      - VarianceThreshold(VAR_THRESHOLD)

    Returns float32 arrays and artifacts for transform-only on further splits.
    """
    num_cols = X_train_raw.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train_raw.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    pipe = get_pipeline(num_cols, cat_cols)
    X_tr = pipe.fit_transform(X_train_raw)
    X_ot = pipe.transform(X_other_raw)

    var_filter = VarianceThreshold(threshold=float(var_threshold))
    X_tr_f = var_filter.fit_transform(X_tr)
    X_ot_f = var_filter.transform(X_ot)

    # Feature names (best-effort)
    feat_names: List[str]
    try:
        raw_names = pipe.get_feature_names_out()
        raw_names = np.asarray(raw_names, dtype=str)
        keep = getattr(var_filter, "get_support", None)
        if callable(keep):
            mask = var_filter.get_support()
            feat_names = raw_names[mask].tolist()
        else:
            feat_names = raw_names.tolist()
    except Exception:
        feat_names = [f"f{i}" for i in range(int(X_tr_f.shape[1]))]

    art = TonPreprocessArtifacts(
        pipeline=pipe, var_filter=var_filter, feature_names=feat_names
    )

    return (
        np.asarray(X_tr_f, dtype=np.float32),
        np.asarray(X_ot_f, dtype=np.float32),
        art,
    )


def ton_preprocess_transform(
    X_raw: pd.DataFrame, *, art: TonPreprocessArtifacts
) -> np.ndarray:
    """
    Transform-only path using fitted artifacts.
    """
    X = art.pipeline.transform(X_raw)
    X = art.var_filter.transform(X)
    return np.asarray(X, dtype=np.float32)


# -----------------------------
# Results writer (CICIDS style)
# -----------------------------


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
    CICIDS-style log writer adapted for TON_IoT (no Optuna).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(str(results_dir), exist_ok=True)

    log_path = os.path.join(str(results_dir), f"{model_name}_{ts}.log")
    lines = [
        f"Model: {model_name}",
        "Dataset: TON_IoT",
        "Split: Stratified holdout -> train/val/test",
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
    with open(log_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved log to {log_path}")
    return best_params
