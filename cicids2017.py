# CICIDS2017.py
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class SplitSpec:
    """Day-based split specification used for CICIDS2017 CSVs named by weekday."""

    train_days: Tuple[str, ...]
    val_days: Tuple[str, ...]
    test_days: Tuple[str, ...]


def day_from_filename(path: str) -> str:
    """
    Infer day-of-week from CICIDS2017 filename prefix.
    Expected: monday/tuesday/wednesday/thursday/friday*.csv (case-insensitive).
    """
    base = os.path.basename(path).strip().lower()
    for d in ("monday", "tuesday", "wednesday", "thursday", "friday"):
        if base.startswith(d):
            return d
    raise ValueError(f"Could not infer day-of-week from filename: {path}")


def stratified_sample_per_file(
    df: pd.DataFrame, *, label_col: str, pct: float, random_state: int
) -> pd.DataFrame:
    """
    Stratified sampling within a single CICIDS file by label_col.
    Matches your prior behavior: groupby(label).sample(frac=pct), then hard-cap to target_n.
    """
    if pct >= 1.0:
        return df
    if label_col not in df.columns:
        raise ValueError(
            f"Expected '{label_col}' column was not found in CICIDS2017 data."
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


def load_cicids2017(
    *,
    cicids_dir: os.PathLike | str,
    label_col: str = "Label",
    pct: float = 1.0,
    random_state: int = 0,
    logger: Any,
) -> pd.DataFrame:
    """
    Load all CICIDS2017 CSVs under cicids_dir, add '__day' column inferred from filenames.
    Sampling is applied *per file* (stratified by label_col) when pct < 1.0.
    """
    files = sorted(glob(str(os.path.join(str(cicids_dir), "*.csv"))))
    logger.info(f"Found {len(files)} files in {cicids_dir}")
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {cicids_dir}")

    dfs: List[pd.DataFrame] = []
    t0 = time.time()

    for f in files:
        day = day_from_filename(f)
        df = pd.read_csv(f, engine="pyarrow", dtype_backend="pyarrow")
        df.columns = df.columns.astype(str).str.strip()

        if label_col not in df.columns:
            raise ValueError(f"Expected '{label_col}' column not found in file: {f}")

        if pct < 1.0:
            n0 = len(df)
            df = stratified_sample_per_file(
                df, label_col=label_col, pct=pct, random_state=random_state
            )
            logger.info(f"Sampled {day}: {len(df)}/{n0} (pct={pct})")

        df["__day"] = day
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(dfs)} files in {time.time() - t0:.2f}s")
    return out


def preprocess_cicids2017(df: pd.DataFrame, *, logger: Any) -> pd.DataFrame:
    """
    EXACT same preprocessing as before:
      - strip column names
      - drop duplicated columns
      - replace +/-inf -> NaN, dropna
      - drop duplicate rows
    """
    t0 = time.time()
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    n1 = len(df)
    df = df.drop_duplicates()
    n2 = len(df)
    logger.info(
        f"Preprocess: -NaN/Inf {n0 - n1}, -dups {n1 - n2}, final {n2} in {time.time() - t0:.2f}s"
    )
    return df


def split_by_day(
    df: pd.DataFrame,
    *,
    label_col: str,
    split: SplitSpec,
    logger: Any,
) -> Tuple[
    pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray
]:
    """
    Day-based split using df['__day'] produced by load_cicids2017().
    Produces binary y: (label != 'BENIGN').
    """
    if "__day" not in df.columns:
        raise ValueError(
            "Expected '__day' column not found. load_cicids2017() must add '__day'."
        )

    day = df["__day"].astype("string").str.strip().str.lower()
    train_df = df.loc[day.isin(split.train_days)].copy()
    val_df = df.loc[day.isin(split.val_days)].copy()
    test_df = df.loc[day.isin(split.test_days)].copy()

    if train_df.empty:
        raise ValueError(f"Train split is empty. Expected days: {split.train_days}.")
    if val_df.empty:
        raise ValueError(f"Val split is empty. Expected days: {split.val_days}.")
    if test_df.empty:
        raise ValueError(f"Test split is empty. Expected days: {split.test_days}.")

    def to_binary_y(frame: pd.DataFrame) -> np.ndarray:
        return (
            (frame[label_col].astype("string").str.strip() != "BENIGN")
            .astype(np.int64)
            .to_numpy()
        )

    y_train = to_binary_y(train_df)
    y_val = to_binary_y(val_df)
    y_test = to_binary_y(test_df)

    X_train = train_df.drop(columns=[label_col, "__day"])
    X_val = val_df.drop(columns=[label_col, "__day"])
    X_test = test_df.drop(columns=[label_col, "__day"])

    logger.info(
        f"Day split: train={len(train_df)} val={len(val_df)} test={len(test_df)} | "
        f"train_pos={float(y_train.mean()):.4f} val_pos={float(y_val.mean()):.4f} test_pos={float(y_test.mean()):.4f}"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def numeric_preprocess_fit_transform(
    X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[str], SimpleImputer, MinMaxScaler]:
    """
    Shared numeric pipeline (EXACT same behavior):
      - numeric-only
      - drop constant columns based on train
      - align test to train's remaining columns
      - median impute (fit on train)
      - MinMax scale (fit on train)
    """
    X_tr = X_train_raw.select_dtypes(include=[np.number]).copy()
    X_te = X_test_raw.select_dtypes(include=[np.number]).copy()

    if X_tr.shape[1] == 0:
        raise ValueError("No numeric features found in training split.")
    if X_te.shape[1] == 0:
        raise ValueError("No numeric features found in test split.")

    nunique = X_tr.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index
    if len(const_cols) > 0:
        X_tr = X_tr.drop(columns=const_cols)
        X_te = X_te.drop(columns=const_cols, errors="ignore")

    names = X_tr.columns.tolist()
    X_te = X_te.reindex(columns=names)

    imp = SimpleImputer(strategy="median")
    X_tr_np = imp.fit_transform(X_tr)
    X_te_np = imp.transform(X_te)

    sc = MinMaxScaler()
    X_tr_np = sc.fit_transform(X_tr_np)
    X_te_np = sc.transform(X_te_np)

    return (
        X_tr_np.astype(np.float32, copy=False),
        X_te_np.astype(np.float32, copy=False),
        names,
        imp,
        sc,
    )


def numeric_preprocess_transform(
    X_raw: pd.DataFrame,
    *,
    names: List[str],
    imp: SimpleImputer,
    sc: MinMaxScaler,
) -> np.ndarray:
    """
    Transform-only path (EXACT same behavior):
      - numeric-only
      - reindex to TRAIN names (missing->NaN->imputed)
      - apply train-fitted imputer+scaler
    """
    X_num = X_raw.select_dtypes(include=[np.number]).copy()
    X_num = X_num.reindex(columns=names)
    X_np = imp.transform(X_num)
    X_np = sc.transform(X_np)
    return X_np.astype(np.float32, copy=False)


def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Shared TP/TN/FP/FN metrics (EXACT same behavior)."""
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

    # Ensure at least 1 sample from each class if possible
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
    Shared result writer (EXACT same behavior), but directories are passed in
    so this module does not “know where it will be used”.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(str(results_dir), exist_ok=True)

    log_path = os.path.join(str(results_dir), f"{model_name}_{ts}.log")
    lines = [
        f"Model: {model_name}",
        "Dataset: CICIDS2017",
        "Split: Train=Mon-Tue | Val=Wed | Test=Thu-Fri",
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
