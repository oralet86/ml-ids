from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, Type
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from base_models_abc import ArrayLike, BaseDLModel, BaseMLModel
from params import (
    N_TRIALS,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_IN_TRAIN,
    BATCH_SIZE,
    MOMENTUM,
    EPOCHS,
    PATIENCE,
    VAR_THRESHOLD,
    GINI_N_ESTIMATORS,
    TOP_K_FEATURES,
    SMOTE_K_NEIGHBORS,
)
from utils import HYPERPARAMS_DIR, MODEL_LIST, RESULTS_DIR, UNSW_NB15_PATH, logger

# This determines how much of the total data will be used.
# 0.25 Means %25 of the original data is picked via stratified sampling.
# This is not in params.py since the value will be different for each dataset.
DATA_PCT = 0.5


def _find_first_csv(pattern: str) -> os.PathLike:
    """Return first match under UNSW_NB15_PATH; raise FileNotFoundError if none."""
    matches = list(UNSW_NB15_PATH.rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{pattern}' under: {UNSW_NB15_PATH}\n"
            "Make sure the files are inside datasets/unsw_nb15/."
        )
    return matches[0]


def get_unsw_nb15(pct: float = DATA_PCT) -> pd.DataFrame:
    """
    Load UNSW-NB15 training and testing CSVs and concatenate them.
    Looks for:
      - UNSW_NB15_training-set.csv
      - UNSW_NB15_testing-set.csv
    """
    train_file = _find_first_csv("UNSW_NB15_training-set.csv")
    test_file = _find_first_csv("UNSW_NB15_testing-set.csv")

    logger.info(f"Found training dataset in {train_file}")
    logger.info(f"Found testing dataset in {test_file}")

    start = time.time()
    df_train = pd.read_csv(train_file, engine="pyarrow", dtype_backend="pyarrow")
    df_test = pd.read_csv(test_file, engine="pyarrow", dtype_backend="pyarrow")

    df_train.columns = df_train.columns.astype(str).str.strip()
    df_test.columns = df_test.columns.astype(str).str.strip()

    df = pd.concat([df_train, df_test], ignore_index=True)

    if pct < 1.0:
        if "label" not in df.columns:
            raise ValueError(
                "Stratified sampling requires 'label' column, but it was not found."
            )

        total_len = len(df)
        target_n = max(1, int(round(total_len * pct)))

        df = (
            df.groupby("label", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=max(1, int(round(len(g) * pct))),
                    replace=False,
                    random_state=RANDOM_STATE,
                )
            )
            .reset_index(drop=True)
        )

        # Correct any rounding drift
        if len(df) > target_n:
            df = df.sample(n=target_n, random_state=RANDOM_STATE).reset_index(drop=True)

        logger.info(
            f"Returning stratified sample on 'label': "
            f"{len(df)}/{total_len} rows (pct={pct})"
        )

    logger.info(f"Loaded datasets in {time.time() - start:.2f} seconds")
    return df


def clean_data_unsw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hygiene-only cleaning for UNSW-NB15:
    - normalize column names
    - inf/-inf -> NA
    - placeholder strings -> NA for categorical-like columns
    - drop duplicate rows
    - drop 'id' if present
    - ensure 'label' exists and is int
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    df.replace([float("inf"), float("-inf")], np.nan, inplace=True)

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
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed {before - len(df)} duplicate rows")

    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)
        logger.info("Dropped columns: ['id']")

    if "label" not in df.columns:
        raise ValueError("Target column 'label' not found in UNSW-NB15 dataset.")
    df["label"] = df["label"].astype("int64")

    return df


def preprocess_unsw_nb15(df: pd.DataFrame) -> pd.DataFrame:
    """Apply hygiene-only preprocessing to UNSW-NB15."""
    logger.info("Starting preprocessing of UNSW-NB15 dataset...")
    start = time.time()
    df = clean_data_unsw(df)
    logger.info(f"Completed preprocessing in {time.time() - start:.2f} seconds")
    return df


def _to_sklearn_nan(
    df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]
) -> pd.DataFrame:
    """
    Convert Arrow / pandas nullable dtypes into sklearn-friendly numpy/object dtypes.

    - Numeric columns -> float64 with np.nan
    - Categorical columns -> object with None for missing

    This prevents: TypeError: boolean value of NA is ambiguous
    """
    out = df.copy()

    # Numeric: force numpy float64 (np.nan for missing)
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    # Categorical: force plain Python objects (None for missing)
    for c in cat_cols:
        if c in out.columns:
            # Convert to object first (so missing can be represented sanely)
            out[c] = out[c].astype("object")
            # Replace pandas NA / NaN with None (SimpleImputer treats None as missing for object)
            out[c] = out[c].where(pd.notna(out[c]), None)

    return out


def _chronological_split(
    X: pd.DataFrame, y: np.ndarray, test_size: float
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Chronological split (no shuffle): first (1-test_size) train, last test_size test."""
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0,1). Got {test_size}.")

    split_idx = int(len(X) * (1.0 - test_size))
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError(
            f"Bad split index {split_idx} for dataset length {len(X)} with test_size={test_size}."
        )

    X_train = X.iloc[:split_idx]
    y_train = y[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y[split_idx:]
    return X_train, y_train, X_test, y_test


def _train_val_split_from_train(
    X_train: pd.DataFrame, y_train: np.ndarray, val_fraction: float
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """From a chronological train set, take the last val_fraction as validation."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0,1). Got {val_fraction}.")

    split_idx = int(len(X_train) * (1.0 - val_fraction))
    if split_idx <= 0 or split_idx >= len(X_train):
        raise ValueError(
            f"Bad split index {split_idx} for train length {len(X_train)} with val_fraction={val_fraction}."
        )

    X_tr = X_train.iloc[:split_idx]
    y_tr = y_train[:split_idx]
    X_val = X_train.iloc[split_idx:]
    y_val = y_train[split_idx:]
    return X_tr, y_tr, X_val, y_val


def get_pipeline(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Preprocessing pipeline fit on train only and applied to validation."""
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


def process_holdout_data(
    X_tr_raw: pd.DataFrame,
    y_tr: ArrayLike,
    X_val_raw: pd.DataFrame,
    y_val: ArrayLike,
    num_cols: List[str],
    cat_cols: List[str],
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Holdout preprocessing (fit on train only, transform val):
    - impute/scale numeric, impute/one-hot categorical
    - variance threshold
    - RF SelectFromModel top-K features
    - SMOTE on training only
    """
    X_tr_raw = _to_sklearn_nan(X_tr_raw, num_cols=num_cols, cat_cols=cat_cols)
    X_val_raw = _to_sklearn_nan(X_val_raw, num_cols=num_cols, cat_cols=cat_cols)

    pipeline = get_pipeline(num_cols, cat_cols)

    X_tr = pipeline.fit_transform(X_tr_raw)
    X_val = pipeline.transform(X_val_raw)

    vt = VarianceThreshold(threshold=VAR_THRESHOLD)
    X_tr = vt.fit_transform(X_tr)
    X_val = vt.transform(X_val)

    y_tr_arr = np.asarray(y_tr)

    if len(np.unique(y_tr_arr)) > 1:
        selector = SelectFromModel(
            estimator=RandomForestClassifier(
                n_estimators=GINI_N_ESTIMATORS, random_state=random_state, n_jobs=-1
            ),
            max_features=TOP_K_FEATURES,
            threshold=-np.inf,
        )
        X_tr = selector.fit_transform(X_tr, y_tr_arr)
        X_val = selector.transform(X_val)
    else:
        if X_tr.shape[1] > TOP_K_FEATURES:
            X_tr = X_tr[:, :TOP_K_FEATURES]
            X_val = X_val[:, :TOP_K_FEATURES]

    if len(np.unique(y_tr_arr)) > 1:
        smote = SMOTE(random_state=random_state, k_neighbors=SMOTE_K_NEIGHBORS)
        X_tr, y_tr_arr = smote.fit_resample(X_tr, y_tr_arr)

    return X_tr, y_tr_arr, X_val, np.asarray(y_val)


def _prepare_unsw_splits(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, List[str], List[str]]:
    """Build chronological 80/20 and 90/10 splits and return train/val plus column lists."""
    if "label" not in df.columns:
        raise ValueError("Target column 'label' not found.")

    y_all = df["label"].values.astype(int)

    drop_cols = ["label"]
    if "attack_cat" in df.columns:
        drop_cols.append("attack_cat")

    X_all = df.drop(columns=drop_cols)

    X_train, y_train, _, _ = _chronological_split(X_all, y_all, test_size=TEST_SIZE)
    X_tr_raw, y_tr, X_val_raw, y_val = _train_val_split_from_train(
        X_train, y_train, val_fraction=VAL_IN_TRAIN
    )

    cat_cols = X_all.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    num_cols = X_all.select_dtypes(include=["number"]).columns.tolist()

    return X_tr_raw, y_tr, X_val_raw, y_val, num_cols, cat_cols


def save_results(
    study: optuna.Study,
    model_name: str,
    n_trials: int,
    n_splits: int,
    total_time: float,
) -> dict:
    """Save best params to JSON and write a detailed tuning log."""
    avg_trial_time = total_time / n_trials if n_trials > 0 else 0.0
    best_params = study.best_params
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_filename = f"UNSW_NB15_{model_name}_BestParams_{timestamp}.json"
    with open(os.path.join(HYPERPARAMS_DIR, json_filename), "w") as f:
        json.dump(best_params, f, indent=4)

    log_filename = f"{model_name}_{timestamp}.log"
    log_path = os.path.join(RESULTS_DIR, log_filename)

    log_content: List[str] = [
        f"Model: {model_name}",
        "Dataset: UNSW_NB15",
        f"Date: {datetime.now().isoformat()}",
        f"Trials: {n_trials} | CV Splits: {n_splits}",
        "-" * 60,
        f"Total Tuning Time: {total_time:.2f}s",
        f"Avg Trial Time: {avg_trial_time:.2f}s",
        "-" * 60,
        f"Best Validation F1: {study.best_value:.6f}",
        f"Best Val Accuracy:  {study.best_trial.user_attrs.get('avg_accuracy', 0.0):.6f}",
        f"Best Val Precision: {study.best_trial.user_attrs.get('avg_precision', 0.0):.6f}",
        f"Best Val Recall:    {study.best_trial.user_attrs.get('avg_recall', 0.0):.6f}",
        f"Best Hyperparameters:\n{json.dumps(best_params, indent=4)}",
        "-" * 60,
        "Trial History:",
    ]

    header = (
        f"{'Trial':<6} | {'F1':<10} | {'Acc':<10} | {'Prec':<10} | "
        f"{'Rec':<10} | {'State':<10} | {'Dur(s)':<10}"
    )
    log_content.append(header)
    log_content.append("-" * len(header))

    for t in study.trials:
        dur = t.duration.total_seconds() if t.duration else 0.0
        f1 = float(t.value) if t.value is not None else 0.0
        acc = float(t.user_attrs.get("avg_accuracy", 0.0))
        prec = float(t.user_attrs.get("avg_precision", 0.0))
        rec = float(t.user_attrs.get("avg_recall", 0.0))
        log_content.append(
            f"{t.number:<6} | {f1:<10.6f} | {acc:<10.6f} | {prec:<10.6f} | "
            f"{rec:<10.6f} | {t.state.name:<10} | {dur:<10.2f}"
        )

    with open(log_path, "w") as f:
        f.write("\n".join(log_content))

    logger.info(f"Tuning complete. Log saved to {log_path}")
    logger.info(f"Total: {total_time:.2f}s | Avg Trial: {avg_trial_time:.2f}s")
    return best_params


def tune_unsw_nb15_dl(
    model_class: Type[BaseDLModel], n_trials: int, random_state: int
) -> Dict[str, Any]:
    """DL tuning with chronological 80/20 and 90/10 splits, single validation objective."""
    start_time = time.time()

    df = get_unsw_nb15()
    df = preprocess_unsw_nb15(df)

    X_tr_raw, y_tr, X_val_raw, y_val, num_cols, cat_cols = _prepare_unsw_splits(df)

    model_name = model_class.__name__
    logger.info(
        f"Starting DL Tuning for {model_name} (chronological 80/20 test, 90/10 val-in-train)..."
    )

    def objective(trial: optuna.Trial) -> float:
        hp = model_class.sample_hyperparameters(trial)

        lr = float(hp.pop("lr", hp.pop("learning_rate", 1e-3)))
        optimizer_name = str(
            hp.pop("optimizer_name", hp.pop("optimizer", "adam"))
        ).lower()

        X_tr, y_tr_bal, X_val, y_val_local = process_holdout_data(
            X_tr_raw,
            y_tr.astype(np.float32),
            X_val_raw,
            y_val.astype(np.float32),
            num_cols=num_cols,
            cat_cols=cat_cols,
            random_state=random_state,
        )

        if len(np.unique(y_tr_bal)) < 2:
            logger.warning(
                f"Trial {trial.number}: Training set contains only 1 class after preprocessing. Returning 0.0."
            )
            return 0.0

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr_bal)),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val_local)),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        model = model_class(input_dim=X_tr.shape[1], **hp)

        if optimizer_name == "adam":
            opt = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "rmsprop":
            opt = optim.RMSprop(model.parameters(), lr=lr)
        else:
            opt = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)

        m = model.train_model(
            train_loader,
            val_loader,
            opt,
            nn.BCELoss(),
            epochs=EPOCHS,
            patience=PATIENCE,
        )

        trial.set_user_attr("avg_accuracy", float(m.get("accuracy", 0.0)))
        trial.set_user_attr("avg_precision", float(m.get("precision", 0.0)))
        trial.set_user_attr("avg_recall", float(m.get("recall", 0.0)))

        return float(m.get("f1", 0.0))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    with tqdm(total=n_trials, desc=f"Optimizing {model_name}", unit="trial") as pbar:
        study.optimize(
            objective, n_trials=n_trials, callbacks=[lambda s, t: pbar.update(1)]
        )

    return save_results(
        study, model_name, n_trials, n_splits=1, total_time=time.time() - start_time
    )


def tune_unsw_nb15_ml(
    model_class: Type[BaseMLModel], n_trials: int, random_state: int
) -> Dict[str, Any]:
    """ML tuning with chronological 80/20 and 90/10 splits, single validation objective."""
    start_time = time.time()

    df = get_unsw_nb15()
    df = preprocess_unsw_nb15(df)

    X_tr_raw, y_tr, X_val_raw, y_val, num_cols, cat_cols = _prepare_unsw_splits(df)

    model_name = model_class.__name__
    logger.info(
        f"Starting ML Tuning for {model_name} (chronological 80/20 test, 90/10 val-in-train)..."
    )

    def objective(trial: optuna.Trial) -> float:
        hp = model_class.sample_hyperparameters(trial)

        X_tr, y_tr_bal, X_val, y_val_local = process_holdout_data(
            X_tr_raw,
            y_tr,
            X_val_raw,
            y_val,
            num_cols=num_cols,
            cat_cols=cat_cols,
            random_state=random_state,
        )

        if len(np.unique(y_tr_bal)) < 2:
            logger.warning(
                f"Trial {trial.number}: Training set contains only 1 class after preprocessing. Returning 0.0."
            )
            return 0.0

        model = model_class(**hp)
        m = model.train_model(X_tr, y_tr_bal, X_val, y_val_local)

        trial.set_user_attr("avg_accuracy", float(m.get("accuracy", 0.0)))
        trial.set_user_attr("avg_precision", float(m.get("precision", 0.0)))
        trial.set_user_attr("avg_recall", float(m.get("recall", 0.0)))

        return float(m.get("f1", 0.0))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    with tqdm(total=n_trials, desc=f"Optimizing {model_name}", unit="trial") as pbar:
        study.optimize(
            objective, n_trials=n_trials, callbacks=[lambda s, t: pbar.update(1)]
        )

    return save_results(
        study, model_name, n_trials, n_splits=1, total_time=time.time() - start_time
    )


def tune_unsw_nb15(
    model_class: Type[Any], n_trials: int, random_state: int
) -> Dict[str, Any]:
    """Route to DL or ML tuning function based on the model base class."""
    if issubclass(model_class, BaseDLModel):
        logger.info(f"Detected Deep Learning model: {model_class.__name__}")
        return tune_unsw_nb15_dl(model_class, n_trials, random_state)

    if issubclass(model_class, BaseMLModel):
        logger.info(f"Detected ML model: {model_class.__name__}")
        return tune_unsw_nb15_ml(model_class, n_trials, random_state)

    raise ValueError(
        f"Model class {model_class.__name__} must inherit from BaseDLModel or BaseMLModel."
    )


if __name__ == "__main__":
    for model in MODEL_LIST:
        tune_unsw_nb15(
            model_class=model,
            n_trials=N_TRIALS,
            random_state=RANDOM_STATE,
        )
