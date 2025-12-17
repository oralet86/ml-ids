from __future__ import annotations

import json
import os
import time
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Tuple, Type
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from base_models_abc import ArrayLike, BaseDLModel, BaseMLModel
from params import N_TRIALS, RANDOM_STATE, TEST_SIZE, VAL_IN_TRAIN
from utils import (
    CICIDS2017_PATH,
    HYPERPARAMS_DIR,
    MODEL_LIST,
    RESULTS_DIR,
    logger,
)

# --- Constants ---
BATCH_SIZE = 4096
EPOCHS = 100
PATIENCE = 5
MOMENTUM = 0.9

# Feature selection via Gini importance (train-only)
GINI_N_ESTIMATORS = 50
TOP_K_FEATURES = 20
SMOTE_K_NEIGHBORS = 5


def get_cicids2017() -> pd.DataFrame:
    """
    Load the CICIDS2017 dataset from CSV files into a single DataFrame.
    """
    data_path = str(CICIDS2017_PATH / "*.csv")
    logger.info(f"Loading CICIDS2017 dataset from {data_path}")

    all_files = glob(data_path)
    logger.info(f"Found {len(all_files)} files in {CICIDS2017_PATH}")
    if not all_files:
        raise FileNotFoundError(f"No CSV files found under: {data_path}")

    df_list: List[pd.DataFrame] = []
    start = time.time()
    for file in all_files:
        df = pd.read_csv(file, engine="pyarrow", dtype_backend="pyarrow")
        df.columns = df.columns.str.strip()
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    end = time.time()
    logger.info(f"Loaded {len(all_files)} files in {end - start:.2f} seconds")
    return combined_df


def _resolve_label_column(df: pd.DataFrame, label_hint: str = "Label") -> str:
    """
    Resolve the label column using case/whitespace-insensitive match.
    """
    cols = list(df.columns)
    if label_hint in cols:
        return label_hint

    normalized = [c.strip().lower() for c in cols]
    target_norm = label_hint.strip().lower()
    try:
        return cols[normalized.index(target_norm)]
    except ValueError as e:
        raise ValueError(
            f"Label column not found: {label_hint}. Available columns: {cols[:50]}..."
        ) from e


def preprocess_cicids2017(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess CICIDS2017:
    - strip column names
    - drop duplicated columns
    - replace inf/-inf with NaN and drop NaNs
    - drop duplicate rows
    """
    logger.info("Starting preprocessing of CICIDS2017 dataset...")
    start = time.time()

    df = df.copy()
    df.columns = df.columns.str.strip()

    if df.columns.duplicated().any():
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        logger.info(f"Dropping duplicated CICIDS2017 columns: {dup_cols}")
        df = df.loc[:, ~df.columns.duplicated()]

    before_len = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    logger.info(
        f"Removed {before_len - len(df)} rows due to NaN/Inf; remaining: {len(df)}"
    )

    before_dups = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before_dups - len(df)} duplicate rows")

    end = time.time()
    logger.info(f"Completed preprocessing in {end - start:.2f} seconds")
    return df


def _top_k_by_gini(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    top_k: int,
    random_state: int,
) -> List[str]:
    """
    Compute top-K features using RandomForest Gini importance.
    """
    rf = RandomForestClassifier(
        n_estimators=GINI_N_ESTIMATORS,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
        oob_score=False,
        bootstrap=True,
    )
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    k = min(top_k, len(feature_names))
    top_idx = order[:k]
    return [feature_names[i] for i in top_idx]


def _chronological_split(
    X: pd.DataFrame, y: np.ndarray, test_size: float
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Chronological split (no shuffle): first (1-test_size) for train, last test_size for test.
    """
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
    """
    From an already-chronological train set, take the last `val_fraction` as validation.
    """
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


def process_holdout_data(
    X_tr_raw: pd.DataFrame,
    y_tr: ArrayLike,
    X_val_raw: pd.DataFrame,
    y_val: ArrayLike,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Holdout preprocessing (train-only fit, then transform val):
    - numeric-only
    - drop constant columns (based on train)
    - impute median (fit train only)
    - MinMaxScaler (fit train only)
    - SMOTE on scaled train only
    - RF Gini top-K on balanced train
    - apply same selected features to validation (no SMOTE on val)
    """
    X_tr_num = X_tr_raw.select_dtypes(include=[np.number]).copy()
    X_val_num = X_val_raw.select_dtypes(include=[np.number]).copy()

    if X_tr_num.empty:
        raise ValueError("No numeric features found in training split.")
    if X_val_num.empty:
        raise ValueError("No numeric features found in validation split.")

    nunique = X_tr_num.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X_tr_num = X_tr_num.drop(columns=constant_cols)
        X_val_num = X_val_num.drop(columns=constant_cols, errors="ignore")

    if X_tr_num.shape[1] == 0:
        raise ValueError("No non-constant numeric features remain after filtering.")

    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr_num)
    X_val_imp = imputer.transform(X_val_num)

    scaler = MinMaxScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    if len(np.unique(y_tr)) > 1:
        smote = SMOTE(random_state=random_state, k_neighbors=SMOTE_K_NEIGHBORS)
        X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_scaled, y_tr)
    else:
        X_tr_bal, y_tr_bal = X_tr_scaled, y_tr

    feature_names = X_tr_num.columns.tolist()

    if len(np.unique(y_tr_bal)) > 1 and X_tr_bal.shape[1] > 1:
        top_features = _top_k_by_gini(
            X_train=X_tr_bal,
            y_train=np.asarray(y_tr_bal),
            feature_names=feature_names,
            top_k=TOP_K_FEATURES,
            random_state=random_state,
        )
        top_idx = [feature_names.index(f) for f in top_features]
        X_tr_final = X_tr_bal[:, top_idx]
        X_val_final = X_val_scaled[:, top_idx]
    else:
        X_tr_final = X_tr_bal
        X_val_final = X_val_scaled

    return X_tr_final, y_tr_bal, X_val_final, y_val


def save_results(
    study: optuna.Study,
    model_name: str,
    n_trials: int,
    n_splits: int,
    total_time: float,
) -> dict:
    """Saves best params to JSON and a readable report to LOG."""
    avg_trial_time = total_time / n_trials if n_trials > 0 else 0.0
    best_params = study.best_params
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_filename = f"CICIDS2017_{model_name}_BestParams_{timestamp}.json"
    with open(os.path.join(HYPERPARAMS_DIR, json_filename), "w") as f:
        json.dump(best_params, f, indent=4)

    log_filename = f"{model_name}_{timestamp}.log"
    log_path = os.path.join(RESULTS_DIR, log_filename)

    log_content = [
        f"Model: {model_name}",
        "Dataset: CICIDS2017",
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


def tune_cicids2017_dl(
    model_class: Type[BaseDLModel], n_trials: int, random_state: int
) -> Dict[str, Any]:
    """
    DL tuning with:
      - chronological 80/20 test split
      - chronological 90/10 split inside train (for validation)
    """
    start_time = time.time()

    df = get_cicids2017()
    df = preprocess_cicids2017(df)

    label_col = _resolve_label_column(df, "Label")
    y_all = (df[label_col].astype(str).str.strip() != "BENIGN").astype(int).values
    X_all = df.drop(columns=[label_col])

    # 80/20 chronological train/test
    X_train, y_train, _, _ = _chronological_split(X_all, y_all, test_size=TEST_SIZE)

    # 90/10 chronological train/val inside train
    X_tr_raw, y_tr, X_val_raw, y_val = _train_val_split_from_train(
        X_train, y_train, val_fraction=VAL_IN_TRAIN
    )

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
            X_tr_raw, y_tr, X_val_raw, y_val, random_state=random_state
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

    # n_splits is now "1 holdout", but keep arg for log format compatibility
    return save_results(
        study, model_name, n_trials, n_splits=1, total_time=time.time() - start_time
    )


def tune_cicids2017_ml(
    model_class: Type[BaseMLModel], n_trials: int, random_state: int
) -> Dict[str, Any]:
    """
    ML tuning with:
      - chronological 80/20 test split
      - chronological 90/10 split inside train (for validation)
    """
    start_time = time.time()

    df = get_cicids2017()
    df = preprocess_cicids2017(df)

    label_col = _resolve_label_column(df, "Label")
    y_all = (df[label_col].astype(str).str.strip() != "BENIGN").astype(int).values
    X_all = df.drop(columns=[label_col])

    # 80/20 chronological train/test
    X_train, y_train, _, _ = _chronological_split(X_all, y_all, test_size=TEST_SIZE)

    # 90/10 chronological train/val inside train
    X_tr_raw, y_tr, X_val_raw, y_val = _train_val_split_from_train(
        X_train, y_train, val_fraction=VAL_IN_TRAIN
    )

    model_name = model_class.__name__
    logger.info(
        f"Starting ML Tuning for {model_name} (chronological 80/20 test, 90/10 val-in-train)..."
    )

    def objective(trial: optuna.Trial) -> float:
        hp = model_class.sample_hyperparameters(trial)

        X_tr, y_tr_bal, X_val, y_val_local = process_holdout_data(
            X_tr_raw, y_tr, X_val_raw, y_val, random_state=random_state
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


def tune_cicids2017(
    model_class: Type[Any], n_trials: int, random_state: int
) -> Dict[str, Any]:
    """
    Automatically routes the model to the correct tuning function (DL vs ML)
    based on its base class.
    """
    if issubclass(model_class, BaseDLModel):
        logger.info(f"Detected Deep Learning model: {model_class.__name__}")
        return tune_cicids2017_dl(model_class, n_trials, random_state)

    if issubclass(model_class, BaseMLModel):
        logger.info(f"Detected ML model: {model_class.__name__}")
        return tune_cicids2017_ml(model_class, n_trials, random_state)

    raise ValueError(
        f"Model class {model_class.__name__} must inherit from BaseDLModel or BaseMLModel."
    )


if __name__ == "__main__":
    for model in MODEL_LIST:
        tune_cicids2017(
            model_class=model,
            n_trials=N_TRIALS,
            random_state=RANDOM_STATE,
        )
