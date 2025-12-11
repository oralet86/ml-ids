import time
import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, Tuple, Type
from base_models_abc import ArrayLike, BaseDLModel, BaseMLModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset
from utils import TON_IOT_PATH, HYPERPARAMS_DIR, RESULTS_DIR, MODEL_LIST, logger

VAR_THRESHOLD = 1e-5
RF_ESTIMATORS = 50
MAX_FEATURES = 20
BATCH_SIZE = 4096
EPOCHS = 100
PATIENCE = 5
MOMENTUM = 0.9

# SEARCH PARAMETERS
N_TRIALS = 3
N_SPLITS = 5
RANDOM_STATE = 42


def get_toniot() -> pd.DataFrame:
    """Load the TON_IoT dataset from a CSV file into a DataFrame."""
    file = next(TON_IOT_PATH.rglob("train_test_network.csv"))
    if not file:
        raise FileNotFoundError("train_test_network.csv not found.")
    logger.info(f"Found dataset in {file}")
    start = time.time()
    df = pd.read_csv(file, engine="pyarrow", dtype_backend="pyarrow")
    df.columns = df.columns.str.strip()
    end = time.time()
    logger.info(f"Loaded dataset in {end - start:.2f} seconds")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and irrelevant identifier/temporal columns."""
    logger.info("Cleaning data: removing duplicates and identifiers...")

    initial_rows = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

    cols_to_drop = [
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

    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols)
    return df


def get_pipeline(num_cols, cat_cols) -> ColumnTransformer:
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )


def process_fold_data(
    X_tr_raw: pd.DataFrame,
    y_tr: ArrayLike,
    X_val_raw: pd.DataFrame,
    y_val: ArrayLike,
    pipeline: ColumnTransformer,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_tr_processed = pipeline.fit_transform(X_tr_raw)
    X_val_processed = pipeline.transform(X_val_raw)

    var_filter = VarianceThreshold(threshold=VAR_THRESHOLD)
    X_tr_filtered = var_filter.fit_transform(X_tr_processed)
    X_val_filtered = var_filter.transform(X_val_processed)

    if len(np.unique(y_tr)) > 1:
        selector = SelectFromModel(
            estimator=RandomForestClassifier(
                n_estimators=RF_ESTIMATORS, random_state=random_state, n_jobs=-1
            ),
            max_features=MAX_FEATURES,
            threshold=-np.inf,
        )
        X_tr_selected = selector.fit_transform(X_tr_filtered, y_tr)
        X_val_selected = selector.transform(X_val_filtered)
    else:
        if X_tr_filtered.shape[1] > MAX_FEATURES:
            X_tr_selected = X_tr_filtered[:, :MAX_FEATURES]
            X_val_selected = X_val_filtered[:, :MAX_FEATURES]
        else:
            X_tr_selected = X_tr_filtered
            X_val_selected = X_val_filtered

    if len(np.unique(y_tr)) > 1:
        smote = SMOTE(random_state=random_state)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_selected, y_tr)
    else:
        X_tr_res, y_tr_res = X_tr_selected, y_tr

    return X_tr_res, y_tr_res, X_val_selected, y_val


def save_results(
    study: optuna.Study,
    model_name: str,
    n_trials: int,
    n_splits: int,
    total_time: float,
) -> dict:
    """Saves best params to JSON and a readable report to LOG."""
    avg_trial_time = total_time / n_trials if n_trials > 0 else 0
    best_params = study.best_params
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save JSON
    json_filename = f"TON_IoT_{model_name}_BestParams_{timestamp}.json"
    with open(os.path.join(HYPERPARAMS_DIR, json_filename), "w") as f:
        json.dump(best_params, f, indent=4)

    # 2. Save Log
    log_filename = f"{model_name}_{timestamp}.log"
    log_path = os.path.join(RESULTS_DIR, log_filename)

    log_content = [
        f"Model: {model_name}",
        "Dataset: TON_IoT",
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

    header = f"{'Trial':<6} | {'F1':<10} | {'Acc':<10} | {'Prec':<10} | {'Rec':<10} | {'State':<10} | {'Dur(s)':<10}"
    log_content.append(header)
    log_content.append("-" * len(header))

    for t in study.trials:
        dur = t.duration.total_seconds() if t.duration else 0
        f1 = t.value if t.value else 0.0
        acc = t.user_attrs.get("avg_accuracy", 0.0)
        prec = t.user_attrs.get("avg_precision", 0.0)
        rec = t.user_attrs.get("avg_recall", 0.0)
        log_content.append(
            f"{t.number:<6} | {f1:<10.6f} | {acc:<10.6f} | {prec:<10.6f} | {rec:<10.6f} | {t.state.name:<10} | {dur:<10.2f}"
        )

    with open(log_path, "w") as f:
        f.write("\n".join(log_content))

    logger.info(f"Tuning complete. Log saved to {log_path}")
    logger.info(f"Total: {total_time:.2f}s | Avg Trial: {avg_trial_time:.2f}s")

    return best_params


def tune_ton_iot_dl(
    model_class: Type[BaseDLModel], n_trials: int, n_splits: int, random_state: int
) -> Dict[str, Any]:
    """Tuning logic specific for Deep Learning models (using PyTorch DataLoaders)."""
    start_time = time.time()

    # Load Data
    df = get_toniot()
    df = clean_data(df)

    target_col = "label"
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    cat_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    model_name = model_class.__name__
    logger.info(f"Starting DL Tuning for {model_name}...")

    def objective(trial: optuna.Trial) -> float:
        hp = model_class.sample_hyperparameters(trial)
        lr = hp.pop("lr")
        optimizer_name = hp.pop("optimizer_name")

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        metrics_lists = {"f1": [], "acc": [], "prec": [], "rec": []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # Use Helper for all data processing
            X_tr, y_tr, X_val, y_val = process_fold_data(
                X.iloc[train_idx],
                y[train_idx],
                X.iloc[val_idx],
                y[val_idx],
                get_pipeline(num_cols, cat_cols),
                random_state,
            )

            if len(np.unique(y_tr)) < 2:
                logger.warning(
                    f"Trial {trial.number}, Fold {fold}: Training set contains only 1 class. Skipping fold."
                )
                continue

            # DL Specific: Tensors & Loaders
            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
                batch_size=BATCH_SIZE,
                shuffle=False,
            )

            # DL Specific: Model Init & Train
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

            metrics_lists["f1"].append(m["f1"])
            metrics_lists["acc"].append(m["accuracy"])
            metrics_lists["prec"].append(m["precision"])
            metrics_lists["rec"].append(m["recall"])

            trial.report(m["f1"], fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if not metrics_lists["f1"]:
            return 0.0

        trial.set_user_attr("avg_accuracy", np.mean(metrics_lists["acc"]))
        trial.set_user_attr("avg_precision", np.mean(metrics_lists["prec"]))
        trial.set_user_attr("avg_recall", np.mean(metrics_lists["rec"]))

        return np.mean(metrics_lists["f1"])

    # Run Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    with tqdm(total=n_trials, desc=f"Optimizing {model_name}", unit="trial") as pbar:
        study.optimize(
            objective, n_trials=n_trials, callbacks=[lambda s, t: pbar.update(1)]
        )

    return save_results(study, model_name, n_trials, n_splits, time.time() - start_time)


def tune_ton_iot_ml(
    model_class: Type[BaseMLModel], n_trials: int, n_splits: int, random_state: int
) -> Dict[str, Any]:
    """Tuning logic specific for Standard ML models (using Numpy Arrays)."""
    start_time = time.time()

    # Load Data
    df = get_toniot()
    df = clean_data(df)

    target_col = "label"
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    cat_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    model_name = model_class.__name__
    logger.info(f"Starting ML Tuning for {model_name}...")

    def objective(trial: optuna.Trial) -> float:
        hp = model_class.sample_hyperparameters(trial)

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        metrics_lists = {"f1": [], "acc": [], "prec": [], "rec": []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # Use Helper for all data processing
            X_tr, y_tr, X_val, y_val = process_fold_data(
                X.iloc[train_idx],
                y[train_idx],
                X.iloc[val_idx],
                y[val_idx],
                get_pipeline(num_cols, cat_cols),
                random_state,
            )

            if len(np.unique(y_tr)) < 2:
                logger.warning(
                    f"Trial {trial.number}, Fold {fold}: Training set contains only 1 class. Skipping fold."
                )
                continue

            model = model_class(**hp)

            # BaseMLModel.train_model expects raw arrays
            m = model.train_model(X_tr, y_tr, X_val, y_val)

            metrics_lists["f1"].append(m["f1"])
            metrics_lists["acc"].append(m["accuracy"])
            metrics_lists["prec"].append(m["precision"])
            metrics_lists["rec"].append(m["recall"])

            trial.report(m["f1"], fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if not metrics_lists["f1"]:
            return 0.0

        trial.set_user_attr("avg_accuracy", np.mean(metrics_lists["acc"]))
        trial.set_user_attr("avg_precision", np.mean(metrics_lists["prec"]))
        trial.set_user_attr("avg_recall", np.mean(metrics_lists["rec"]))

        return np.mean(metrics_lists["f1"])

    # Run Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    with tqdm(total=n_trials, desc=f"Optimizing {model_name}", unit="trial") as pbar:
        study.optimize(
            objective, n_trials=n_trials, callbacks=[lambda s, t: pbar.update(1)]
        )

    return save_results(study, model_name, n_trials, n_splits, time.time() - start_time)


def tune_ton_iot(
    model_class: Type[Any], n_trials: int, n_splits: int, random_state: int
) -> Dict[str, Any]:
    """
    Automatically routes the model to the correct tuning function (DL vs ML)
    based on its base class.
    """
    if issubclass(model_class, BaseDLModel):
        logger.info(f"Detected Deep Learning model: {model_class.__name__}")
        return tune_ton_iot_dl(model_class, n_trials, n_splits, random_state)

    elif issubclass(model_class, BaseMLModel):
        logger.info(f"Detected ML model: {model_class.__name__}")
        return tune_ton_iot_ml(model_class, n_trials, n_splits, random_state)

    else:
        raise ValueError(
            f"Model class {model_class.__name__} must inherit from "
            "BaseDLModel or BaseMLModel."
        )


if __name__ == "__main__":
    for model in MODEL_LIST:
        tune_ton_iot(
            model_class=model,
            n_trials=N_TRIALS,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
        )
