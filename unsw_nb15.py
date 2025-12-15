import time
import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Type, List

import numpy as np
import pandas as pd

import optuna
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from utils import (
    UNSW_NB15_PATH,
    HYPERPARAMS_DIR,
    MODEL_LIST,
    logger,
)
from base_models_abc import ArrayLike, BaseDLModel, BaseMLModel
from params import N_TRIALS, N_SPLITS, RANDOM_STATE


# -----------------------------
# Tunable constants (ToN-IoT-style training pipeline)
# -----------------------------
VAR_THRESHOLD = 1e-5
RF_ESTIMATORS = 50
MAX_FEATURES = 20

BATCH_SIZE = 4096
EPOCHS = 100
PATIENCE = 5
MOMENTUM = 0.9


# -----------------------------
# DATA LOADING
# -----------------------------
def _find_first_csv(pattern: str):
    """Return first match under UNSW_NB15_PATH; raise FileNotFoundError if none."""
    matches = list(UNSW_NB15_PATH.rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{pattern}' under: {UNSW_NB15_PATH}\n"
            f"Make sure the files are inside datasets/unsw_nb15/."
        )
    return matches[0]


def get_unsw_nb15() -> pd.DataFrame:
    """
    Load UNSW-NB15 train/test CSVs and concatenate.
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

    combined_df = pd.concat([df_train, df_test], ignore_index=True)

    logger.info(f"Loaded datasets in {time.time() - start:.2f} seconds")
    return combined_df


# -----------------------------
# CLEANING (UNSW common practice)
# -----------------------------
def clean_data_unsw(df: pd.DataFrame) -> pd.DataFrame:
    """
    UNSW-NB15 için yaygın veri temizliği (internet/literatür standardı):
    - Kolon isimlerini normalize et
    - inf/-inf -> NA
    - placeholder stringler ('-', 'None', 'null', ...) -> NA
    - duplicate satırları sil
    - identifier kolonlarını sil (en yaygını 'id')
    - label dtype düzelt (0/1)
    Not: encoding/scaling/SMOTE burada yapılmaz (leakage için fold içinde yapılacak).
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    # 1) Replace inf
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)

    # 2) Replace placeholder strings with NA (only for categorical-like cols)
    placeholder_values = {"-", " -", "- ", "None", "none", "NULL", "null", "NaN", "nan", ""}
    obj_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in obj_cols:
        s = df[c].astype("string")
        df[c] = s.where(~s.isin(list(placeholder_values)), pd.NA)

    # 3) Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed {before - len(df)} duplicate rows")

    # 4) Drop identifier column(s) (most commonly 'id')
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)
        logger.info("Dropped columns: ['id']")
    else:
        logger.info("Dropped columns: []")

    # 5) Ensure label exists and fix dtype
    if "label" not in df.columns:
        raise ValueError("Target column 'label' not found in UNSW-NB15 dataset.")

    df["label"] = df["label"].astype("int64")

    return df


def preprocess_unsw_nb15(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess UNSW-NB15 DataFrame (common practice cleaning only)."""
    logger.info("Starting preprocessing of UNSW-NB15 dataset...")
    start = time.time()

    df = clean_data_unsw(df)

    logger.info(f"Completed preprocessing in {time.time() - start:.2f} seconds")
    return df


# -----------------------------
# PIPELINE (ToN-IoT-style training pipeline)
# -----------------------------
def get_pipeline(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Fold içinde kullanılacak preprocessing pipeline:
    - numeric: median impute + standard scale
    - categorical: most_frequent impute + onehot
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


def process_fold_data(
    X_tr_raw: pd.DataFrame,
    y_tr: ArrayLike,
    X_val_raw: pd.DataFrame,
    y_val: ArrayLike,
    pipeline: ColumnTransformer,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ToN-IoT gibi:
      1) fit_transform(train), transform(val)
      2) variance threshold
      3) RF SelectFromModel (MAX_FEATURES)
      4) SMOTE sadece train'e
    """
    X_tr = pipeline.fit_transform(X_tr_raw)
    X_val = pipeline.transform(X_val_raw)

    vt = VarianceThreshold(threshold=VAR_THRESHOLD)
    X_tr = vt.fit_transform(X_tr)
    X_val = vt.transform(X_val)

    # RF feature selection (train only)
    if len(np.unique(y_tr)) > 1:
        selector = SelectFromModel(
            estimator=RandomForestClassifier(
                n_estimators=RF_ESTIMATORS, random_state=random_state, n_jobs=-1
            ),
            max_features=MAX_FEATURES,
            threshold=-np.inf,
        )
        X_tr = selector.fit_transform(X_tr, y_tr)
        X_val = selector.transform(X_val)
    else:
        # edge-case: one class in fold
        if X_tr.shape[1] > MAX_FEATURES:
            X_tr = X_tr[:, :MAX_FEATURES]
            X_val = X_val[:, :MAX_FEATURES]

    # SMOTE only on training
    if len(np.unique(y_tr)) > 1:
        smote = SMOTE(random_state=random_state)
        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

    return X_tr, y_tr, X_val, y_val


# -----------------------------
# RESULTS SAVING
# -----------------------------
def save_results(
    study: optuna.Study,
    model_name: str,
    n_trials: int,
    n_splits: int,
    total_time: float,
) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "dataset": "UNSW-NB15",
        "model": model_name,
        "n_trials": n_trials,
        "n_splits": n_splits,
        "best_f1": float(study.best_value),
        "best_params": study.best_params,
        "avg_accuracy": float(study.best_trial.user_attrs.get("avg_accuracy", 0.0)),
        "avg_precision": float(study.best_trial.user_attrs.get("avg_precision", 0.0)),
        "avg_recall": float(study.best_trial.user_attrs.get("avg_recall", 0.0)),
        "total_time_sec": float(total_time),
        "timestamp": timestamp,
    }

    os.makedirs(HYPERPARAMS_DIR, exist_ok=True)
    out_path = HYPERPARAMS_DIR / f"UNSW_NB15_{model_name}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info(f"[UNSW-NB15] Best params saved to: {out_path}")
    logger.info(
        f"[UNSW-NB15] {model_name} | best_f1={result['best_f1']:.4f} "
        f"| acc={result['avg_accuracy']:.4f} prec={result['avg_precision']:.4f} rec={result['avg_recall']:.4f} "
        f"| time={result['total_time_sec']:.2f}s"
    )
    return result


# -----------------------------
# OPTUNA TUNING (ML) - ToN-IoT style
# -----------------------------
def tune_unsw_nb15_ml(
    model_class: Type[BaseMLModel],
    n_trials: int,
    n_splits: int,
    random_state: int,
) -> Dict[str, Any]:
    start_time = time.time()

    df = get_unsw_nb15()
    df = clean_data_unsw(df)

    # Binary target
    y = df["label"].values

    # IMPORTANT: attack_cat varsa (multi-class label), binary deneyde feature'dan çıkar (leakage riskini azaltır)
    drop_cols = ["label"]
    if "attack_cat" in df.columns:
        drop_cols.append("attack_cat")

    X = df.drop(columns=drop_cols)

    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    model_name = model_class.__name__
    logger.info(f"Starting ML Tuning for {model_name} (UNSW-NB15)...")

    def objective(trial: optuna.Trial) -> float:
        hp = model_class.sample_hyperparameters(trial)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        f1_list, acc_list, prec_list, rec_list = [], [], [], []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr, X_val, y_val = process_fold_data(
                X.iloc[tr_idx], y[tr_idx],
                X.iloc[val_idx], y[val_idx],
                get_pipeline(num_cols, cat_cols),
                random_state,
            )

            if len(np.unique(y_tr)) < 2:
                continue

            model = model_class(**hp)
            m = model.train_model(X_tr, y_tr, X_val, y_val)

            f1_list.append(m["f1"])
            acc_list.append(m["accuracy"])
            prec_list.append(m["precision"])
            rec_list.append(m["recall"])

            trial.report(m["f1"], fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if not f1_list:
            return 0.0

        trial.set_user_attr("avg_accuracy", float(np.mean(acc_list)))
        trial.set_user_attr("avg_precision", float(np.mean(prec_list)))
        trial.set_user_attr("avg_recall", float(np.mean(rec_list)))

        return float(np.mean(f1_list))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    with tqdm(total=n_trials, desc=f"Optimizing {model_name}", unit="trial") as pbar:
        study.optimize(objective, n_trials=n_trials, callbacks=[lambda s, t: pbar.update(1)])

    return save_results(study, model_name, n_trials, n_splits, time.time() - start_time)


# -----------------------------
# OPTUNA TUNING (DL) - ToN-IoT style
# -----------------------------
def tune_unsw_nb15_dl(
    model_class: Type[BaseDLModel],
    n_trials: int,
    n_splits: int,
    random_state: int,
) -> Dict[str, Any]:
    start_time = time.time()

    df = get_unsw_nb15()
    df = clean_data_unsw(df)

    y = df["label"].values.astype(np.float32)

    drop_cols = ["label"]
    if "attack_cat" in df.columns:
        drop_cols.append("attack_cat")

    X = df.drop(columns=drop_cols)

    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    model_name = model_class.__name__
    logger.info(f"Starting DL Tuning for {model_name} (UNSW-NB15)...")

    def objective(trial: optuna.Trial) -> float:
        hp = model_class.sample_hyperparameters(trial)

        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        f1_list, acc_list, prec_list, rec_list = [], [], [], []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr, X_val, y_val = process_fold_data(
                X.iloc[tr_idx], y[tr_idx],
                X.iloc[val_idx], y[val_idx],
                get_pipeline(num_cols, cat_cols),
                random_state,
            )

            if len(np.unique(y_tr)) < 2:
                continue

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

            f1_list.append(m["f1"])
            acc_list.append(m["accuracy"])
            prec_list.append(m["precision"])
            rec_list.append(m["recall"])

            trial.report(m["f1"], fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if not f1_list:
            return 0.0

        trial.set_user_attr("avg_accuracy", float(np.mean(acc_list)))
        trial.set_user_attr("avg_precision", float(np.mean(prec_list)))
        trial.set_user_attr("avg_recall", float(np.mean(rec_list)))

        return float(np.mean(f1_list))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    with tqdm(total=n_trials, desc=f"Optimizing {model_name}", unit="trial") as pbar:
        study.optimize(objective, n_trials=n_trials, callbacks=[lambda s, t: pbar.update(1)])

    return save_results(study, model_name, n_trials, n_splits, time.time() - start_time)


# -----------------------------
# MAIN: Run all models automatically
# -----------------------------
if __name__ == "__main__":
    # 1) Sanity check only
    df = get_unsw_nb15()
    df = preprocess_unsw_nb15(df)
    print(df.shape)
    print(df.dtypes.head(20))
    print(df["label"].value_counts())

    # 2) Full auto benchmark (training/tuning part is ToN-IoT style)
    N_TRIALS_ML = N_TRIALS
    N_TRIALS_DL = max(5, N_TRIALS // 4)  # DL daha yavaş olduğu için daha az trial
    N_SPLITS_RUN = N_SPLITS

    logger.info("========================================")
    logger.info("Starting FULL AUTO benchmark for UNSW-NB15")
    logger.info(f"ML trials={N_TRIALS_ML}, DL trials={N_TRIALS_DL}, splits={N_SPLITS_RUN}")
    logger.info("========================================")

    all_results: List[Dict[str, Any]] = []

    for model_cls in MODEL_LIST:
        name = model_cls.__name__
        logger.info("----------------------------------------")
        logger.info(f"Running model: {name}")
        logger.info("----------------------------------------")

        try:
            if issubclass(model_cls, BaseMLModel):
                res = tune_unsw_nb15_ml(model_cls, N_TRIALS_ML, N_SPLITS_RUN, RANDOM_STATE)
                all_results.append(res)

            elif issubclass(model_cls, BaseDLModel):
                res = tune_unsw_nb15_dl(model_cls, N_TRIALS_DL, N_SPLITS_RUN, RANDOM_STATE)
                all_results.append(res)

            else:
                logger.warning(f"Skipping {name}: not BaseMLModel/BaseDLModel subclass.")

        except Exception as e:
            logger.exception(f"Model {name} failed: {e}")
            continue

    logger.info("========================================")
    logger.info("ALL MODELS COMPLETED")
    logger.info("========================================")

    # Print summary
    if all_results:
        all_results_sorted = sorted(all_results, key=lambda x: x.get("best_f1", 0.0), reverse=True)
        print("\n=== SUMMARY (sorted by best_f1) ===")
        for r in all_results_sorted:
            print(
                f"{r['model']:<25} best_f1={r['best_f1']:.4f} "
                f"acc={r.get('avg_accuracy', 0):.4f} "
                f"prec={r.get('avg_precision', 0):.4f} "
                f"rec={r.get('avg_recall', 0):.4f} "
                f"time={r.get('total_time_sec', 0):.1f}s"
            )
    else:
        print("\nNo results produced (all models failed or were skipped).")
