from typing import Dict, Any
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from base_models_abc import BaseMLModel, ArrayLike
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


class RandomForestModel(BaseMLModel):
    """
    Wrapper for Scikit-Learn RandomForestClassifier.
    """

    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    @classmethod
    def sample_hyperparameters(cls, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_jobs": -1,  # Use all cores
            "random_state": 42,
        }

    def train_model(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
    ) -> Dict[str, float]:
        # Fit
        self.model.fit(X_train, y_train)

        # Predict
        y_pred = self.model.predict(X_val)

        # Metrics
        return {
            "f1": float(f1_score(y_val, y_pred, average="binary", zero_division=0)),
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(
                precision_score(y_val, y_pred, average="binary", zero_division=0)
            ),
            "recall": float(
                recall_score(y_val, y_pred, average="binary", zero_division=0)
            ),
        }


class XGBoostModel(BaseMLModel):
    """
    Wrapper for XGBoost Classifier.
    """

    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)

    @classmethod
    def sample_hyperparameters(cls, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "n_jobs": -1,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }

    def train_model(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
    ) -> Dict[str, float]:
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = self.model.predict(X_val)

        return {
            "f1": float(f1_score(y_val, y_pred, average="binary", zero_division=0)),
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(
                precision_score(y_val, y_pred, average="binary", zero_division=0)
            ),
            "recall": float(
                recall_score(y_val, y_pred, average="binary", zero_division=0)
            ),
        }


class LightGBMModel(BaseMLModel):
    """
    Wrapper for Microsoft's LightGBM.
    """

    def __init__(self, **kwargs):
        self.model = LGBMClassifier(**kwargs)

    @classmethod
    def sample_hyperparameters(cls, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", -1, 15),  # -1 is no limit
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
            "n_jobs": -1,
            "random_state": 42,
            "verbosity": -1,
        }

    def train_model(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
    ) -> Dict[str, float]:
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)

        return {
            "f1": float(f1_score(y_val, y_pred, average="binary", zero_division=0)),
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(
                precision_score(y_val, y_pred, average="binary", zero_division=0)
            ),
            "recall": float(
                recall_score(y_val, y_pred, average="binary", zero_division=0)
            ),
        }


class CatBoostModel(BaseMLModel):
    """
    Wrapper for Yandex's CatBoost.
    """

    def __init__(self, **kwargs):
        self.model = CatBoostClassifier(**kwargs)

    @classmethod
    def sample_hyperparameters(cls, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "thread_count": -1,
            "random_seed": 42,
            "verbose": 0,
            "allow_writing_files": False,
        }

    def train_model(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
    ) -> Dict[str, float]:
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        y_pred = self.model.predict(X_val)

        return {
            "f1": float(f1_score(y_val, y_pred, average="binary", zero_division=0)),
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(
                precision_score(y_val, y_pred, average="binary", zero_division=0)
            ),
            "recall": float(
                recall_score(y_val, y_pred, average="binary", zero_division=0)
            ),
        }
