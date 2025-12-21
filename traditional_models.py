from __future__ import annotations

from typing import Dict
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from params import RANDOM_STATE


def _spw(y: np.ndarray) -> float:
    y = np.asarray(y)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / pos if pos else 1.0


def _m(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

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


class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    def fit(self, X_train, y_train) -> "RandomForestModel":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def train_model(self, X_train, y_train, X_val, y_val) -> Dict[str, float]:
        self.fit(X_train, y_train)
        return _m(y_val, self.predict(X_val))


class XGBoostModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.0,
            min_child_weight=1,
            reg_lambda=1.0,
            reg_alpha=0.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=1.0,
        )

    def fit(self, X_train, y_train) -> "XGBoostModel":
        self.model.set_params(scale_pos_weight=_spw(y_train))
        self.model.fit(X_train, y_train, verbose=False)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def train_model(self, X_train, y_train, X_val, y_val) -> Dict[str, float]:
        # Keep eval_set behavior for callers that still use train_model()
        self.model.set_params(scale_pos_weight=_spw(y_train))
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return _m(y_val, self.predict(X_val))


class LightGBMModel:
    def __init__(self):
        self.model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
            scale_pos_weight=1.0,
        )

    def fit(self, X_train, y_train) -> "LightGBMModel":
        self.model.set_params(scale_pos_weight=_spw(y_train))
        self.model.fit(np.ascontiguousarray(X_train), y_train)
        return self

    def predict(self, X):
        return self.model.predict(np.ascontiguousarray(X))

    def predict_proba(self, X):
        return self.model.predict_proba(np.ascontiguousarray(X))

    def train_model(self, X_train, y_train, X_val, y_val) -> Dict[str, float]:
        self.fit(X_train, y_train)
        return _m(y_val, self.predict(X_val))


class CatBoostModel:
    def __init__(self):
        self.model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=8,
            l2_leaf_reg=3.0,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            thread_count=-1,
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        )

    def fit(self, X_train, y_train) -> "CatBoostModel":
        self.model.fit(X_train, y_train, verbose=False)
        return self

    def predict(self, X):
        p = self.model.predict(X)
        return np.asarray(p).reshape(-1).astype(np.int64)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def train_model(self, X_train, y_train, X_val, y_val) -> Dict[str, float]:
        # Keep eval_set behavior for callers that still use train_model()
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        return _m(y_val, self.predict(X_val))
