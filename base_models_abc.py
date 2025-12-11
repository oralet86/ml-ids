from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import optuna
import torch.nn as nn
import torch
import numpy as np

ArrayLike = Union[np.ndarray, Any]


class BaseDLModel(nn.Module, ABC):
    """
    Abstract Base Class for all Deep Learning models.
    Enforces standard interfaces for hyperparameter sampling and training.
    """

    def __init__(self):
        super(BaseDLModel, self).__init__()

    @classmethod
    @abstractmethod
    def sample_hyperparameters(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Defines the Optuna search space.
        """
        pass

    @abstractmethod
    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int,
        patience: int,
    ) -> Dict[str, float]:
        """
        Executes the training loop and returns validation metrics.
        """
        pass


class BaseMLModel(ABC):
    """
    Abstract Base Class for Standard ML models (Scikit-Learn/XGBoost style).
    Expects Matrix/Array inputs and single-shot 'fit' training.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the underlying model (e.g., self.model = RandomForestClassifier(**kwargs))
        """
        pass

    @classmethod
    @abstractmethod
    def sample_hyperparameters(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Defines the Optuna search space.
        Returns a dictionary of parameters compatible with __init__.
        """
        pass

    @abstractmethod
    def train_model(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
    ) -> Dict[str, float]:
        """
        Executes the training (fit) and evaluation process.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (for early stopping or score calculation).
            y_val: Validation labels.

        Returns:
            Dict containing 'f1', 'accuracy', 'precision', 'recall' on X_val.
        """
        pass
