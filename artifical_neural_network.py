from typing import Any, List, Dict
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from utils import DEVICE
from base_models_abc import BaseDLModel


class ArtificialNeuralNetwork(BaseDLModel):
    """
    Standard multi-layer perceptron.
    """

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        units: List[int],
        activations: List[str],
        use_bn: List[bool],
        dropouts: List[float],
    ):
        super(ArtificialNeuralNetwork, self).__init__()
        layers = []
        in_features = input_dim

        for i in range(n_layers):
            layers.append(nn.Linear(in_features, units[i]))

            act_name = activations[i]
            if act_name == "relu":
                layers.append(nn.ReLU())
            elif act_name == "tanh":
                layers.append(nn.Tanh())
            elif act_name == "sigmoid":
                layers.append(nn.Sigmoid())

            if use_bn[i]:
                layers.append(nn.BatchNorm1d(units[i]))

            layers.append(nn.Dropout(dropouts[i]))
            in_features = units[i]

        # Output Layer (Binary)
        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @classmethod
    def sample_hyperparameters(cls, trial: optuna.Trial) -> Dict[str, Any]:
        n_layers = trial.suggest_int("n_layers", 1, 4)
        units = []
        activations = []
        use_bn = []
        dropouts = []

        for i in range(n_layers):
            units.append(trial.suggest_int(f"units_{i}", 16, 512, step=16))
            activations.append(
                trial.suggest_categorical(
                    f"activation_{i}", ["relu", "tanh", "sigmoid"]
                )
            )
            use_bn.append(trial.suggest_categorical(f"use_bn_{i}", [True, False]))
            dropouts.append(trial.suggest_float(f"dropout_{i}", 0.1, 0.5, step=0.1))

        return {
            "n_layers": n_layers,
            "units": units,
            "activations": activations,
            "use_bn": use_bn,
            "dropouts": dropouts,
            "lr": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "optimizer_name": trial.suggest_categorical(
                "optimizer", ["adam", "rmsprop", "sgd"]
            ),
        }

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int = 20,
        patience: int = 3,
    ) -> Dict[str, float]:
        """
        Instance method for training.
        """
        self.to(DEVICE)
        best_val_f1 = 0.0
        patience_counter = 0
        best_model_state = None
        best_metrics = {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

        for _ in range(epochs):
            # --- Training ---
            self.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = (
                    X_batch.to(DEVICE),
                    y_batch.to(DEVICE).float().unsqueeze(1),
                )
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # --- Validation ---
            self.eval()
            y_true = []
            y_pred = []

            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(DEVICE)
                    outputs = self(X_val)
                    predicted = (outputs >= 0.5).float().cpu().numpy()
                    targets = y_val.float().cpu().numpy()
                    y_pred.extend(predicted)
                    y_true.extend(targets)

            val_f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
            val_acc = accuracy_score(y_true, y_pred)
            val_prec = precision_score(
                y_true, y_pred, average="binary", zero_division=0
            )
            val_rec = recall_score(y_true, y_pred, average="binary", zero_division=0)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_metrics = {
                    "f1": val_f1,
                    "accuracy": val_acc,
                    "precision": val_prec,
                    "recall": val_rec,
                }
                best_model_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_model_state:
            self.load_state_dict(best_model_state)

        return best_metrics
