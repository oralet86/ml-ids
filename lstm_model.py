from typing import Dict, Any
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from base_models_abc import BaseDLModel

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


class LSTMModel(BaseDLModel):
    """
    Standard LSTM for tabular data.
    Architecture:
        1. Reshape Input (Batch, Features) -> (Batch, 1, Features)
        2. LSTM Layer(s)
        3. Optional Dense Layer(s)
        4. Output Layer (Sigmoid)
    """

    def __init__(
        self,
        input_dim: int,
        lstm_hidden: int,
        n_lstm_layers: int,
        lstm_dropout: float,
        n_dense_layers: int,
        dense_units: int,
        dense_dropout: float,
        bidirectional: bool = False,
        **kwargs,
    ):
        super(LSTMModel, self).__init__()

        # --- 1. LSTM Feature Extractor ---
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            # PyTorch requires dropout=0 if num_layers=1
            dropout=lstm_dropout if n_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Calculate LSTM output dimension
        self.lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # --- 2. Dynamic Dense Layers ---
        # Stacks 0 to 2 dense layers after the LSTM, as seen in the reference implementation.
        layers = []
        in_features = self.lstm_out_dim

        for _ in range(n_dense_layers):
            layers.append(nn.Linear(in_features, dense_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dense_dropout))
            in_features = dense_units

        self.dense_block = nn.Sequential(*layers)

        # --- 3. Output Layer ---
        self.output_layer = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape: [Batch, Features] -> [Batch, Seq_Len=1, Features]
        # Treats the row of tabular features as a sequence of length 1.
        x = x.unsqueeze(1)

        # LSTM Forward
        # Output: (batch, seq, hidden)
        lstm_out, _ = self.lstm(x)

        # Take the output of the last time step
        x = lstm_out[:, -1, :]

        # Pass through optional dense layers
        x = self.dense_block(x)

        # Final prediction
        return self.output_layer(x)

    @classmethod
    def sample_hyperparameters(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Defines the search space based on the reference architecture.
        """
        # LSTM Configuration
        n_lstm_layers = trial.suggest_int("n_lstm_layers", 1, 3)
        lstm_hidden = trial.suggest_int("lstm_hidden", 32, 256, step=32)
        lstm_dropout = trial.suggest_float("lstm_dropout", 0.2, 0.5, step=0.1)

        # Dense Configuration (0 to 2 layers)
        n_dense_layers = trial.suggest_int("n_dense_layers", 0, 2)
        dense_units = 0
        dense_dropout = 0.0

        # Only suggest dense params if we actually have dense layers to tune
        if n_dense_layers > 0:
            dense_units = trial.suggest_int("dense_units", 16, 128, step=16)
            dense_dropout = trial.suggest_float("dense_dropout", 0.2, 0.5, step=0.1)

        return {
            "n_lstm_layers": n_lstm_layers,
            "lstm_hidden": lstm_hidden,
            "lstm_dropout": lstm_dropout,
            "n_dense_layers": n_dense_layers,
            "dense_units": dense_units,
            "dense_dropout": dense_dropout,
            # Bidirectional is False by default to match reference, but can be enabled if needed
            "bidirectional": False,
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "optimizer_name": trial.suggest_categorical(
                "optimizer_name", ["adam", "rmsprop"]
            ),
        }

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int,
        patience: int,
    ) -> Dict[str, float]:
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

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_metrics = {
                    "f1": float(val_f1),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(
                        precision_score(
                            y_true, y_pred, average="binary", zero_division=0
                        )
                    ),
                    "recall": float(
                        recall_score(y_true, y_pred, average="binary", zero_division=0)
                    ),
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
