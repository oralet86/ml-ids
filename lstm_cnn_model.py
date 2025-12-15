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


class CNNLSTMModel(BaseDLModel):
    """
    Hybrid CNN-LSTM architecture.
    Structure:
    1. CNN Feature Extractor (Conv1d -> ReLU -> MaxPool -> Dropout)
    2. LSTM Stack (1-3 Layers)
    3. Dense Stack (0-2 Layers)
    4. Output (Sigmoid)
    """

    def __init__(
        self,
        input_dim: int,
        filters: int,
        kernel_size: int,
        cnn_dropout: float,
        lstm_hidden: int,
        n_lstm_layers: int,
        lstm_dropout: float,
        n_dense_layers: int,
        dense_units: int,
        dense_dropout: float,
        **kwargs,
    ):
        super(CNNLSTMModel, self).__init__()

        # --- 1. CNN Feature Extractor ---
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=filters, kernel_size=kernel_size, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(cnn_dropout),
        )

        # We rely on dynamic shape inference in the forward pass/training loop
        # to handle the connection between CNN output and LSTM input.
        # Initial dummy input_size for LSTM (will be fixed in training loop if needed).
        # We approximate it as filters (channels become features).
        dummy_input_size = filters

        # --- 2. LSTM Stack ---
        self.lstm = nn.LSTM(
            input_size=dummy_input_size,
            hidden_size=lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if n_lstm_layers > 1 else 0,
            bidirectional=False,
        )

        self.lstm_out_dim = lstm_hidden  # Unidirectional

        # --- 3. Dynamic Dense Layers ---
        layers = []
        in_features = self.lstm_out_dim

        for _ in range(n_dense_layers):
            layers.append(nn.Linear(in_features, dense_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dense_dropout))
            in_features = dense_units

        self.dense_block = nn.Sequential(*layers)

        # --- 4. Output Layer ---
        self.output_layer = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Features]

        # 1. Reshape for Conv1d: [Batch, Channels=1, Length=Features]
        x = x.unsqueeze(1)

        # 2. CNN Forward
        x = self.cnn(x)
        # Output shape: [Batch, Filters, New_Length]

        # 3. Prepare for LSTM
        # We permute to [Batch, New_Length, Filters]
        # Treats the spatial length (New_Length) as time steps,
        # and the filters as features per time step.
        x = x.permute(0, 2, 1)

        # 4. LSTM Forward
        lstm_out, _ = self.lstm(x)

        # Take last time step
        x = lstm_out[:, -1, :]

        # 5. Dense Block
        x = self.dense_block(x)

        # 6. Output
        return self.output_layer(x)

    @classmethod
    def sample_hyperparameters(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Combines CNN params with the Legacy Keras LSTM/Dense search space.
        """
        # CNN Params
        filters = trial.suggest_int("filters", 16, 64, step=16)
        kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
        cnn_dropout = trial.suggest_float("cnn_dropout", 0.1, 0.5)

        # LSTM Configuration (Legacy)
        n_lstm_layers = trial.suggest_int("n_lstm_layers", 1, 3)
        lstm_hidden = trial.suggest_int("lstm_hidden", 32, 256, step=32)
        lstm_dropout = trial.suggest_float("lstm_dropout", 0.2, 0.5, step=0.1)

        # Dense Configuration (Legacy 0-2 layers)
        n_dense_layers = trial.suggest_int("n_dense_layers", 0, 2)
        dense_units = 0
        dense_dropout = 0.0

        if n_dense_layers > 0:
            dense_units = trial.suggest_int("dense_units", 16, 128, step=16)
            dense_dropout = trial.suggest_float("dense_dropout", 0.2, 0.5, step=0.1)

        return {
            "filters": filters,
            "kernel_size": kernel_size,
            "cnn_dropout": cnn_dropout,
            "n_lstm_layers": n_lstm_layers,
            "lstm_hidden": lstm_hidden,
            "lstm_dropout": lstm_dropout,
            "n_dense_layers": n_dense_layers,
            "dense_units": dense_units,
            "dense_dropout": dense_dropout,
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "optimizer_name": trial.suggest_categorical("optimizer_name", ["adam"]),
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

                # The CNN output dimension depends on input size, kernel, stride, etc.
                # We calculate it once on the first batch and re-init the LSTM if needed.
                if _ == 0 and not hasattr(self, "_checked_dims"):
                    with torch.no_grad():
                        # Dry run to check shapes
                        x_test = X_batch.unsqueeze(1)
                        x_test = self.cnn(x_test)
                        x_test = x_test.permute(0, 2, 1)
                        # LSTM input size must match the number of channels (filters)
                        # because we permuted to [Batch, Length, Channels]
                        actual_feat_dim = x_test.shape[2]

                        if self.lstm.input_size != actual_feat_dim:
                            self.lstm = nn.LSTM(
                                input_size=actual_feat_dim,
                                hidden_size=self.lstm.hidden_size,
                                num_layers=self.lstm.num_layers,
                                dropout=self.lstm.dropout,
                                batch_first=True,
                                bidirectional=False,
                            ).to(DEVICE)
                    self._checked_dims = True

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
