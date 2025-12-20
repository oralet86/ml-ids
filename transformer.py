from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from base_models_abc import BaseDLModel

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


class _TabularFeatureTokenizer(nn.Module):
    """
    Tokenizer for already-preprocessed numeric tabular features.

    Your pipeline produces dense numeric arrays (after StandardScaler + OneHotEncoder,
    plus VarianceThreshold + SelectFromModel). DataLoader yields:
      (torch.FloatTensor(X), torch.FloatTensor(y))

    Each scalar feature becomes a token embedding:
        token_i = x_i * W_i + b_i, where W_i, b_i in R^{d_token}

    Output token sequence:
        [CLS] + feature_1 + ... + feature_F
        shape: [B, 1 + F, d_token]
    """

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        if n_features <= 0:
            raise ValueError("n_features must be > 0")
        if d_token <= 0:
            raise ValueError("d_token must be > 0")

        self.n_features = n_features
        self.d_token = d_token

        # Per-feature affine parameters: (F, D)
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))

        # Learnable [CLS] token: (1, 1, D)
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape [B, F], float tensor

        Returns:
            tokens: shape [B, 1+F, D]
        """
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"x must have shape [B, {self.n_features}], got {tuple(x.shape)}"
            )
        if x.dtype not in (torch.float32, torch.float64):
            x = x.float()

        # Numerical tokens: [B, F, D]
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        # Prepend CLS: [B, 1, D]
        cls = self.cls.expand(tokens.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)


class _GEGLU(nn.Module):
    """
    GEGLU feed-forward gating variant (common modern Transformer FFN choice).
    """

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * torch.nn.functional.gelu(b)


class _TransformerBlock(nn.Module):
    """
    FlowTransformer-inspired Transformer block:
      - PreNorm
      - Multi-head self-attention + residual
      - PreNorm
      - FFN (optionally GEGLU) + residual
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        ffn_mult: int,
        use_geglu: bool,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        hidden = ffn_mult * d_model

        if use_geglu:
            self.ffn_in = _GEGLU(d_model, hidden)
            self.ffn_out = nn.Linear(hidden, d_model)
        else:
            self.ffn_in = nn.Linear(d_model, hidden)
            self.ffn_out = nn.Linear(hidden, d_model)

        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention (PreNorm)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)

        # FFN (PreNorm)
        h = self.ln2(x)
        if isinstance(self.ffn_in, _GEGLU):
            h = self.ffn_in(h)
        else:
            h = torch.nn.functional.gelu(self.ffn_in(h))
        h = self.ffn_out(h)
        x = x + self.drop2(h)

        return x


class _AttentionPooling(nn.Module):
    """
    Learned attention pooling over tokens (excluding/including CLS depending on your choice).

    This is a simple, fast alternative head that often performs well on tabular transformers.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            pooled: [B, D]
        """
        h = self.ln(x)
        w = self.score(h).squeeze(-1)  # [B, T]
        a = torch.softmax(w, dim=-1).unsqueeze(-1)  # [B, T, 1]
        pooled = (a * x).sum(dim=1)  # [B, D]
        return self.dropout(pooled)


class TransformerModel(BaseDLModel):
    """
    FlowTransformer-inspired model for your benchmark suite (binary classification).

    Key points for compatibility with your TON_IoT tuning script:
      - forward(x) expects x: FloatTensor [B, input_dim]
      - outputs sigmoid probabilities [B, 1] (use nn.BCELoss())
      - sample_hyperparameters() returns keys: lr, optimizer_name, plus model args
      - __init__(input_dim=..., **hp) works exactly as your tuner calls it
    """

    def __init__(
        self,
        input_dim: int,
        d_token: int,
        n_blocks: int,
        n_heads: int,
        dropout: float,
        ffn_mult: int = 4,
        attn_dropout: float = 0.0,
        use_positional_embedding: bool = False,
        use_geglu: bool = True,
        pooling: Literal["cls", "attn", "mean"] = "cls",
        head_hidden_mult: int = 2,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if d_token <= 0:
            raise ValueError("d_token must be > 0")
        if n_blocks <= 0:
            raise ValueError("n_blocks must be > 0")
        if n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if d_token % n_heads != 0:
            raise ValueError("d_token must be divisible by n_heads")
        if not (0.0 <= dropout <= 0.9):
            raise ValueError("dropout must be in [0.0, 0.9]")
        if not (0.0 <= attn_dropout <= 0.9):
            raise ValueError("attn_dropout must be in [0.0, 0.9]")
        if ffn_mult <= 0:
            raise ValueError("ffn_mult must be > 0")
        if head_hidden_mult <= 0:
            raise ValueError("head_hidden_mult must be > 0")

        self.input_dim = input_dim
        self.d_token = d_token
        self.n_tokens = 1 + input_dim  # CLS + features

        # Tokenizer (feature-wise linear embeddings)
        self.tokenizer = _TabularFeatureTokenizer(n_features=input_dim, d_token=d_token)

        # Optional positional embedding (often not necessary for tabular; default False)
        self.use_positional_embedding = use_positional_embedding
        if self.use_positional_embedding:
            self.pos_emb = nn.Parameter(torch.empty(1, self.n_tokens, d_token))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        else:
            self.register_parameter("pos_emb", None)

        # Transformer encoder blocks (FlowTransformer-like: PreNorm blocks)
        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    d_model=d_token,
                    n_heads=n_heads,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                    use_geglu=use_geglu,
                    attn_dropout=attn_dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.final_ln = nn.LayerNorm(d_token)

        # Pooling (head choice matters a lot; provide options)
        self.pooling: Literal["cls", "attn", "mean"] = pooling
        if pooling == "attn":
            self.pool = _AttentionPooling(d_model=d_token, dropout=dropout)
        else:
            self.pool = None

        # MLP head (stronger than single linear in many tabular transformer setups)
        head_hidden = head_hidden_mult * d_token
        self.head = nn.Sequential(
            nn.Linear(d_token, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)  # [B, T, D]
        if self.use_positional_embedding:
            tokens = tokens + self.pos_emb

        h = tokens
        for blk in self.blocks:
            h = blk(h)

        h = self.final_ln(h)  # [B, T, D]

        if self.pooling == "cls":
            pooled = h[:, 0, :]  # CLS
        elif self.pooling == "mean":
            # Mean over feature tokens only (exclude CLS) tends to be more stable than over all tokens
            pooled = h[:, 1:, :].mean(dim=1)
        elif self.pooling == "attn":
            pooled = self.pool(h)  # type: ignore[union-attr]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        logits = self.head(pooled)  # [B, 1]
        return self.out_act(logits)

    @classmethod
    def sample_hyperparameters(cls, trial: optuna.Trial) -> Dict[str, Any]:
        # Width: tabular transformers usually benefit from moderate widths.
        d_token = trial.suggest_categorical("d_token", [16, 32, 64])

        # Heads: keep small; avoid too many heads at small d_token.
        if d_token == 16:
            n_heads = trial.suggest_categorical("n_heads", [1, 2, 4])  # 16%{1,2,4}=0
        elif d_token == 32:
            n_heads = trial.suggest_categorical("n_heads", [1, 2, 4])
        else:  # 64
            n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])

        # Depth: 2–6 is a sweet spot; 1 is often underfit, >6 often pointless/overfit.
        n_blocks = trial.suggest_int("n_blocks", 2, 6)

        # Dropout: keep low for tabular; high dropout hurts precision -> hurts F1.
        dropout = trial.suggest_float("dropout", 0.0, 0.15, step=0.05)
        attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.10, step=0.05)

        # FFN: 4 is usually best; 2 can underfit; 6/8 can overfit + slow.
        ffn_mult = trial.suggest_categorical("ffn_mult", [2, 4])

        # GEGLU often helps; don't waste too many trials toggling it.
        use_geglu = trial.suggest_categorical("use_geglu", [True, False])

        # Positional embedding: for tabular feature tokens it usually doesn't help.
        # Keep it mostly False, but allow it sometimes.
        use_positional_embedding = trial.suggest_categorical(
            "use_positional_embedding", [False, False, True]
        )

        # Pooling/head: CLS and attention pooling are generally safer than mean on IDS.
        pooling = trial.suggest_categorical("pooling", ["cls", "attn"])

        # Head width: 2 is a good default; 4 only if d_token is small.
        head_hidden_mult = trial.suggest_categorical("head_hidden_mult", [1, 2, 2, 4])

        # Optimizer/LR: Adam dominates here. RMSprop sometimes ok. SGD usually worse.
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["adam", "adam", "rmsprop"]
        )

        # LR: concentrate around 3e-4–3e-3 (tabular transformers often like ~1e-3).
        lr = trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True)

        return {
            "d_token": d_token,
            "n_blocks": n_blocks,
            "n_heads": n_heads,
            "dropout": dropout,
            "attn_dropout": attn_dropout,
            "ffn_mult": ffn_mult,
            "use_geglu": use_geglu,
            "use_positional_embedding": use_positional_embedding,
            "pooling": pooling,
            "head_hidden_mult": head_hidden_mult,
            "lr": lr,
            "optimizer_name": optimizer_name,
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
        best_model_state: Optional[Dict[str, torch.Tensor]] = None
        best_metrics = {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

        for _ in range(epochs):
            # --- Training ---
            self.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE).float().unsqueeze(1)  # [B, 1]

                optimizer.zero_grad(set_to_none=True)
                outputs = self(X_batch)  # [B, 1]
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # --- Validation ---
            self.eval()
            y_true: List[float] = []
            y_pred: List[float] = []

            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(DEVICE)
                    outputs = self(X_val)  # [B, 1]
                    preds = (outputs >= 0.5).float().cpu().numpy().reshape(-1)
                    targets = y_val.float().cpu().numpy().reshape(-1)

                    y_pred.extend(preds.tolist())
                    y_true.extend(targets.tolist())

            val_f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
            val_acc = accuracy_score(y_true, y_pred)
            val_prec = precision_score(
                y_true, y_pred, average="binary", zero_division=0
            )
            val_rec = recall_score(y_true, y_pred, average="binary", zero_division=0)

            if val_f1 > best_val_f1:
                best_val_f1 = float(val_f1)
                best_metrics = {
                    "f1": float(val_f1),
                    "accuracy": float(val_acc),
                    "precision": float(val_prec),
                    "recall": float(val_rec),
                }
                best_model_state = {
                    k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        return best_metrics
