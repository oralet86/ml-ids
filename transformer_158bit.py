from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal, Tuple
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from base_models_abc import BaseDLModel

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


# Quantization utilities
def _fake_quant_symmetric(
    x: torch.Tensor, bits: int, eps: float = 1e-8
) -> torch.Tensor:
    """
    Simple symmetric per-tensor fake quantization for activations.

    This is a lightweight approximation (good enough for QAT experiments).
    If bits <= 0, returns x unchanged.
    """
    if bits <= 0:
        return x
    qmax = float((1 << (bits - 1)) - 1)  # e.g., 127 for 8-bit signed
    max_abs = x.detach().abs().max().clamp_min(eps)
    scale = qmax / max_abs
    x_q = torch.round(x * scale).clamp(-qmax, qmax) / scale
    # STE: pass gradients as if identity
    return x + (x_q - x).detach()


def _ternary_absmean_quant(
    w: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    BitNet b1.58-style ternary quantization using absmean scaling.

    Returns:
      w_hat: quantized weight used in forward (same shape as w)
      alpha: per-out-feature scaling, shape [out_features, 1]
    """
    # Per-output-channel absmean scale (stable and common for linear layers).
    # Shape: [out_features, 1]
    alpha = w.detach().abs().mean(dim=1, keepdim=True).clamp_min(eps)
    w_scaled = w / alpha
    w_tern = torch.round(w_scaled).clamp(-1, 1)  # -> {-1, 0, +1}
    w_hat = w_tern * alpha

    # STE: forward uses w_hat, backward uses gradients wrt original w
    w_hat_ste = w + (w_hat - w).detach()
    return w_hat_ste, alpha


class BitLinear(nn.Module):
    """
    BitNet b1.58-style Linear layer:
      - Maintains FP32 weight parameter
      - Uses ternary {-1,0,1} absmean-quantized weights in forward (STE)

    Optional activation fake-quantization can be enabled via act_bits.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_bits: int = 0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_bits = act_bits

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = _fake_quant_symmetric(x, bits=self.act_bits) if self.act_bits > 0 else x
        wq, _ = _ternary_absmean_quant(self.weight)
        return F.linear(xq, wq, self.bias)


class _TabularFeatureTokenizer(nn.Module):
    """
    Same idea as before: scalar feature -> token embedding using per-feature affine map.

    NOTE: This is not a Linear layer; it is a feature-wise embedding mechanism.
    You can quantize it too, but I leave it FP by default because:
      - it’s small relative to attention/FFN
      - it’s more “embedding-like” than a matrix-multiply
    """

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        if n_features <= 0:
            raise ValueError("n_features must be > 0")
        if d_token <= 0:
            raise ValueError("d_token must be > 0")

        self.n_features = n_features
        self.d_token = d_token

        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"x must have shape [B, {self.n_features}], got {tuple(x.shape)}"
            )
        if x.dtype not in (torch.float32, torch.float64):
            x = x.float()

        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(
            0
        )  # [B,F,D]
        cls = self.cls.expand(tokens.size(0), -1, -1)  # [B,1,D]
        return torch.cat([cls, tokens], dim=1)  # [B,1+F,D]


# BitNet-style Transformer pieces
class _GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)


class _BitFFN(nn.Module):
    """
    FFN using BitLinear layers.
    Supports GELU or GEGLU-style gating.
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int,
        dropout: float,
        act_bits: int,
        use_geglu: bool,
    ) -> None:
        super().__init__()
        hidden = hidden_mult * d_model
        self.use_geglu = use_geglu

        if use_geglu:
            # project to 2*hidden then GEGLU -> hidden
            self.fc1 = BitLinear(d_model, 2 * hidden, bias=True, act_bits=act_bits)
            self.geglu = _GEGLU()
            self.fc2 = BitLinear(hidden, d_model, bias=True, act_bits=act_bits)
        else:
            self.fc1 = BitLinear(d_model, hidden, bias=True, act_bits=act_bits)
            self.fc2 = BitLinear(hidden, d_model, bias=True, act_bits=act_bits)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_geglu:
            h = self.fc1(x)
            h = self.geglu(h)
            h = self.dropout(h)
            h = self.fc2(h)
        else:
            h = self.fc1(x)
            h = F.gelu(h)
            h = self.dropout(h)
            h = self.fc2(h)
        return self.dropout(h)


class _BitSelfAttention(nn.Module):
    """
    Multi-head self-attention with BitLinear projections.

    This replaces nn.MultiheadAttention so we can quantize QKV and output projections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        act_bits: int,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        # BitLinear projections
        self.q_proj = BitLinear(d_model, d_model, bias=False, act_bits=act_bits)
        self.k_proj = BitLinear(d_model, d_model, bias=False, act_bits=act_bits)
        self.v_proj = BitLinear(d_model, d_model, bias=False, act_bits=act_bits)
        self.o_proj = BitLinear(d_model, d_model, bias=False, act_bits=act_bits)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,D] -> [B,H,T,dh]
        b, t, d = x.shape
        x = x.view(b, t, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,H,T,dh] -> [B,T,D]
        b, h, t, dh = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(b, t, h * dh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # attention scores: [B,H,T,T]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # [B,H,T,dh]
        out = self._merge_heads(out)  # [B,T,D]
        out = self.o_proj(out)
        return self.out_drop(out)


class _BitTransformerBlock(nn.Module):
    """
    FlowTransformer-ish block structure, but with BitNet b1.58 quantized linears:
      - PreNorm
      - BitSelfAttention + residual
      - PreNorm
      - BitFFN + residual
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        ffn_mult: int,
        act_bits: int,
        use_geglu: bool,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _BitSelfAttention(
            d_model=d_model, n_heads=n_heads, dropout=dropout, act_bits=act_bits
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = _BitFFN(
            d_model=d_model,
            hidden_mult=ffn_mult,
            dropout=dropout,
            act_bits=act_bits,
            use_geglu=use_geglu,
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x


class _AttentionPooling(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        w = self.score(h).squeeze(-1)  # [B,T]
        a = torch.softmax(w, dim=-1).unsqueeze(-1)  # [B,T,1]
        pooled = (a * x).sum(dim=1)  # [B,D]
        return self.dropout(pooled)


# Main model
class Transformer158Model(BaseDLModel):
    """
    BitNet b1.58-style (ternary-weight) FlowTransformer-inspired model for your suite.

    - Weights in Linear projections are ternary in forward pass (STE).
    - Optional activation fake-quantization (default 8-bit) can be enabled.
    - API compatible with your tuning code (binary, sigmoid + BCELoss).
    """

    def __init__(
        self,
        input_dim: int,
        d_token: int,
        n_blocks: int,
        n_heads: int,
        dropout: float,
        ffn_mult: int = 4,
        use_positional_embedding: bool = False,
        use_geglu: bool = True,
        pooling: Literal["cls", "attn", "mean"] = "cls",
        head_hidden_mult: int = 2,
        act_bits: int = 8,  # BitNet-style often uses low-bit activations; 8 is common.
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
        if ffn_mult <= 0:
            raise ValueError("ffn_mult must be > 0")
        if head_hidden_mult <= 0:
            raise ValueError("head_hidden_mult must be > 0")
        if act_bits not in (0, 4, 8):
            raise ValueError(
                "act_bits should be 0 (off), 4, or 8 for this implementation."
            )

        self.input_dim = input_dim
        self.d_token = d_token
        self.n_tokens = 1 + input_dim
        self.act_bits = act_bits

        self.tokenizer = _TabularFeatureTokenizer(n_features=input_dim, d_token=d_token)

        self.use_positional_embedding = use_positional_embedding
        if self.use_positional_embedding:
            self.pos_emb = nn.Parameter(torch.empty(1, self.n_tokens, d_token))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        else:
            self.register_parameter("pos_emb", None)

        self.blocks = nn.ModuleList(
            [
                _BitTransformerBlock(
                    d_model=d_token,
                    n_heads=n_heads,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                    act_bits=act_bits,
                    use_geglu=use_geglu,
                )
                for _ in range(n_blocks)
            ]
        )
        self.final_ln = nn.LayerNorm(d_token)

        self.pooling: Literal["cls", "attn", "mean"] = pooling
        if pooling == "attn":
            self.pool = _AttentionPooling(d_model=d_token, dropout=dropout)
        else:
            self.pool = None

        # Quantized head (BitLinear)
        head_hidden = head_hidden_mult * d_token
        self.head_fc1 = BitLinear(d_token, head_hidden, bias=True, act_bits=act_bits)
        self.head_fc2 = BitLinear(head_hidden, 1, bias=True, act_bits=act_bits)
        self.head_drop = nn.Dropout(dropout)
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)  # [B,T,D]
        if self.use_positional_embedding:
            tokens = tokens + self.pos_emb

        h = tokens
        for blk in self.blocks:
            h = blk(h)

        h = self.final_ln(h)

        if self.pooling == "cls":
            pooled = h[:, 0, :]
        elif self.pooling == "mean":
            pooled = h[:, 1:, :].mean(dim=1)
        elif self.pooling == "attn":
            pooled = self.pool(h)  # type: ignore[union-attr]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Head
        z = self.head_fc1(pooled)
        z = F.gelu(z)
        z = self.head_drop(z)
        z = self.head_fc2(z)
        return self.out_act(z)

    @classmethod
    def sample_hyperparameters(cls, trial: optuna.Trial) -> Dict[str, Any]:
        # Width: QAT needs a bit more capacity.
        # Do NOT include 16 here unless you're doing a tiny ablation.
        d_token = trial.suggest_categorical("d_token", [32, 64, 128])

        # Heads: keep modest.
        if d_token == 32:
            n_heads = trial.suggest_categorical("n_heads", [1, 2, 4])
        elif d_token == 64:
            n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        else:  # 128
            n_heads = trial.suggest_categorical("n_heads", [4, 8])

        # Depth: QAT is less stable; keep it moderate.
        # If you want extra capacity, go wider before deeper.
        n_blocks = trial.suggest_int("n_blocks", 2, 6)

        # Dropout: keep small; QAT already injects noise.
        dropout = trial.suggest_float("dropout", 0.0, 0.10, step=0.05)

        # FFN: 4 usually best; 2 may underfit once weights ternarize.
        ffn_mult = trial.suggest_categorical("ffn_mult", [4, 6])

        # GEGLU tends to help optimization under quant noise.
        use_geglu = trial.suggest_categorical("use_geglu", [True, True, False])

        # Positional embedding: almost always False for tabular.
        use_positional_embedding = trial.suggest_categorical(
            "use_positional_embedding", [False, False, False, True]
        )

        # Pooling: CLS is simplest and usually stable; attn pooling can help but costs params.
        pooling = trial.suggest_categorical("pooling", ["cls", "attn"])

        # Head width: give it some capacity since encoder weights are ternary.
        head_hidden_mult = trial.suggest_categorical("head_hidden_mult", [2, 4])

        # Activations: for "b1.58" experiment, don't waste trials on act_bits=0.
        # If you want a clean ablation, run a separate experiment.
        act_bits = trial.suggest_categorical("act_bits", [8])

        # Optimizer/LR: Adam is the safest for STE QAT.
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["adam", "adam", "rmsprop"]
        )

        # LR: smaller than FP.
        lr = trial.suggest_float("learning_rate", 8e-5, 6e-4, log=True)

        return {
            "d_token": d_token,
            "n_blocks": n_blocks,
            "n_heads": n_heads,
            "dropout": dropout,
            "ffn_mult": ffn_mult,
            "use_geglu": use_geglu,
            "use_positional_embedding": use_positional_embedding,
            "pooling": pooling,
            "head_hidden_mult": head_hidden_mult,
            "act_bits": act_bits,
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
            self.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE).float().unsqueeze(1)

                optimizer.zero_grad(set_to_none=True)
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            self.eval()
            y_true: List[float] = []
            y_pred: List[float] = []
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(DEVICE)
                    outputs = self(X_val)
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
