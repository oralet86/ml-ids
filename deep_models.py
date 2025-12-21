from typing import Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArtificialNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1280),
            nn.ReLU(),
            nn.BatchNorm1d(1280),
            nn.Dropout(0.3),
            nn.Linear(1280, 640),
            nn.ReLU(),
            nn.BatchNorm1d(640),
            nn.Dropout(0.3),
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size=288,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.head = nn.Sequential(
            nn.Linear(288, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: [B, T, F]
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            256,
            hidden_size=240,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        self.head = nn.Sequential(
            nn.Linear(240, 1),
        )

    def forward(self, x):
        # x: [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, T, C]
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


def _default_feature_names(input_dim: int) -> List[str]:
    return [f"f{i}" for i in range(int(input_dim))]


class NoInputEncoder(nn.Module):
    def __init__(self, feature_names: List[str]):
        super().__init__()
        self.feature_names = feature_names

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        xs = [inputs[n].to(torch.float32) for n in self.feature_names]
        return torch.cat(xs, dim=-1)


def _cls_augment(x: torch.Tensor) -> torch.Tensor:
    b, t, f = x.shape

    # (B, T+1, F): append a zero token without allocating a separate tensor
    x2 = F.pad(x, (0, 0, 0, 1), mode="constant", value=0.0)

    # (B, T+1, 1): marker feature with last timestep = 1
    marker = x.new_zeros((b, t + 1, 1))
    marker[:, -1, 0] = 1.0

    return torch.cat([x2, marker], dim=-1)


class MaxPoolHead(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=1).values  # (B,D)


class CustomMultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, key_dim: int, dropout: float):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.inner_dim = self.num_heads * self.key_dim
        self.dropout = float(dropout)

        self.wq = nn.Linear(self.input_dim, self.inner_dim, bias=True)
        self.wk = nn.Linear(self.input_dim, self.inner_dim, bias=True)
        self.wv = nn.Linear(self.input_dim, self.inner_dim, bias=True)

        self.out_proj = nn.Linear(self.inner_dim, self.input_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.num_heads, self.key_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.num_heads, self.key_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        ctx = attn.transpose(1, 2).contiguous().view(b, t, self.inner_dim)
        return self.out_proj(ctx)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim: int, inner_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = CustomMultiHeadAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            key_dim=inner_dim,
            dropout=dropout,
        )
        eps = 1e-6
        self.ln1 = nn.LayerNorm(input_dim, eps=eps)
        self.ln2 = nn.LayerNorm(input_dim, eps=eps)

        self.ff0 = nn.Linear(input_dim, inner_dim)
        self.ff1 = nn.Linear(inner_dim, input_dim)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(inputs)
        x = self.ln1(inputs + attn_out)
        x = F.relu(self.ff0(x))
        x = F.relu(self.ff1(x))
        ff_out = self.drop(x)
        return self.ln2(attn_out + ff_out)


class BasicTransformer(nn.Module):
    def __init__(self, n_layers: int, inner_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_layers = int(n_layers)
        self.inner_dim = int(inner_dim)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)

        self._built_for: Optional[int] = None
        self.blocks = nn.ModuleList()

    def _build(self, input_dim: int) -> None:
        if self._built_for is not None:
            return
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    input_dim=input_dim,
                    inner_dim=self.inner_dim,
                    num_heads=self.n_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.n_layers)
            ]
        )
        self._built_for = int(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._built_for is None:
            self._build(x.size(-1))
        for blk in self.blocks:
            x = blk(x)
        return x


class LastTokenHead(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1, :]


class FlowTransformerRepoCore(nn.Module):
    def __init__(
        self,
        feature_names: List[str],
        *,
        base_input_dim: Optional[int] = None,
        use_cls_augment: bool = True,
        n_layers: int = 5,
        inner_dim: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
        mlp_sizes: Sequence[int] = (128, 64),
        mlp_dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_names = feature_names
        self.use_cls_augment = bool(use_cls_augment)

        self.input_encoder = NoInputEncoder(feature_names)
        self.seq = BasicTransformer(
            n_layers=n_layers, inner_dim=inner_dim, n_heads=n_heads, dropout=dropout
        )
        self.head = MaxPoolHead()

        self._mlp_sizes = [int(x) for x in mlp_sizes]
        self._mlp_dropout = float(mlp_dropout)
        self._mlp: Optional[nn.Sequential] = None
        self._out: Optional[nn.Linear] = None

        if base_input_dim is not None:
            d = int(base_input_dim) + (1 if self.use_cls_augment else 0)
            self.seq._build(d)
            self._ensure_mlp(d)

    def _ensure_mlp(self, in_dim: int) -> None:
        if self._mlp is not None:
            return
        layers: List[nn.Module] = []
        d = int(in_dim)
        for h in self._mlp_sizes:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            if self._mlp_dropout > 0:
                layers.append(nn.Dropout(self._mlp_dropout))
            d = h
        self._mlp = nn.Sequential(*layers) if layers else nn.Sequential()
        self._out = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)

        if self.use_cls_augment:
            x = _cls_augment(x)

        x = self.seq(x)
        x = self.head(x)

        self._ensure_mlp(x.size(-1))
        assert self._mlp is not None and self._out is not None
        x = self._mlp(x)
        return self._out(x)


class FlowTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        feature_names: Optional[List[str]] = None,
        use_cls_augment: bool = True,
        n_layers: int = 5,
        inner_dim: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
        mlp_sizes: Sequence[int] = (128, 64),
        mlp_dropout: float = 0.2,
    ):
        super().__init__()
        names = (
            feature_names
            if feature_names is not None
            else _default_feature_names(input_dim)
        )
        if len(names) != int(input_dim):
            raise ValueError(
                f"len(feature_names)={len(names)} != input_dim={int(input_dim)}"
            )

        self.feature_names = names
        self.core = FlowTransformerRepoCore(
            feature_names=names,
            base_input_dim=int(input_dim),
            use_cls_augment=use_cls_augment,
            n_layers=n_layers,
            inner_dim=inner_dim,
            n_heads=n_heads,
            dropout=dropout,
            mlp_sizes=mlp_sizes,
            mlp_dropout=mlp_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)
