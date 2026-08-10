"""End-to-end baselines: map (signal, encoded query) -> answer directly.

These exist to demonstrate that compositional temporal questions are hard for
end-to-end models, and -- in the necessity experiment -- that such models are
free to exploit a spurious shortcut and therefore collapse under distribution
shift. Three architectures are provided (LSTM, Transformer, temporal CNN) so the
finding is not specific to one inductive bias. All share the contract

    forward(X:[B,T,C], q:[B,query_dim]) -> logit:[B]

The query is a fixed-length encoding of the structured program (template/ops/
predicates/intervals); the model must learn the temporal reasoning implicitly.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(
        self, n_channels: int, query_dim: int, hidden: int = 64, dropout: float = 0.0
    ):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + query_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, X: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # X: [B,T,C], q: [B,query_dim] -> logit [B]
        _, (h, _) = self.lstm(X)
        z = self.dropout(torch.cat([h[-1], q], dim=1))
        return self.head(z).squeeze(-1)


class TransformerBaseline(nn.Module):
    """Transformer encoder over time, mean-pooled, concatenated with the query."""

    def __init__(
        self,
        n_channels: int,
        query_dim: int,
        hidden: int = 64,
        dropout: float = 0.1,
        nhead: int = 4,
        n_layers: int = 2,
        ff_mult: int = 2,
        max_len: int = 256,
    ):
        super().__init__()
        self.in_proj = nn.Linear(n_channels, hidden)
        self.pos = nn.Parameter(torch.zeros(max_len, hidden))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * ff_mult,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + query_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, X: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        B, T, C = X.shape
        h = self.in_proj(X) + self.pos[:T].unsqueeze(0)   # [B,T,hidden]
        h = self.encoder(h)                                # [B,T,hidden]
        pooled = h.mean(dim=1)                             # [B,hidden]
        z = self.dropout(torch.cat([pooled, q], dim=1))
        return self.head(z).squeeze(-1)


class TCNBaseline(nn.Module):
    """Dilated temporal CNN over the signal, global-pooled, concat with query."""

    def __init__(
        self,
        n_channels: int,
        query_dim: int,
        hidden: int = 64,
        dropout: float = 0.1,
        kernel: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_channels
        for d in dilations:
            pad = (kernel - 1) * d // 2  # same length
            layers += [nn.Conv1d(prev, hidden, kernel, padding=pad, dilation=d),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hidden + query_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, X: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        z = X.permute(0, 2, 1)        # [B,C,T]
        z = self.tcn(z)               # [B,hidden,T]
        pooled = z.mean(dim=2)        # [B,hidden] global average pool
        z = torch.cat([pooled, q], dim=1)
        return self.head(z).squeeze(-1)


BASELINES = {
    "lstm": LSTMBaseline,
    "transformer": TransformerBaseline,
    "tcn": TCNBaseline,
}


def make_baseline(name: str, n_channels: int, query_dim: int, **kwargs) -> nn.Module:
    """Construct a baseline by name. Unknown kwargs for a given model are dropped
    so a single config block can drive all three architectures."""
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; choose from {list(BASELINES)}")
    cls = BASELINES[name]
    import inspect
    valid = set(inspect.signature(cls.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return cls(n_channels=n_channels, query_dim=query_dim, **filtered)