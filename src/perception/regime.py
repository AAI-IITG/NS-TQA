"""Regime predicates: soft assignment to learned prototypes (OPTIONAL, ablation).

NOTE: in prior work, flexible regime predicates suppressed recovery of the
interpretable level/trend predicates. Keep n_regimes SMALL (<=1-2) or disable.
This module is a stub for the learned-predicate ablation; the core system uses
only the fixed level/trend predicates from grounding.py.
"""
from __future__ import annotations
import torch


class RegimeAssigner(torch.nn.Module):
    def __init__(self, embed_dim: int, n_regimes: int = 1, kappa: float = 1.0):
        super().__init__()
        self.prototypes = torch.nn.Parameter(torch.randn(n_regimes, embed_dim))
        self.kappa = kappa

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        # e: [T, D] -> [T, n_regimes] soft assignment
        d2 = ((e.unsqueeze(1) - self.prototypes.unsqueeze(0)) ** 2).sum(-1)
        return torch.softmax(-d2 / self.kappa, dim=1)
