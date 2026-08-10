"""Differentiable (soft) STL robustness semantics.

Mirrors ``hard_logic`` but replaces exact min/max with smooth softmin/softmax
controlled by a sharpness ``beta``. As beta -> infinity these converge to the
hard operators. Used only where gradients must flow (e.g. through the perception
encoder or a learned regime assignment); inference always uses ``hard_logic``.

softmax_beta(x) = (1/beta) * logsumexp(beta * x)
softmin_beta(x) = -(1/beta) * logsumexp(-beta * x)
"""
from __future__ import annotations

import torch

from .grammar import (
    Always,
    And,
    Eventually,
    Node,
    Not,
    Or,
    Predicate,
    Until,
)
from .hard_logic import PredIndex, _window


def _softmax(x: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.logsumexp(beta * x, dim=0) / beta


def _softmin(x: torch.Tensor, beta: float) -> torch.Tensor:
    return -torch.logsumexp(-beta * x, dim=0) / beta


def soft_robustness(
    mu: torch.Tensor, phi: Node, pidx: PredIndex, beta: float = 8.0
) -> torch.Tensor:
    """Differentiable robustness signal rho(t), shape [T]. Higher beta -> sharper."""
    T = mu.shape[0]

    if isinstance(phi, Predicate):
        key = (phi.name, phi.channel)
        if key not in pidx:
            raise KeyError(f"predicate {phi.canonical()} not in predicate index")
        return 2.0 * mu[:, pidx[key]] - 1.0

    if isinstance(phi, Not):
        return -soft_robustness(mu, phi.child, pidx, beta)

    if isinstance(phi, And):
        l = soft_robustness(mu, phi.left, pidx, beta)
        r = soft_robustness(mu, phi.right, pidx, beta)
        return _softmin(torch.stack([l, r], dim=0), beta)

    if isinstance(phi, Or):
        l = soft_robustness(mu, phi.left, pidx, beta)
        r = soft_robustness(mu, phi.right, pidx, beta)
        return _softmax(torch.stack([l, r], dim=0), beta)

    if isinstance(phi, Eventually):
        child = soft_robustness(mu, phi.child, pidx, beta)
        out = []
        for t in range(T):
            lo, hi = _window(t, phi.a, phi.b, T)
            if lo > hi:
                out.append(torch.tensor(-1e6, device=mu.device))
            else:
                out.append(_softmax(child[lo : hi + 1], beta))
        return torch.stack(out)

    if isinstance(phi, Always):
        child = soft_robustness(mu, phi.child, pidx, beta)
        out = []
        for t in range(T):
            lo, hi = _window(t, phi.a, phi.b, T)
            if lo > hi:
                out.append(torch.tensor(1e6, device=mu.device))
            else:
                out.append(_softmin(child[lo : hi + 1], beta))
        return torch.stack(out)

    if isinstance(phi, Until):
        left = soft_robustness(mu, phi.left, pidx, beta)
        right = soft_robustness(mu, phi.right, pidx, beta)
        out = []
        for t in range(T):
            lo, hi = _window(t, phi.a, phi.b, T)
            vals = []
            for tp in range(lo, hi + 1):
                left_min = _softmin(left[t : tp + 1], beta) if tp >= t else torch.tensor(1e6, device=mu.device)
                vals.append(_softmin(torch.stack([right[tp], left_min]), beta))
            out.append(_softmax(torch.stack(vals), beta) if vals else torch.tensor(-1e6, device=mu.device))
        return torch.stack(out)

    raise TypeError(f"unknown STL node type: {type(phi).__name__}")
