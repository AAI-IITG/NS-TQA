"""Exact (hard) STL robustness semantics.

Computes the quantitative robustness rho(mu, phi, t) of a bounded-STL program
``phi`` over a symbolic state ``mu`` using exact min/max recursion. This is the
deterministic executor: no learnable parameters, fully reproducible. Robustness
is a real number; rho>0 means the formula is satisfied, rho<0 violated, |rho| is
the margin.

Symbolic state convention
-------------------------
``mu`` is a tensor of shape [T, P] of fuzzy predicate truths in [0, 1]. A
``Predicate(name, channel)`` is resolved to a column index via a predicate index
map ``pidx: (name, channel) -> p``. The signed leaf robustness is ``2*mu - 1``,
lifting [0,1] truths to [-1,1] margins so the min/max Boolean semantics apply.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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

PredIndex = dict[tuple[str, int], int]


@dataclass
class Trace:
    """Robustness signal over time plus a critical timestep for evidence.

    ``rho`` is the robustness as a function of the anchor time t (shape [T]).
    ``critical_t`` is the timestep that determined the robustness at the anchor
    (the arg of the governing min/max), used to report the evidence interval tau.
    """

    rho: torch.Tensor  # [T]
    critical_t: Optional[int] = None


def _window(t: int, a: int, b: int, T: int) -> tuple[int, int]:
    """Clamp the interval [t+a, t+b] to valid timesteps; returns (lo, hi) inclusive."""
    lo = max(0, t + a)
    hi = min(T - 1, t + b)
    return lo, hi


def robustness(mu: torch.Tensor, phi: Node, pidx: PredIndex) -> torch.Tensor:
    """Return the robustness signal rho(t) of phi over mu, shape [T].

    mu: [T, P] fuzzy truths. Evaluated for every anchor t; callers typically read
    rho[0] (robustness over the whole window from its start) or max over t.
    """
    T = mu.shape[0]

    if isinstance(phi, Predicate):
        key = (phi.name, phi.channel)
        if key not in pidx:
            raise KeyError(f"predicate {phi.canonical()} not in predicate index")
        col = pidx[key]
        return 2.0 * mu[:, col] - 1.0  # [T] signed margin

    if isinstance(phi, Not):
        return -robustness(mu, phi.child, pidx)

    if isinstance(phi, And):
        return torch.minimum(
            robustness(mu, phi.left, pidx), robustness(mu, phi.right, pidx)
        )

    if isinstance(phi, Or):
        return torch.maximum(
            robustness(mu, phi.left, pidx), robustness(mu, phi.right, pidx)
        )

    if isinstance(phi, Eventually):
        child = robustness(mu, phi.child, pidx)  # [T]
        out = torch.full((T,), float("-inf"))
        for t in range(T):
            lo, hi = _window(t, phi.a, phi.b, T)
            if lo > hi:
                out[t] = float("-inf")
            else:
                out[t] = child[lo : hi + 1].max()
        return out

    if isinstance(phi, Always):
        child = robustness(mu, phi.child, pidx)
        out = torch.full((T,), float("inf"))
        for t in range(T):
            lo, hi = _window(t, phi.a, phi.b, T)
            if lo > hi:
                out[t] = float("inf")
            else:
                out[t] = child[lo : hi + 1].min()
        return out

    if isinstance(phi, Until):
        left = robustness(mu, phi.left, pidx)   # [T]
        right = robustness(mu, phi.right, pidx)  # [T]
        out = torch.full((T,), float("-inf"))
        for t in range(T):
            lo, hi = _window(t, phi.a, phi.b, T)
            best = float("-inf")
            for tp in range(lo, hi + 1):
                # phi1 must hold on [t, tp]; phi2 at tp
                left_min = left[t : tp + 1].min() if tp >= t else float("inf")
                val = min(right[tp].item(), left_min.item() if torch.is_tensor(left_min) else left_min)
                best = max(best, val)
            out[t] = best
        return out

    raise TypeError(f"unknown STL node type: {type(phi).__name__}")


def evaluate(
    mu: torch.Tensor, phi: Node, pidx: PredIndex, anchor: int = 0
) -> tuple[float, Trace]:
    """Evaluate phi at a given anchor time; return (scalar robustness, Trace).

    The scalar robustness rho[anchor] is the standard answer for whole-window
    questions. The Trace carries the full rho(t) signal and a critical timestep
    for evidence-interval reporting.
    """
    rho = robustness(mu, phi, pidx)
    crit = _critical_timestep(mu, phi, pidx, anchor)
    return float(rho[anchor]), Trace(rho=rho, critical_t=crit)


def _critical_timestep(
    mu: torch.Tensor, phi: Node, pidx: PredIndex, anchor: int
) -> Optional[int]:
    """Timestep determining robustness at the anchor for the outermost temporal op.

    For Eventually/Always this is the arg of the max/min over the window; for
    other roots it is the anchor itself. Used to report the evidence interval tau.
    """
    T = mu.shape[0]
    if isinstance(phi, Eventually):
        child = robustness(mu, phi.child, pidx)
        lo, hi = _window(anchor, phi.a, phi.b, T)
        if lo > hi:
            return None
        return lo + int(child[lo : hi + 1].argmax())
    if isinstance(phi, Always):
        child = robustness(mu, phi.child, pidx)
        lo, hi = _window(anchor, phi.a, phi.b, T)
        if lo > hi:
            return None
        return lo + int(child[lo : hi + 1].argmin())
    return anchor


def satisfied(mu: torch.Tensor, phi: Node, pidx: PredIndex, anchor: int = 0) -> bool:
    """Boolean answer: is phi satisfied at the anchor (rho > 0)?"""
    rho, _ = evaluate(mu, phi, pidx, anchor)
    return rho > 0.0
