"""Turn a post-hoc saliency map into NS-TQA's explanation tuple (WO-2B).

NS-TQA emits a *critical interval* and a *supporting channel/predicate*. A saliency
map ``[T,C]`` carries no predicate, but it does implicate a time region and a channel,
so we extract the SAME two things for a fair, identical-axes comparison:

  * supporting channel  = the channel with the most total saliency mass;
  * critical interval   = the length-matched window (same width as the oracle's,
                          ``2*radius+1``) with the maximum saliency mass;
  * decisive step       = the centre of that window.

This lets a saliency explanation be scored by the exact same interval-IoU,
supporting-channel-accuracy, and (mask-and-re-answer) counterfactual metrics as the
by-construction explanation.
"""
from __future__ import annotations

from typing import Optional

import torch


def saliency_to_explanation(saliency: torch.Tensor, radius: int, T: int) -> dict:
    """``saliency`` [T,C] -> {channel, t_star, interval}. Window width = 2*radius+1."""
    if saliency is None:
        return {"channel": None, "t_star": None, "interval": None}
    sal = saliency.detach()
    channel = int(sal.sum(0).argmax())               # most-salient channel
    mass = sal.sum(1)                                 # [T] time-importance
    w = min(2 * radius + 1, T)
    # sliding-window sum via cumulative sum -> centre of the max-mass window
    csum = torch.cat([torch.zeros(1, device=mass.device), mass.cumsum(0)])
    best_lo, best_val = 0, -1.0
    for lo in range(0, T - w + 1):
        val = float(csum[lo + w] - csum[lo])
        if val > best_val:
            best_val, best_lo = val, lo
    hi = best_lo + w - 1
    t_star = (best_lo + hi) // 2
    return {"channel": channel, "t_star": int(t_star), "interval": (best_lo, hi)}


def masked_counterfactual(answer_fn, X: torch.Tensor, interval: Optional[tuple]) -> Optional[float]:
    """Zero the cited interval (all channels) and re-answer; 1.0 if the answer flips.

    ``answer_fn(X) -> bool`` is the black-box's answer. This mirrors the signal-level
    intervention used for NS-TQA, so both methods' counterfactual validity is the same
    protocol (does removing the cited evidence change the answer?)."""
    if interval is None:
        return None
    lo, hi = interval
    a0 = answer_fn(X)
    Xm = X.clone()
    Xm[lo:hi + 1, :] = 0.0
    a1 = answer_fn(Xm)
    return float(a0 != a1)
