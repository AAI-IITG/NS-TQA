"""STL-only baseline: hand-written grounding rules + the deterministic executor.

This is the ``no learned perception'' baseline. It answers the SAME compositional
programs with the SAME parameter-free STL executor as NS-TQA, but replaces the
learned perception ``f_theta: X -> mu_hat`` with a set of **fixed, hand-chosen
thresholds** -- the rules a domain expert might write without any training:

    high_c   : the (normalized) channel is more than ``hi`` std above its mean
    low_c    : more than ``|lo|`` std below the mean
    rising_c : the local slope is positive (beyond a fixed scale)
    falling_c: the local slope is negative

Because the model input ``X`` is already train-normalized, sensible fixed cuts are
``hi=+1.0`` / ``lo=-1.0`` (about the 84th / 16th percentile under a Gaussian) and a
unit slope scale. Crucially these thresholds are **not fit to the data**, so they
differ from the privileged train-fit calibrator that defines the labels -- the
baseline is therefore non-circular (it does NOT reduce to the oracle) and measures
how far a fixed-rule grounding can reproduce the privileged predicate semantics.

The point of the comparison: NS-TQA's learned perception adapts to each
channel/dataset/regime, while a fixed rule cannot -- so the gap (especially under
distribution shift) isolates the value of the *learned* perception, separately
from the value of the symbolic executor (which both share).
"""
from __future__ import annotations

import torch

from executor.hard_logic import evaluate
from perception.grounding import Calibrator, ground, predicate_index
from utils.oracle_metrics import group_accuracy


def hand_rule_calibrator(
    n_channels: int,
    hi: float = 1.0,
    lo: float = -1.0,
    slope_scale: float = 1.0,
    a_level: float = 4.0,
    smooth_k: int = 5,
) -> Calibrator:
    """A ``Calibrator`` with FIXED (not data-fit) per-channel thresholds.

    Reuses the standard grounding machinery (``perception.grounding.ground``) so
    the predicate semantics are identical to NS-TQA -- only the thresholds are
    hand-set rather than learned/fit.
    """
    ones = torch.ones(n_channels)
    return Calibrator(hi=ones * float(hi), lo=ones * float(lo),
                      slope_scale=ones * float(slope_scale),
                      a_level=a_level, smooth_k=smooth_k)


@torch.no_grad()
def stl_only_evaluate(
    instances: list,
    n_channels: int,
    hi: float = 1.0,
    lo: float = -1.0,
    slope_scale: float = 1.0,
    a_level: float = 4.0,
    smooth_k: int = 5,
) -> dict:
    """Answer accuracy of the hand-rule grounding + executor over ``instances``.

    Each instance's (normalized) window is grounded with the fixed rules, then the
    privileged program ``phi_star`` is executed on it; the boolean answer is
    compared to ``answer_star``. No training, no model state.
    """
    if not instances:
        return {"n": 0, "answer_accuracy": None, "by_depth": {}}
    cal = hand_rule_calibrator(n_channels, hi, lo, slope_scale, a_level, smooth_k)
    pidx = predicate_index(n_channels)
    pred, gold, depth = [], [], []
    for inst in instances:
        mu, _ = ground(inst.X, cal)
        rho, _ = evaluate(mu, inst.phi_star, pidx, anchor=0)
        pred.append(bool(rho > 0.0))
        gold.append(bool(inst.answer_star))
        depth.append(inst.depth)
    acc = sum(p == g for p, g in zip(pred, gold)) / len(gold)
    by_depth = {int(k): v["accuracy"] for k, v in group_accuracy(depth, pred, gold).items()}
    return {"n": len(gold), "answer_accuracy": acc, "by_depth": by_depth}
