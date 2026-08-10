"""Explanation-faithfulness metrics for NS-TQA (the title's "Explainable").

Every NS-TQA answer is produced by executing a transparent program over perceived
health predicates, so the explanation is the *structure that produced the answer*,
not a post-hoc rationalisation. This module measures whether that promise holds,
quantitatively, with four metrics:

* **Counterfactual validity** (structural faithfulness, executor-internal):
  identify the atomic predicate the executor's min/max actually used (blame
  assignment via ``governing_leaf``), flip its perceived truth at the decisive
  timestep, re-execute, and check the answer flips. High validity means the cited
  evidence genuinely *caused* the answer — a property of the architecture, not of
  perception accuracy.
* **Supporting-predicate accuracy** (explanation correctness vs the privileged
  oracle): does the model's governing leaf match the one the privileged grounding
  ``mu_star`` would cite? Bounded by perception quality.
* **Critical-interval IoU** (evidence localisation vs oracle): overlap of the
  model's vs the oracle's decisive evidence interval (a window of ``radius`` around
  the governing timestep). Also bounded by perception quality.
* **Leakage probe** (benchmark soundness): the best single privileged predicate's
  optimistic accuracy at predicting the answer, per depth — must approach chance
  for depth >= 2, else the benchmark is shortcut-solvable.

``governing_leaf`` walks the same exact semantics as ``executor.hard_logic`` and
returns the (atomic predicate, timestep) that is the binding constraint on the
robustness, i.e. the leaf a maintenance engineer should be pointed at.
"""
from __future__ import annotations

from typing import Optional

import torch

from executor.grammar import (Always, And, Eventually, Node, Not, Or, Predicate,
                              Until)
from executor.hard_logic import evaluate, robustness


def _window(t: int, a: int, b: int, T: int) -> tuple[int, int]:
    """Clamp [t+a, t+b] to valid timesteps (inclusive). Mirrors hard_logic."""
    return max(0, t + a), min(T - 1, t + b)


def governing_leaf(
    mu: torch.Tensor, phi: Node, pidx: dict, t: int = 0
) -> tuple[Optional[Predicate], Optional[int]]:
    """The atomic predicate + timestep that determines rho(phi) at anchor ``t``.

    Follows the exact min/max recursion of the executor, descending into the
    branch / timestep that is the binding extremum, so the returned leaf is the
    evidence the answer actually rests on. Returns (None, None) for a temporal
    window that clamps empty (vacuous robustness).
    """
    T = mu.shape[0]
    if isinstance(phi, Predicate):
        return phi, t
    if isinstance(phi, Not):
        return governing_leaf(mu, phi.child, pidx, t)
    if isinstance(phi, And):
        rl = robustness(mu, phi.left, pidx)[t]
        rr = robustness(mu, phi.right, pidx)[t]
        return governing_leaf(mu, phi.left if rl <= rr else phi.right, pidx, t)
    if isinstance(phi, Or):
        rl = robustness(mu, phi.left, pidx)[t]
        rr = robustness(mu, phi.right, pidx)[t]
        return governing_leaf(mu, phi.left if rl >= rr else phi.right, pidx, t)
    if isinstance(phi, Eventually):
        child = robustness(mu, phi.child, pidx)
        lo, hi = _window(t, phi.a, phi.b, T)
        if lo > hi:
            return None, None
        tstar = lo + int(child[lo:hi + 1].argmax())
        return governing_leaf(mu, phi.child, pidx, tstar)
    if isinstance(phi, Always):
        child = robustness(mu, phi.child, pidx)
        lo, hi = _window(t, phi.a, phi.b, T)
        if lo > hi:
            return None, None
        tstar = lo + int(child[lo:hi + 1].argmin())
        return governing_leaf(mu, phi.child, pidx, tstar)
    if isinstance(phi, Until):
        left = robustness(mu, phi.left, pidx)
        right = robustness(mu, phi.right, pidx)
        lo, hi = _window(t, phi.a, phi.b, T)
        if lo > hi:
            return None, None
        best, tp_star = float("-inf"), lo
        for tp in range(lo, hi + 1):
            lm = float(left[t:tp + 1].min()) if tp >= t else float("inf")
            val = min(float(right[tp]), lm)
            if val > best:
                best, tp_star = val, tp
        lm = float(left[t:tp_star + 1].min())
        if float(right[tp_star]) <= lm:
            return governing_leaf(mu, phi.right, pidx, tp_star)
        ti = t + int(left[t:tp_star + 1].argmin())
        return governing_leaf(mu, phi.left, pidx, ti)
    raise TypeError(f"unknown node type {type(phi).__name__}")


def _interval(t: Optional[int], radius: int, T: int) -> Optional[tuple[int, int]]:
    if t is None:
        return None
    return max(0, t - radius), min(T - 1, t + radius)


def _iou(a: Optional[tuple[int, int]], b: Optional[tuple[int, int]]) -> Optional[float]:
    if a is None or b is None:
        return None
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


def _expected_random_iou(gold: Optional[tuple[int, int]], radius: int, T: int) -> Optional[float]:
    """Chance IoU: mean IoU of a uniformly-random radius-window centre vs the gold
    interval (deterministic expectation over all T candidate centres). This is the
    reference that makes the model's critical-interval IoU interpretable, the way the
    leakage probe has chance 0.5."""
    if gold is None:
        return None
    vals = [_iou(_interval(tp, radius, T), gold) for tp in range(T)]
    return sum(vals) / len(vals)


def empirical_chance_localization(gold: Optional[tuple[int, int]], radius: int, T: int,
                                  n_samples: int = 1000, seed: int = 0) -> Optional[dict]:
    """Monte-Carlo chance baseline for localisation (WO-5A).

    Draws ``n_samples`` random windows of the SAME width as the oracle interval, placed
    uniformly in ``[0,T]``, and reports the chance IoU distribution (mean and 95th
    percentile) plus the chance exact-step agreement (P[random centre == gold centre]).
    A model's interval IoU is only meaningful relative to this: if it is not well above
    the 95th chance percentile, "matches the oracle" is not supported."""
    if gold is None:
        return None
    import random as _r
    rng = _r.Random(seed)
    w = 2 * radius + 1
    lo_max = max(0, T - w)
    gold_c = (gold[0] + gold[1]) // 2
    ious, exact = [], 0
    for _ in range(n_samples):
        lo = rng.randint(0, lo_max)
        iv = (lo, min(T - 1, lo + w - 1))
        ious.append(_iou(iv, gold) or 0.0)
        c = (iv[0] + iv[1]) // 2
        exact += int(c == gold_c)
    ious.sort()
    return {"iou_mean": sum(ious) / len(ious),
            "iou_p95": ious[min(len(ious) - 1, int(0.95 * len(ious)))],
            "step_chance": exact / n_samples}


def _counterfactual_flips(mu: torch.Tensor, phi: Node, pidx: dict,
                          leaf: Predicate, t: int) -> bool:
    """mu-LEVEL counterfactual (WO-5B(i)): remove the cited predicate; did the answer
    change? For the deterministic executor this must flip whenever the cited predicate
    is genuinely decisive, so the aggregate should be ~1.0; a deficit indicates a
    supporting-predicate identification bug, not executor unfaithfulness.

    Saturates the governing leaf's whole column to the extreme OPPOSITE its
    decisive value, then re-executes. Removing it at only the critical timestep
    under-reports faithfulness whenever a temporal operator (Eventually/Always)
    has a redundant satisfying step elsewhere in its window; saturating the column
    tests whether the answer genuinely depends on the cited predicate.
    """
    col = pidx[(leaf.name, leaf.channel)]
    rho0, _ = evaluate(mu, phi, pidx, 0)
    target = 0.0 if float(mu[t, col]) > 0.5 else 1.0
    mu_cf = mu.clone()
    mu_cf[:, col] = target
    rho1, _ = evaluate(mu_cf, phi, pidx, 0)
    return (float(rho1) > 0.0) != (float(rho0) > 0.0)


def _agg(xs: list[Optional[float]]) -> tuple[float, int]:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs) if xs else float("nan"), len(xs))


def faithfulness_report(nst, instances: list, radius: int = 2) -> dict:
    """Faithfulness metrics for a ``LearnedNSTQA`` over QA instances.

    Compares the model's executor-derived explanation (over perceived ``mu_hat``)
    against the privileged oracle explanation (over ``mu_star``):

      * ``counterfactual_validity`` — flip-rate of the model's own cited evidence
        (structural; independent of perception accuracy);
      * ``supporting_predicate_accuracy`` — model vs oracle governing leaf match;
      * ``critical_interval_iou`` — overlap of model vs oracle evidence interval;
      * ``critical_step_agreement`` — exact governing-timestep match;
      * ``support_acc_on_correct`` — supporting-predicate accuracy restricted to
        answers the model got right (the cleanest "when right, right for the
        right reason" number).

    Each is also broken down per program depth.
    """
    pidx = nst.pidx
    rows = []
    for inst in instances:
        X = inst.X
        phi = inst.phi_star
        mu_star = inst.mu_star
        depth = inst.depth
        T = X.shape[0]

        mu_hat = nst.perceive(X)
        rho_hat, _ = evaluate(mu_hat, phi, pidx, 0)
        pred_ans = float(rho_hat) > 0.0
        correct = (pred_ans == bool(inst.answer_star))

        h_leaf, h_t = governing_leaf(mu_hat, phi, pidx, 0)
        cf = (_counterfactual_flips(mu_hat, phi, pidx, h_leaf, h_t)
              if h_leaf is not None else None)

        support_acc = iou = step_match = iou_chance = step_chance = None
        if mu_star is not None and h_leaf is not None:
            g_leaf, g_t = governing_leaf(mu_star, phi, pidx, 0)
            if g_leaf is not None:
                support_acc = float(
                    (h_leaf.name, h_leaf.channel) == (g_leaf.name, g_leaf.channel))
                gold_iv = _interval(g_t, radius, T)
                iou = _iou(_interval(h_t, radius, T), gold_iv)
                step_match = float(h_t == g_t)
                iou_chance = _expected_random_iou(gold_iv, radius, T)   # random-interval reference
                step_chance = 1.0 / T                                   # uniform-random-step reference

        rows.append({"depth": depth, "correct": correct, "cf": cf,
                     "support_acc": support_acc, "iou": iou,
                     "step_match": step_match, "iou_chance": iou_chance,
                     "step_chance": step_chance})

    def summarize(subset):
        return {
            "n": len(subset),
            "answer_accuracy": (sum(r["correct"] for r in subset) / len(subset)
                                if subset else float("nan")),
            "counterfactual_validity": _agg([r["cf"] for r in subset])[0],
            "supporting_predicate_accuracy": _agg([r["support_acc"] for r in subset])[0],
            "critical_interval_iou": _agg([r["iou"] for r in subset])[0],
            "critical_interval_iou_chance": _agg([r["iou_chance"] for r in subset])[0],
            "critical_step_agreement": _agg([r["step_match"] for r in subset])[0],
            "critical_step_agreement_chance": _agg([r["step_chance"] for r in subset])[0],
            "support_acc_on_correct": _agg(
                [r["support_acc"] for r in subset if r["correct"]])[0],
        }

    out = summarize(rows)
    depths = sorted({r["depth"] for r in rows})
    out["by_depth"] = {d: summarize([r for r in rows if r["depth"] == d]) for d in depths}
    return out


def leakage_probe(instances: list) -> dict:
    """Optimistic single-predicate leakage per depth (benchmark soundness check).

    For each privileged predicate column, use its anchor-time truth as a 1-feature
    classifier of the answer (best threshold direction, evaluated on the same set —
    deliberately optimistic), and report the BEST column's accuracy per depth. If
    even this optimistic probe is ~0.5 for depth >= 2, no single predicate encodes
    the answer (the executor is non-decorative). Returns {depth: best_acc}.
    """
    by_depth: dict[int, list] = {}
    for inst in instances:
        if inst.mu_star is None:
            continue
        by_depth.setdefault(inst.depth, []).append(inst)
    out = {}
    for d, insts in sorted(by_depth.items()):
        feats = torch.stack([i.mu_star[0] for i in insts])          # [N, P] anchor truths
        y = torch.tensor([float(i.answer_star) for i in insts])      # [N]
        # accuracy of thresholding each feature at 0.5, both polarities
        pred_hi = (feats >= 0.5).float()                              # [N, P]
        acc_hi = (pred_hi == y[:, None]).float().mean(0)             # [P]
        acc_lo = ((1 - pred_hi) == y[:, None]).float().mean(0)
        best = float(torch.maximum(acc_hi, acc_lo).max())
        out[d] = best
    return out
