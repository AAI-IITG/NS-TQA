"""Split-conformal answer confidence for the neuro-symbolic yes/no path (WO-7).

The executor returns a signed robustness ``rho`` for each (window, program); the
point answer is ``yes`` iff ``rho > 0`` and ``|rho|`` is a natural, calibrated-free
confidence margin. Split conformal turns that margin into a *distribution-free*
guarantee: given a held-out calibration set, we emit a prediction SET over the two
labels {yes, no} that contains the true answer with probability >= 1 - alpha,
under exchangeability -- no distributional assumption on ``rho``.

Score (nonconformity = how much the evidence CONTRADICTS a candidate label):

    s(x, yes) = -rho ,   s(x, no) = +rho .

Calibrate the ``ceil((n+1)(1-alpha))/n`` empirical quantile ``qhat`` of the
true-label scores ``s(x_i, y_i)`` on n calibration points. The test-time set is

    C(x) = { y in {yes,no} : s(x, y) <= qhat } ,

which resolves to four regimes by the sign/size of ``rho`` relative to ``qhat``:

    rho >  qhat   -> {yes}        (confident yes)
    rho < -qhat   -> {no}         (confident no)
    |rho| <= qhat -> {yes, no}    (ABSTAIN: margin too small to commit)
    qhat < 0 and qhat < rho < -qhat -> {}   (empty: flagged, only if qhat < 0)

The abstain rate and the accuracy restricted to singleton (committed) answers give
a rigorous confidence stratification; comparing calibrate-on-indist coverage under
a distribution SHIFT exposes exchangeability breaking (coverage drops below 1-alpha).
"""
from __future__ import annotations

import math
from typing import Sequence


def _scores_true(rho: Sequence[float], y_true: Sequence[bool]) -> list[float]:
    """s(x_i, y_i): -rho if the true label is yes, +rho if it is no."""
    return [(-r if y else r) for r, y in zip(rho, y_true)]


def conformal_quantile(cal_rho: Sequence[float], cal_y: Sequence[bool], alpha: float) -> float:
    """qhat = the finite-sample-corrected (1-alpha) quantile of true-label scores.

    Uses the ``ceil((n+1)(1-alpha))/n`` rank so the marginal coverage guarantee
    P(y in C) >= 1-alpha holds exactly under exchangeability. Returns +inf when the
    rank exceeds n (alpha too small for the calibration size -> always cover)."""
    s = sorted(_scores_true(cal_rho, cal_y))
    n = len(s)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf
    return s[k - 1]


def predict_set(rho: float, qhat: float) -> tuple[bool, bool]:
    """C(x) as (yes_in, no_in) using s(x,yes)=-rho, s(x,no)=+rho, threshold qhat."""
    return (-rho <= qhat, rho <= qhat)


def evaluate_conformal(cal_rho, cal_y, test_rho, test_y, alpha: float) -> dict:
    """Calibrate on (cal_rho, cal_y) and score the prediction sets on the test split.

    Returns marginal coverage, set-size regime rates, and the *selective* accuracy
    restricted to committed (singleton) answers -- the operational confident subset.
    """
    qhat = conformal_quantile(cal_rho, cal_y, alpha)
    n = len(test_rho)
    covered = singleton = abstain = empty = 0
    sing_correct = sing_total = 0
    for r, y in zip(test_rho, test_y):
        yes_in, no_in = predict_set(r, qhat)
        size = int(yes_in) + int(no_in)
        # true label in set?
        if (y and yes_in) or ((not y) and no_in):
            covered += 1
        if size == 1:
            singleton += 1
            pred = yes_in  # the single label present
            sing_total += 1
            sing_correct += int(pred == y)
        elif size == 2:
            abstain += 1
        else:
            empty += 1
    return {
        "alpha": alpha,
        "target_coverage": round(1.0 - alpha, 4),
        "qhat": (float(qhat) if math.isfinite(qhat) else None),
        "n_test": n,
        "empirical_coverage": round(covered / max(1, n), 4),
        "singleton_rate": round(singleton / max(1, n), 4),
        "abstain_rate": round(abstain / max(1, n), 4),
        "empty_rate": round(empty / max(1, n), 4),
        "selective_accuracy": round(sing_correct / max(1, sing_total), 4) if sing_total else None,
        "selective_n": sing_total,
    }


def coverage_curve(cal_rho, cal_y, test_rho, test_y, alphas: Sequence[float]) -> list[dict]:
    """evaluate_conformal across a grid of alpha (risk levels)."""
    return [evaluate_conformal(cal_rho, cal_y, test_rho, test_y, a) for a in alphas]
