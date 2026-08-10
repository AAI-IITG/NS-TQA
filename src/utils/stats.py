"""Statistical tests for headline comparisons (journal protocol).

Paired Wilcoxon signed-rank across seeds, bootstrap confidence intervals, and
Holm–Bonferroni correction for the many comparisons in the extension. ``scipy`` is
an allowed extension dependency; if it is present we use its exact/continuity-
corrected Wilcoxon, otherwise we fall back to a self-contained normal-approximation
implementation so the core stays importable without scipy.

All functions take plain Python lists / numpy arrays of per-seed scalars (e.g. one
accuracy per model seed) and never touch torch, so they are cheap to call from any
post-processing script.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

try:                                    # optional, from requirements-ext.txt
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except Exception:                       # pragma: no cover - exercised on lean installs
    _HAVE_SCIPY = False


def wilcoxon(x: Sequence[float], y: Sequence[float], *,
             alternative: str = "two-sided") -> dict:
    """Paired Wilcoxon signed-rank test on the differences ``x - y``.

    Returns ``{"stat", "p", "n", "n_nonzero", "median_diff", "backend"}``. Zero
    differences are dropped (Wilcoxon convention). With scipy we use its exact test
    for small n; otherwise a normal approximation with tie/continuity correction.
    ``p`` is ``nan`` when there are no non-zero differences.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"wilcoxon needs paired samples, got {x.shape} vs {y.shape}")
    d = x - y
    nz = d[d != 0.0]
    out = {"stat": float("nan"), "p": float("nan"), "n": int(d.size),
           "n_nonzero": int(nz.size), "median_diff": float(np.median(d)) if d.size else float("nan")}
    if nz.size == 0:
        out["backend"] = "none(all-zero)"
        return out
    if _HAVE_SCIPY:
        mode = "exact" if nz.size <= 25 else "approx"
        try:
            res = _scipy_stats.wilcoxon(x, y, alternative=alternative,
                                        zero_method="wilcox", mode=mode)
            out["stat"], out["p"], out["backend"] = float(res.statistic), float(res.pvalue), f"scipy-{mode}"
            return out
        except Exception:
            pass  # fall through to numpy approximation
    out.update(_wilcoxon_normal_approx(nz, alternative))
    out["backend"] = "numpy-approx"
    return out


def _wilcoxon_normal_approx(nz: np.ndarray, alternative: str) -> dict:
    """Normal approximation to the signed-rank statistic with tie + continuity correction.

    The one-sided tests use the DIRECTED statistic ``W+`` (sum of positive-difference
    ranks): ``E[W+]=n(n+1)/4``, and a large ``W+`` supports ``x>y``. Using the
    undirected ``min(W+,W-)`` here would invert one-sided p-values (a strongly positive
    effect would report ``p≈1`` under ``alternative='greater'``)."""
    n = nz.size
    ranks = _rankdata(np.abs(nz))
    r_plus = float(ranks[nz > 0].sum())
    r_minus = float(ranks[nz < 0].sum())
    mean = n * (n + 1) / 4.0
    # tie correction on the variance
    _, counts = np.unique(np.abs(nz), return_counts=True)
    tie = (counts ** 3 - counts).sum()
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0
    if var <= 0:
        return {"stat": float(min(r_plus, r_minus)), "p": float("nan")}
    sd = math.sqrt(var)
    Phi = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    if alternative == "greater":            # H1: median(x-y) > 0  => large W+
        p = 1.0 - Phi((r_plus - mean - 0.5) / sd)
    elif alternative == "less":             # H1: median(x-y) < 0  => small W+
        p = Phi((r_plus - mean + 0.5) / sd)
    else:                                   # two-sided on the undirected statistic
        w = min(r_plus, r_minus)
        p = 2.0 * Phi((w - mean + 0.5) / sd)
    return {"stat": float(min(r_plus, r_minus)), "p": float(min(1.0, max(0.0, p)))}


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties averaged — matches scipy.stats.rankdata."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0    # average of the tied block
        i = j + 1
    return ranks


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0, statistic=np.mean) -> dict:
    """Percentile bootstrap CI for a statistic (default the mean) of ``values``."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot = statistic(v[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(statistic(v)), "lo": float(lo), "hi": float(hi), "n": int(v.size)}


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> dict:
    """Holm–Bonferroni step-down correction.

    Returns ``{"reject": [...], "p_adjusted": [...], "order": [...]}`` aligned to the
    input order. NaN p-values are never rejected and carry NaN adjusted values.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    finite = np.isfinite(p)
    order = np.argsort(np.where(finite, p, np.inf))    # NaNs sort last
    adj = np.full(m, np.nan)
    reject = np.zeros(m, dtype=bool)
    running = 0.0
    k = int(finite.sum())
    for rank, i in enumerate(order[:k]):
        factor = m - rank
        a = min(1.0, p[i] * factor)
        running = max(running, a)                      # enforce monotone adjusted p
        adj[i] = running
        reject[i] = running <= alpha
    # once a hypothesis fails, all later (larger-p) ones also fail (step-down)
    failed = False
    for i in order[:k]:
        if failed:
            reject[i] = False
        elif not reject[i]:
            failed = True
    return {"reject": reject.tolist(), "p_adjusted": adj.tolist(), "order": order.tolist()}
