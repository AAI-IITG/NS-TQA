"""Leakage-SAFE compositional anomaly QA (WO-1C).

Phase G found the pure-anomaly benchmark is *answer-shaped*: questions dominated by
``anomalous`` near end-of-life are trivially answerable from a single predicate
(leakage probe > 0.55 at depth >= 2), which makes the executor decorative. This
module keeps the learned ``anomalous`` predicate but constructs questions so the
executor stays load-bearing, via three mitigations from the work order:

  (a) FULL-LIFE windows — anomaly questions are drawn from windows spanning the
      whole trajectory (``window_all``), never only the near-failure tail, so the
      ``anomalous`` truth is not a proxy for "we are at end of life".
  (b) COMPOSITION constraint — at depth >= 2, any program containing an
      ``anomalous`` leaf must ALSO contain >= 1 non-anomaly (level/trend) leaf, so
      no answer is decidable from the anomaly predicate alone.
  (c) LEAKAGE-PROBE rejection — during construction, each depth's balanced batch is
      measured with the single-predicate probe; batches exceeding ``leak_max`` are
      rebuilt (up to a retry budget), keeping the lowest-leakage batch achieved.

Everything else (privileged 5-family ``mu_star_5``, non-circular labels, healthy
baseline, evaluators) is reused from ``benchmark.anomaly_qa``; this module only
changes the SAMPLER and adds the probe gate around the balanced build.
"""
from __future__ import annotations

import math
import random
from typing import Optional

import torch

from executor.grammar import Predicate, validate
from executor.hard_logic import evaluate
from perception.anomaly import HealthyBaseline, healthy_windows
from perception.grounding import Calibrator, ground

from .adapters import (ChannelNormalizer, RealInstance, Series, Window,
                       split_by_unit, window_all)
from .anomaly_qa import (_LEVEL_FAMILIES, FIVE_FAMILIES, anomaly_leakage_probe,
                         mu_star_5, pidx5, sample_anomaly_program)
from .templates import Question


# --------------------------------------------------------------------------- #
# (b) composition constraint on programs
# --------------------------------------------------------------------------- #

def _leaf_families(node) -> list[str]:
    """Collect the family name of every atomic predicate leaf under ``node``."""
    if isinstance(node, Predicate):
        return [node.name]
    fams: list[str] = []
    for attr in ("child", "left", "right"):
        sub = getattr(node, attr, None)
        if sub is not None:
            fams.extend(_leaf_families(sub))
    return fams


def is_composition_safe(phi, depth: int) -> bool:
    """(b): at depth>=2, an anomaly-containing program must also contain a
    non-anomaly leaf. Pure-level programs and depth-1 programs are always fine."""
    if depth < 2:
        return True
    fams = _leaf_families(phi)
    has_anom = "anomalous" in fams
    has_level = any(f in _LEVEL_FAMILIES for f in fams)
    return (not has_anom) or has_level


def sample_safe_anomaly_program(depth, channels, T, rng, anomaly_p=0.5,
                                allow_until=False, max_tries=40):
    """Sample a depth-d 5-family program satisfying the composition constraint (b).

    Rejection-samples from the base ``sample_anomaly_program`` until (b) holds; the
    constraint is easy to meet (most mixed programs already satisfy it), so the
    retry budget is rarely exhausted. Returns ``None`` if it cannot (caller skips)."""
    for _ in range(max_tries):
        phi = sample_anomaly_program(depth, channels, T, rng, anomaly_p, allow_until)
        if is_composition_safe(phi, depth):
            return phi
    return None


# --------------------------------------------------------------------------- #
# (c) leakage-probe-gated balanced build
# --------------------------------------------------------------------------- #

def heldout_leakage(instances, seed: int = 0, n_splits: int = 5) -> float:
    """HONEST single-predicate leakage: pick the best (column, sign, threshold) rule
    on a fit half, score it on a disjoint eval half, AVERAGED over ``n_splits``
    random splits (the average kills the selection-on-noise variance of a single
    split at small n).

    The project's ``anomaly_leakage_probe`` fits AND scores on the same instances,
    so with P=5C candidate columns it is optimistically biased (its chance ceiling
    at n≈100 is ~0.66, above the 0.55 gate). Held-out selection removes that
    multiple-comparison optimism, so a genuinely non-leaky depth lands near 0.5."""
    n = len(instances)
    if n < 8:
        return float(anomaly_leakage_probe(instances).get(instances[0].depth, 0.0)) if n else 0.0
    feats = torch.stack([i.mu_star.mean(0) for i in instances])      # [n, P]
    y = torch.tensor([float(i.answer_star) for i in instances])
    P = feats.shape[1]
    half = n // 2
    evals = []
    for s in range(n_splits):
        g = torch.Generator().manual_seed(int(seed) * 97 + s)
        perm = torch.randperm(n, generator=g)
        fit, ev = perm[:half], perm[half:]
        Ff, yf, Fe, ye = feats[fit], y[fit], feats[ev], y[ev]
        best_fit, best_eval = -1.0, 0.5
        for col in range(P):
            thr = Ff[:, col].median()
            for sign in (1.0, -1.0):
                acc_f = ((sign * Ff[:, col] > sign * thr) == (yf > 0.5)).float().mean().item()
                acc_f = max(acc_f, 1 - acc_f)
                if acc_f > best_fit:                                  # select on fit only
                    best_fit = acc_f
                    pred_e = (sign * Fe[:, col] > sign * thr)
                    best_eval = (pred_e == (ye > 0.5)).float().mean().item()
                    best_eval = max(best_eval, 1 - best_eval)
        evals.append(best_eval)
    return float(sum(evals) / len(evals))


def _depth_leakage(instances, seed: int = 0) -> float:
    """Honest held-out single-predicate leakage for a single-depth instance list."""
    if not instances:
        return 0.0
    return heldout_leakage(instances, seed=seed)


def _build_one_depth(windows, d, regime, n_per_depth, C, T, cal, baseline,
                     normalizer, pidx, anomaly_p, allow_until, over_factor, rng,
                     cache):
    """Build a yes/no-balanced batch of depth-d instances with the SAFE sampler."""
    channels = list(range(C))
    target = max(1, n_per_depth // 2)
    yes, no = [], []
    attempts, cap = 0, max(1, n_per_depth) * over_factor

    def ground_window(w):
        key = (w.unit_id, w.start)
        if key not in cache:
            Xn = normalizer.transform(w.X)
            cache[key] = (Xn, mu_star_5(Xn, cal, baseline))
        return cache[key]

    while (len(yes) < target or len(no) < target) and attempts < cap:
        attempts += 1
        w = rng.choice(windows)
        phi = sample_safe_anomaly_program(d, channels, T, rng, anomaly_p, allow_until)
        if phi is None:
            continue
        try:
            validate(phi, n_predicates=C)      # pidx5 has 5C cols but validate counts channels*families internally
        except Exception:
            continue
        Xn, mu = ground_window(w)
        rho, tr = evaluate(mu, phi, pidx, anchor=0)
        if not math.isfinite(rho):
            continue
        ans = bool(rho > 0.0)
        if (ans and len(yes) >= target) or ((not ans) and len(no) >= target):
            continue
        inst = RealInstance(
            X=Xn, question=Question(template=f"safe_anom_d{d}", program=phi,
                                    bindings={"depth": d}, text=None),
            phi_star=phi, answer_star=ans, depth=d, unit_id=w.unit_id,
            n_channels=C, condition=w.condition, regime=regime,
            mu_star=mu, rho_star=float(rho), spurious_channel=None,
            provenance={"dataset": w.dataset, "start": w.start, "five_family": True,
                        "safe": True, "critical_t": tr.critical_t})
        (yes if ans else no).append(inst)
    keep = yes[:target] + no[:target]
    rng.shuffle(keep)
    return keep


def _build_balanced_safe(windows, regime, depths, n_per_depth, C, T, cal, baseline,
                         normalizer, pidx, anomaly_p, allow_until, over_factor, rng,
                         leak_max=0.55, max_leak_retries=6):
    """Balanced build with the (c) leakage gate: per depth, rebuild the batch until
    its single-predicate probe <= ``leak_max`` or the retry budget is spent (then
    keep the lowest-leakage batch). Returns (instances, per_depth_leak_report)."""
    if not windows:
        return [], {}
    cache: dict = {}
    out: list[RealInstance] = []
    report: dict = {}
    for d in depths:
        best_batch, best_leak = [], 1.0
        for attempt in range(max_leak_retries):
            batch = _build_one_depth(windows, d, regime, n_per_depth, C, T, cal,
                                     baseline, normalizer, pidx, anomaly_p,
                                     allow_until, over_factor, rng, cache)
            leak = _depth_leakage(batch, seed=1000 * d + attempt) if d >= 2 else 0.0
            if leak < best_leak:
                best_batch, best_leak = batch, leak
            if d < 2 or leak <= leak_max:
                break
        out.extend(best_batch)
        report[d] = {"leak": round(best_leak, 4), "n": len(best_batch)}
    return out, report


# --------------------------------------------------------------------------- #
# benchmark builder (mirrors build_anomaly_benchmark, safe sampler + probe gate)
# --------------------------------------------------------------------------- #

def build_safe_anomaly_benchmark(
    adapter, *, T=32, stride=2, depths=(1, 2, 3, 4),
    shift="condition", train_conditions=("35Hz12kN",),
    test_conditions=("37.5Hz11kN", "40Hz10kN"),
    indist_holdout_frac=0.3, healthy_frac=0.25,
    n_train_per_depth=200, n_test_per_depth=120,
    hi_q=0.85, lo_q=0.15, smooth_k=5, a_level=4.0, a_anom=4.0, anomaly_q=0.99,
    anomaly_p=0.5, allow_until=False, max_windows_per_unit: Optional[int] = 200,
    over_factor=120, seed=0, leak_max=0.55, max_leak_retries=6,
) -> dict:
    """Leakage-safe 5-family anomaly QA benchmark. Same non-circular protocol as
    ``build_anomaly_benchmark`` (train-only normalizer/calibrator/healthy-baseline,
    grouped splits, balanced) with the (a)+(b)+(c) mitigations."""
    rng = random.Random(seed)
    series = adapter.load()
    C = series[0].C
    pidx = pidx5(C)
    all_w = window_all(series, T=T, stride=stride,
                       max_windows_per_unit=max_windows_per_unit, seed=seed)  # (a) full life

    if shift == "condition":
        train_pool = [w for w in all_w if w.condition in set(train_conditions)]
        shift_pool = [w for w in all_w if w.condition in set(test_conditions)]
        units = sorted({w.unit_id for w in train_pool})
        assign = split_by_unit(units, fracs=(1 - indist_holdout_frac, 0.0, indist_holdout_frac), seed=seed)
        train_w = [w for w in train_pool if assign[w.unit_id] == "train"]
        indist_w = [w for w in train_pool if assign[w.unit_id] == "test"]
        train_series = [s for s in series if s.condition in set(train_conditions)
                        and assign.get(s.unit_id) == "train"]
    elif shift == "unit":
        units = sorted({w.unit_id for w in all_w})
        assign = split_by_unit(units, seed=seed)
        train_w = [w for w in all_w if assign[w.unit_id] == "train"]
        indist_w = [w for w in all_w if assign[w.unit_id] == "test"]
        shift_pool = []
        train_series = [s for s in series if assign.get(s.unit_id) == "train"]
    else:
        raise ValueError(f"unknown shift {shift!r}")
    if not train_w:
        raise SystemExit("no training windows; check conditions")

    normalizer = ChannelNormalizer().fit(train_w)
    train_norm = torch.stack([normalizer.transform(w.X) for w in train_w])
    cal = Calibrator.fit(train_norm, hi_q=hi_q, lo_q=lo_q, smooth_k=smooth_k)
    cal.a_level = a_level
    hw = healthy_windows(train_series, healthy_frac=healthy_frac, T=T, stride=stride)
    hw_norm = torch.stack([normalizer.transform(w) for w in hw])
    baseline = HealthyBaseline.fit(hw_norm, q=anomaly_q, a_anom=a_anom, smooth_k=smooth_k)

    def build(pool, regime, n):
        return _build_balanced_safe(pool, regime, depths, n, C, T, cal, baseline,
                                    normalizer, pidx, anomaly_p, allow_until,
                                    over_factor, rng, leak_max, max_leak_retries)

    train, train_leak = build(train_w, "train", n_train_per_depth)
    test_indist, indist_leak = build(indist_w, "indist", n_test_per_depth)
    test_shift, shift_leak = build(shift_pool, "shift", n_test_per_depth)
    meta = {"dataset": adapter.name, "n_channels": C, "T": T, "depths": list(depths),
            "shift": shift, "families": FIVE_FAMILIES, "anomaly_p": anomaly_p,
            "leak_max": leak_max, "safe": True,
            "n_train": len(train), "n_indist": len(test_indist), "n_shift": len(test_shift),
            "leakage": {"train": train_leak, "indist": indist_leak, "shift": shift_leak}}
    return {"train": train, "test_indist": test_indist, "test_shift": test_shift,
            "meta": meta, "normalizer": normalizer, "calibrator": cal, "baseline": baseline}
