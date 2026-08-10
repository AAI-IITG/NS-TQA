"""Answer-level benchmark over the learned `anomalous` predicate (Phase G payoff).

Builds non-circular QA instances whose programs reference a FIVE-family predicate
vocabulary ``[high, low, rising, falling, anomalous]``. The first four come from
the calibrated grounding (a fixed threshold a hand-rule can reproduce); the fifth,
``anomalous``, is the privileged healthy-baseline-deviation target from
``perception.anomaly`` that a fixed threshold *cannot* reproduce. Labels are
``executor(phi, mu_star_5)`` as usual, so the benchmark is non-circular.

The point: on anomaly-using programs, faithful NS-TQA (learned perception + learned
anomaly head) should clearly beat the STL-only hand-rule baseline (which has no
faithful ``anomalous`` rule and must fall back to a level proxy) -- the setting
where learned perception is necessary on real signals.

No executor/grammar change is needed: ``validate`` only checks channel indices and
``evaluate`` resolves predicates through a supplied index map, so a 5-family
``pidx5`` + a ``[T, 5C]`` ``mu_star`` is sufficient. The global 4-family
``PRED_FAMILIES`` is left untouched.
"""
from __future__ import annotations

import math
import random
from typing import Optional

import torch

from executor.grammar import (Always, And, Eventually, Not, Or, Predicate, Until,
                              validate)
from executor.hard_logic import evaluate
from perception.anomaly import HealthyBaseline, healthy_windows
from perception.grounding import Calibrator, ground

from .adapters import (ChannelNormalizer, RealInstance, Series, split_by_unit,
                       window_all)
from .spurious import _rand_interval
from .templates import Question

FIVE_FAMILIES = ["high", "low", "rising", "falling", "anomalous"]
_LEVEL_FAMILIES = ["high", "low", "rising", "falling"]


def pidx5(C: int) -> dict:
    """Predicate index for the 5-family vocabulary, family-major (col = fam*C + c)."""
    return {(f, c): i * C + c for i, f in enumerate(FIVE_FAMILIES) for c in range(C)}


def mu_star_5(x: torch.Tensor, cal: Calibrator, baseline: HealthyBaseline) -> torch.Tensor:
    """Privileged 5-family truths mu*:[T,5C] for normalized window x:[T,C]."""
    mu4, _ = ground(x, cal)                       # [T,4C] high,low,rising,falling
    anom = baseline.target(x)                     # [T,C]
    return torch.cat([mu4, anom], dim=1)          # [T,5C]


def sample_anomaly_program(depth, channels, T, rng, anomaly_p: float = 0.5,
                           allow_until: bool = False, force_temporal_root: bool = True):
    """Sample a depth-d program over the 5-family vocab; leaves are ``anomalous``
    with prob ``anomaly_p`` (else a level/trend family) so the benchmark exercises
    the learned predicate."""
    if depth <= 0:
        c = rng.choice(channels)
        fam = "anomalous" if rng.random() < anomaly_p else rng.choice(_LEVEL_FAMILIES)
        return Predicate(fam, c)
    temporal = ["eventually", "always"] + (["until"] if allow_until else [])
    ops = temporal if (force_temporal_root and rng.random() < 0.8) else temporal + ["and", "or"]
    op = rng.choice(ops)
    sub = lambda: sample_anomaly_program(depth - 1, channels, T, rng, anomaly_p, allow_until, False)
    if op in ("eventually", "always"):
        a, b = _rand_interval(T, rng)
        return (Eventually if op == "eventually" else Always)(a, b, sub())
    if op == "until":
        a, b = _rand_interval(T, rng)
        return Until(a, b, sub(), sub())
    return (And if op == "and" else Or)(sub(), sub())


def _build_balanced_5(windows, regime, depths, n_per_depth, C, T, cal, baseline,
                      normalizer, pidx, anomaly_p, allow_until, over_factor, rng):
    if not windows:
        return []
    channels = list(range(C))
    cache: dict = {}
    out: list[RealInstance] = []

    def ground_window(w):
        key = (w.unit_id, w.start)
        if key not in cache:
            Xn = normalizer.transform(w.X)
            cache[key] = (Xn, mu_star_5(Xn, cal, baseline))
        return cache[key]

    for d in depths:
        target = max(1, n_per_depth // 2)
        yes, no = [], []
        attempts, cap = 0, max(1, n_per_depth) * over_factor
        while (len(yes) < target or len(no) < target) and attempts < cap:
            attempts += 1
            w = rng.choice(windows)
            phi = sample_anomaly_program(d, channels, T, rng, anomaly_p, allow_until)
            try:
                validate(phi, n_predicates=C)
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
                X=Xn, question=Question(template=f"anom_d{d}", program=phi,
                                        bindings={"depth": d}, text=None),
                phi_star=phi, answer_star=ans, depth=d, unit_id=w.unit_id,
                n_channels=C, condition=w.condition, regime=regime,
                mu_star=mu, rho_star=float(rho), spurious_channel=None,
                provenance={"dataset": w.dataset, "start": w.start, "five_family": True,
                            "critical_t": tr.critical_t})
            (yes if ans else no).append(inst)
        keep = yes[:target] + no[:target]
        rng.shuffle(keep)
        out.extend(keep)
    return out


def build_anomaly_benchmark(
    adapter, *, T=32, stride=2, depths=(1, 2, 3, 4),
    shift="condition", train_conditions=("35Hz12kN",), test_conditions=("37.5Hz11kN", "40Hz10kN"),
    indist_holdout_frac=0.3, healthy_frac=0.25,
    n_train_per_depth=200, n_test_per_depth=120,
    hi_q=0.85, lo_q=0.15, smooth_k=5, a_level=4.0, a_anom=4.0, anomaly_q=0.99,
    anomaly_p=0.5, allow_until=False, max_windows_per_unit: Optional[int] = 200,
    over_factor=120, seed=0,
) -> dict:
    """Non-circular 5-family (anomaly) QA benchmark; mirrors ``build_real_benchmark``
    but with the ``anomalous`` family and a healthy-baseline target."""
    rng = random.Random(seed)
    series = adapter.load()
    C = series[0].C
    pidx = pidx5(C)
    all_w = window_all(series, T=T, stride=stride, max_windows_per_unit=max_windows_per_unit, seed=seed)

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
        return _build_balanced_5(pool, regime, depths, n, C, T, cal, baseline,
                                 normalizer, pidx, anomaly_p, allow_until, over_factor, rng)

    train = build(train_w, "train", n_train_per_depth)
    test_indist = build(indist_w, "indist", n_test_per_depth)
    test_shift = build(shift_pool, "shift", n_test_per_depth)
    meta = {"dataset": adapter.name, "n_channels": C, "T": T, "depths": list(depths),
            "shift": shift, "families": FIVE_FAMILIES, "anomaly_p": anomaly_p,
            "n_train": len(train), "n_indist": len(test_indist), "n_shift": len(test_shift)}
    return {"train": train, "test_indist": test_indist, "test_shift": test_shift,
            "meta": meta, "normalizer": normalizer, "calibrator": cal, "baseline": baseline}


# --------------------------------------------------------------------------- #
# evaluation: NS-TQA (learned perception + learned anomaly head) vs STL-only vs oracle
# --------------------------------------------------------------------------- #

def _acc(pred, gold, depth) -> dict:
    from utils.oracle_metrics import group_accuracy
    acc = sum(p == g for p, g in zip(pred, gold)) / max(1, len(gold))
    by_d = {int(k): v["accuracy"] for k, v in group_accuracy(depth, pred, gold).items()}
    return {"n": len(gold), "answer_accuracy": acc, "by_depth": by_d}


def _eval(instances, mu_of) -> dict:
    """Generic: ``mu_of(inst) -> mu5[T,5C]``, execute phi_star, score vs answer_star."""
    if not instances:
        return {"n": 0, "answer_accuracy": None, "by_depth": {}}
    C = instances[0].n_channels
    pidx = pidx5(C)
    pred, gold, depth = [], [], []
    for i in instances:
        rho, _ = evaluate(mu_of(i), i.phi_star, pidx, anchor=0)
        pred.append(bool(rho > 0.0)); gold.append(bool(i.answer_star)); depth.append(i.depth)
    return _acc(pred, gold, depth)


@torch.no_grad()
def nstqa_anomaly_eval(instances, perception_model, anomaly_head) -> dict:
    """Faithful path: learned 4-family perception + learned anomaly head -> executor."""
    from perception.anomaly import predict_anomaly
    from perception.learned import predict_mu
    return _eval(instances, lambda i: torch.cat(
        [predict_mu(perception_model, i.X), predict_anomaly(anomaly_head, i.X)], dim=1))


def stl_only_anomaly_eval(instances, hi: float = 1.0, lo: float = -1.0,
                          smooth_k: int = 5, a_level: float = 4.0) -> dict:
    """STL-only: hand-rule level/trend + a hand-rule PROXY for anomalous (max(high,low))."""
    from models.stl_only import hand_rule_calibrator
    if not instances:
        return {"n": 0, "answer_accuracy": None, "by_depth": {}}
    C = instances[0].n_channels
    cal = hand_rule_calibrator(C, hi=hi, lo=lo, slope_scale=1.0, a_level=a_level, smooth_k=smooth_k)

    def mu_of(i):
        mu4, _ = ground(i.X, cal)
        proxy = torch.maximum(mu4[:, :C], mu4[:, C:2 * C])   # "anomalous" ~ high OR low
        return torch.cat([mu4, proxy], dim=1)

    return _eval(instances, mu_of)


def oracle_anomaly_eval(instances) -> dict:
    """Executor on the privileged mu_star_5 (upper bound)."""
    return _eval(instances, lambda i: i.mu_star)


def anomaly_leakage_probe(instances) -> dict:
    """Best single privileged predicate column's accuracy at predicting the answer,
    per depth (want ~chance for depth >= 2 -> executor non-decorative)."""
    if not instances:
        return {}
    C = instances[0].n_channels
    P = 5 * C
    per_depth: dict = {}
    for i in instances:
        per_depth.setdefault(i.depth, []).append(i)
    out = {}
    for d, insts in sorted(per_depth.items()):
        feats = torch.stack([inst.mu_star.mean(0) for inst in insts])   # [n, P]
        y = torch.tensor([float(inst.answer_star) for inst in insts])
        best = 0.0
        for col in range(P):
            f = feats[:, col]
            for sign in (1.0, -1.0):
                acc = ((sign * f > sign * f.median()) == (y > 0.5)).float().mean().item()
                best = max(best, acc, 1 - acc)
        out[int(d)] = best
    return out
