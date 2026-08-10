"""Non-circular QA construction over real signals.

Turns a ``DatasetAdapter``'s ``Series`` into balanced ``RealInstance``s with the
honest protocol:

  1. window the series and split by UNIT (held-out engines/bearings) and/or by
     CONDITION (e.g. C-MAPSS FD001 -> FD002/FD004);
  2. fit the model-input normalizer AND the privileged grounding calibrator on
     TRAIN windows ONLY;
  3. the answer label is computed by executing a sampled depth-d program on the
     privileged grounding ``mu_star`` -- never by the model;
  4. rejection-balance yes/no per depth (fixing the ~0.19 imbalance of the old
     circular build);
  5. emit instances where the model sees the normalized window and ``mu_star``
     is the privileged supervision target / oracle.

The output dict has the SAME keys the necessity engine expects
(``train`` / ``test_indist`` / ``test_shift``), so it drops straight into
``benchmark.necessity.run_single`` and ``perception.learned.train_perception``.

Non-circularity: the model never receives the calibrator/thresholds; it must
infer ``mu_hat`` from the normalized signal and is scored on held-out engines
(``test_indist``) and a shifted operating regime (``test_shift``). The executor
on ``mu_star`` is reported only as the oracle upper bound.
"""
from __future__ import annotations

import math
import random
from typing import Optional

import torch

from executor.grammar import validate
from executor.hard_logic import evaluate
from perception.grounding import Calibrator, ground, predicate_index

from .adapters import (ChannelNormalizer, RealInstance, Series, Window,
                       split_by_unit, window_all)
from .spurious import sample_program, sample_rul_program
from .templates import Question


def _build_balanced(
    windows: list[Window], regime: str, depths, n_per_depth: int, C: int, T: int,
    cal, normalizer: ChannelNormalizer, pidx, allow_until: bool,
    over_factor: int, rng: random.Random,
    family: str = "generic", rul_by_unit: Optional[dict] = None, rul_k: Optional[float] = None,
    label_cal=None, label_normalizer: Optional[ChannelNormalizer] = None,
) -> list[RealInstance]:
    """Rejection-sample (window, program) pairs into a yes/no-balanced set per depth.

    A real window can supply several QA pairs (one per sampled program); the
    privileged grounding decides the answer. Vacuous (non-finite robustness)
    pairs are dropped, exactly as in the synthetic generator.

    The **model input** ``inst.X`` is always ``normalizer.transform(w.X)`` (the
    train-side view the model is scored on). The **privileged label** ``mu_star``
    (hence ``answer_star``) is computed from ``label_normalizer``/``label_cal`` when
    given; they default to ``normalizer``/``cal`` (the L-src regime). Passing a
    distinct test-rig-fit grounding here realizes the L-tgt regime — see
    ``docs/labeling_regimes.md``.
    """
    if not windows:
        return []
    lab_cal = cal if label_cal is None else label_cal
    lab_norm = normalizer if label_normalizer is None else label_normalizer
    causal = list(range(C))
    cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}  # (unit,start) -> (Xn, mu_star)
    out: list[RealInstance] = []

    def ground_window(w: Window):
        key = (w.unit_id, w.start)
        if key not in cache:
            Xn = normalizer.transform(w.X)               # model input (train-side)
            mu_star, _ = ground(lab_norm.transform(w.X), lab_cal)  # privileged label
            cache[key] = (Xn, mu_star)
        return cache[key]

    for d in depths:
        target = max(1, n_per_depth // 2)
        yes, no = [], []
        attempts, cap = 0, max(1, n_per_depth) * over_factor
        while (len(yes) < target or len(no) < target) and attempts < cap:
            attempts += 1
            w = rng.choice(windows)
            if family == "rul":
                rw = rul_by_unit.get(w.unit_id) if rul_by_unit else None
                if rw is None:
                    continue
                rul_window = rw[w.start : w.start + T]
                phi = sample_rul_program(d, causal, T, rng, rul_window, rul_k,
                                         allow_until=allow_until)
                if phi is None:                       # window not near failure -> skip
                    continue
            else:
                phi = sample_program(d, causal, T, rng, allow_until=allow_until)
            try:
                validate(phi, n_predicates=4 * C)
            except Exception:
                continue
            Xn, mu_star = ground_window(w)
            rho, tr = evaluate(mu_star, phi, pidx, anchor=0)
            if not math.isfinite(rho):
                continue
            ans = bool(rho > 0.0)
            if ans and len(yes) >= target:
                continue
            if (not ans) and len(no) >= target:
                continue
            inst = RealInstance(
                X=Xn, question=Question(template=f"real_d{d}", program=phi,
                                        bindings={"depth": d}, text=None),
                phi_star=phi, answer_star=ans, depth=d, unit_id=w.unit_id,
                n_channels=C, condition=w.condition, regime=regime,
                mu_star=mu_star, rho_star=float(rho), spurious_channel=None,
                provenance={"dataset": w.dataset, "start": w.start,
                            "channel_names": w.channel_names,
                            "critical_t": tr.critical_t, "family": family},
            )
            (yes if ans else no).append(inst)
        keep = yes[:target] + no[:target]
        rng.shuffle(keep)
        out.extend(keep)
    return out


def real_balance_report(instances: list[RealInstance]) -> dict:
    n = len(instances)
    yes = sum(i.answer_star for i in instances)
    per_depth: dict[int, dict] = {}
    for i in instances:
        per_depth.setdefault(i.depth, {"n": 0, "yes": 0})
        per_depth[i.depth]["n"] += 1
        per_depth[i.depth]["yes"] += int(i.answer_star)
    return {
        "n": n, "yes": yes, "no": n - yes,
        "yes_frac": (yes / n if n else 0.0),
        "per_depth": {d: {"n": v["n"], "yes_frac": v["yes"] / max(1, v["n"])}
                      for d, v in sorted(per_depth.items())},
    }


def build_real_benchmark(
    adapter,
    *,
    T: int = 48,
    stride: int = 4,
    depths=(1, 2, 3, 4),
    shift: str = "condition",              # "condition" | "unit"
    train_conditions=("FD001",),
    test_conditions=("FD002", "FD004"),
    indist_holdout_frac: float = 0.2,
    n_train_per_depth: int = 300,
    n_test_per_depth: int = 120,
    hi_q: float = 0.85, lo_q: float = 0.15, smooth_k: int = 5, a_level: float = 4.0,
    allow_until: bool = False,
    max_windows_per_unit: Optional[int] = None,
    over_factor: int = 60,
    seed: int = 0,
    question_family: str = "generic",      # "generic" | "rul"
    rul_k: Optional[float] = None,         # required for question_family="rul"
    label_regime: str = "L-src",           # "L-src" (train-side labels) | "L-tgt" (test-rig-fit labels)
    label_fit_frac: float = 0.5,           # L-tgt: fraction of each test condition's units used to fit its grounding
) -> dict:
    """Build a non-circular QA benchmark from a dataset adapter.

    ``shift="condition"``: train + held-out-engine in-dist test drawn from
    ``train_conditions``; shifted test drawn from ``test_conditions``.
    ``shift="unit"``: train + held-out-unit test from all data; no shifted test.

    ``question_family="rul"`` anchors every program's temporal windows to the
    near-failure region (per-step RUL <= ``rul_k``, from ``Series.meta['rul']``),
    yielding PHM-prognostic questions; the answer is still ``executor(phi, mu*)``
    over sensor predicates (non-circular). ``over_factor`` may need raising since
    early-life windows are rejected.

    ``label_regime`` (see ``docs/labeling_regimes.md``):
      * ``"L-src"`` (default, the conference-paper regime): the train-side
        normalizer+calibrator define ``mu_star`` (hence the answer) on every pool,
        including the shifted test.
      * ``"L-tgt"``: the shifted test's labels are defined by a grounding fit on the
        **test condition's own** ``label_fit_frac`` units (never the evaluated
        windows); the model input stays train-normalized, so only the label-defining
        grounding moves. Isolates learned-perception necessity (C3b) from executor +
        vocabulary transfer (C4). No effect when there is no shifted pool.
    """
    if question_family == "rul" and rul_k is None:
        raise ValueError("question_family='rul' requires rul_k")
    if label_regime not in ("L-src", "L-tgt"):
        raise ValueError(f"unknown label_regime {label_regime!r}")
    rng = random.Random(seed)
    series = adapter.load()
    if not series:
        raise SystemExit("adapter returned no series")
    C = series[0].C
    pidx = predicate_index(C)
    rul_by_unit = {s.unit_id: s.meta.get("rul") for s in series}
    if question_family == "rul" and all(v is None for v in rul_by_unit.values()):
        raise ValueError("question_family='rul' but no Series carries meta['rul']")
    all_windows = window_all(series, T=T, stride=stride,
                             max_windows_per_unit=max_windows_per_unit, seed=seed)

    # ---- partition windows into train / in-dist test / shifted test ----
    if shift == "condition":
        train_pool = [w for w in all_windows if w.condition in set(train_conditions)]
        shift_pool = [w for w in all_windows if w.condition in set(test_conditions)]
        units = sorted({w.unit_id for w in train_pool})
        assign = split_by_unit(units, fracs=(1 - indist_holdout_frac, 0.0,
                                             indist_holdout_frac), seed=seed)
        train_windows = [w for w in train_pool if assign[w.unit_id] == "train"]
        indist_windows = [w for w in train_pool if assign[w.unit_id] == "test"]
    elif shift == "unit":
        units = sorted({w.unit_id for w in all_windows})
        assign = split_by_unit(units, seed=seed)
        train_windows = [w for w in all_windows if assign[w.unit_id] == "train"]
        indist_windows = [w for w in all_windows if assign[w.unit_id] == "test"]
        shift_pool = []
    else:
        raise ValueError(f"unknown shift mode {shift!r}")

    if not train_windows:
        raise SystemExit("no training windows after split; check conditions/units")

    # ---- fit model normalizer + privileged calibrator on TRAIN windows ONLY ----
    normalizer = ChannelNormalizer().fit(train_windows)
    train_norm = torch.stack([normalizer.transform(w.X) for w in train_windows])  # [N,T,C]
    cal = Calibrator.fit(train_norm, hi_q=hi_q, lo_q=lo_q, smooth_k=smooth_k)
    cal.a_level = a_level

    # ---- build balanced instances ----
    def build(pool, regime, n, label_cal=None, label_normalizer=None):
        return _build_balanced(pool, regime, depths, n, C, T, cal, normalizer,
                               pidx, allow_until, over_factor, rng,
                               family=question_family, rul_by_unit=rul_by_unit, rul_k=rul_k,
                               label_cal=label_cal, label_normalizer=label_normalizer)

    train = build(train_windows, "train", n_train_per_depth)
    test_indist = build(indist_windows, "indist", n_test_per_depth)

    # ---- shifted test: label regime decides whose grounding defines the labels ----
    if not shift_pool:
        test_shift = []
        label_regime_used = label_regime  # recorded even when inert (no shift pool)
    elif label_regime == "L-src":
        # train-side grounding labels the shifted windows (conference-paper regime)
        test_shift = build(shift_pool, "shift", n_test_per_depth)
        label_regime_used = "L-src"
    else:
        # L-tgt: per test condition, fit a grounding on that condition's own
        # label_fit_frac units (disjoint from the evaluated units) and use it to
        # define the labels; the model input stays train-normalized.
        test_shift = []
        for cond in sorted({w.condition for w in shift_pool}):
            cond_windows = [w for w in shift_pool if w.condition == cond]
            cond_units = sorted({w.unit_id for w in cond_windows})
            if len(cond_units) < 2:
                # cannot hold out a disjoint label-fitting unit -> skip honestly
                continue
            cassign = split_by_unit(cond_units,
                                    fracs=(label_fit_frac, 0.0, 1.0 - label_fit_frac),
                                    seed=seed)
            fit_windows = [w for w in cond_windows if cassign[w.unit_id] == "train"]
            eval_windows = [w for w in cond_windows if cassign[w.unit_id] == "test"]
            if not fit_windows or not eval_windows:
                continue
            norm_tgt = ChannelNormalizer().fit(fit_windows)
            fit_norm = torch.stack([norm_tgt.transform(w.X) for w in fit_windows])
            cal_tgt = Calibrator.fit(fit_norm, hi_q=hi_q, lo_q=lo_q, smooth_k=smooth_k)
            cal_tgt.a_level = a_level
            test_shift.extend(build(eval_windows, "shift", n_test_per_depth,
                                    label_cal=cal_tgt, label_normalizer=norm_tgt))
        label_regime_used = "L-tgt"

    meta = {
        "dataset": adapter.name, "n_channels": C, "T": T, "depths": list(depths),
        "shift": shift, "train_conditions": list(train_conditions),
        "test_conditions": list(test_conditions) if shift == "condition" else [],
        "channel_names": series[0].channel_names, "seed": seed,
        "question_family": question_family, "rul_k": rul_k,
        "label_regime": label_regime_used, "label_fit_frac": label_fit_frac,
        "n_train": len(train), "n_indist": len(test_indist), "n_shift": len(test_shift),
        "balance": {"train": real_balance_report(train),
                    "test_indist": real_balance_report(test_indist),
                    "test_shift": real_balance_report(test_shift)},
    }
    return {"train": train, "test_indist": test_indist, "test_shift": test_shift,
            "meta": meta, "calibrator": cal, "normalizer": normalizer}