"""Level-2 synthetic benchmark: planted CAUSAL rule + SPURIOUS shortcut channel.

This is the *necessity* experiment. Each instance carries

    (X, mu_star, question, phi_star, answer_star, rho_star, depth, regime)

The answer is determined ONLY by a depth-``d`` Signal Temporal Logic program
``phi_star`` over a set of CAUSAL channels, evaluated on the *planted* crisp
symbolic state ``mu_star``. A separate SPURIOUS channel carries a static,
high-SNR level that is correlated with the answer in the training distribution
but whose correlation is removed (or inverted) in the shifted test distribution.

Why this dataset exists
-----------------------
It is the one setting where we can *prove*, not assert, that the symbolic layer
is load-bearing:

* An end-to-end model is free to shortcut on the spurious channel (a depth-0
  level read) instead of learning the depth-``d`` temporal rule. Under
  distribution shift the shortcut breaks and its accuracy collapses.
* The neuro-symbolic path -- learned perception ``X -> mu_hat`` then the
  *deterministic* executor running ``phi_star`` -- cannot use the spurious
  channel, because ``phi_star`` never references it. Its accuracy is bounded
  only by perception error on the causal channels, which is the same in both
  regimes.

Ground-truth independence (no circularity)
------------------------------------------
``answer_star`` is computed from the *planted* crisp truths ``mu_star`` via the
exact generator profile -- NOT from the calibrated grounding and NOT from any
learned net. ``mu_star`` is the privileged supervision target (the analogue of
CLEVR's ground-truth object attributes). The evaluated model must *infer*
``mu_hat`` from the noisy signal ``X``; ``executor(phi_star, mu_hat)`` is
therefore a real prediction that can be wrong. ``executor(phi_star, mu_star)``
is only the upper bound (the "oracle"), reported as such.

Channel layout
--------------
Channels ``0 .. n_causal-1`` are causal; channel ``n_causal`` is the single
spurious channel (so ``C = n_causal + 1``). Programs reference causal channels
only. The symbolic state ``mu_star`` has the SAME column order as
``perception.grounding`` (family-major: high, low, rising, falling), so a
learned perception head can be supervised against it directly.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import torch

from executor.grammar import (
    Always,
    And,
    Eventually,
    Node,
    Or,
    Predicate,
    Until,
    validate,
)
from executor.hard_logic import evaluate
from perception.grounding import PRED_FAMILIES, predicate_index
from .templates import Question

LEVEL = ["high", "low"]
TREND = ["rising", "falling"]


@dataclass
class SpuriousInstance:
    X: torch.Tensor            # [T, C] signal (causal channels + 1 spurious)
    mu_star: torch.Tensor      # [T, 4C] PLANTED crisp predicate truths in {0,1}
    question: Question         # carries phi_star as .program
    phi_star: Node             # depth-d ground-truth program (causal channels only)
    answer_star: bool          # executor(phi_star, mu_star) > 0
    rho_star: float            # robustness on the planted (crisp) truths (= +/-1)
    depth: int                 # operator-nesting depth of phi_star
    regime: str                # "train" | "indist" | "shift"
    spurious_channel: int
    n_causal: int
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# depth-controlled program sampler (causal channels only)
# --------------------------------------------------------------------------- #

def _rand_interval(T: int, rng: random.Random, max_len: Optional[int] = None) -> tuple[int, int]:
    max_len = max_len or T
    a = rng.randint(0, max(0, T - 2))
    b = min(T - 1, a + rng.randint(1, max_len))
    return a, b


def sample_program(
    depth: int,
    causal_channels: list[int],
    T: int,
    rng: random.Random,
    allow_until: bool = False,
    force_temporal_root: bool = True,
) -> Node:
    """Sample a program of exactly the given operator-nesting depth.

    depth 0 -> a single Predicate over a causal channel.
    depth d -> one operator (temporal or boolean) over depth-(d-1) subtree(s).

    ``force_temporal_root`` wraps the top level in a temporal operator so the
    rule genuinely requires temporal reasoning (otherwise a shallow boolean
    combination of point predicates is too easy and weakens the contrast with
    the static spurious shortcut). ``allow_until`` is off by default because
    nested Until is O(T^2) per anchor and bloats generation cost.
    """
    if depth <= 0:
        c = rng.choice(causal_channels)
        fam = rng.choice(PRED_FAMILIES)
        return Predicate(fam, c)

    temporal_ops = ["eventually", "always"] + (["until"] if allow_until else [])
    boolean_ops = ["and", "or"]
    if force_temporal_root and depth >= 1 and rng.random() < 0.8:
        ops = temporal_ops
    else:
        ops = temporal_ops + boolean_ops
    op = rng.choice(ops)

    if op in ("eventually", "always"):
        a, b = _rand_interval(T, rng)
        child = sample_program(depth - 1, causal_channels, T, rng, allow_until, False)
        return (Eventually if op == "eventually" else Always)(a, b, child)
    if op == "until":
        a, b = _rand_interval(T, rng)
        left = sample_program(depth - 1, causal_channels, T, rng, allow_until, False)
        right = sample_program(depth - 1, causal_channels, T, rng, allow_until, False)
        return Until(a, b, left, right)
    # boolean: both children at depth-1 so overall depth is d
    left = sample_program(depth - 1, causal_channels, T, rng, allow_until, False)
    right = sample_program(depth - 1, causal_channels, T, rng, allow_until, False)
    return (And if op == "and" else Or)(left, right)


# --------------------------------------------------------------------------- #
# RUL-grounded programs: structurally identical to sample_program, but every
# temporal window is anchored to the NEAR-FAILURE region (per-step RUL <= k).
# RUL only sets the interval; the answer is still executor(phi, mu*) over sensor
# predicates, so the family stays non-circular and (for depth>=2) non-leaky, while
# the questions become PHM-prognostic ("as the unit approaches failure, ...").
# --------------------------------------------------------------------------- #

def near_failure_interval(rul_window, k: float) -> Optional[tuple[int, int]]:
    """Inclusive [a,b] of timesteps whose RUL <= k (the approach-to-failure region).

    ``rul_window`` is the per-step RUL over a window (monotonically decreasing), so
    these steps form the window's tail. Returns None if the window never gets within
    ``k`` of failure (an early-life window, ineligible for a RUL-grounded question).
    """
    idx = [t for t, r in enumerate(rul_window) if float(r) <= k]
    if not idx:
        return None
    return min(idx), max(idx)


def _rand_subinterval(a: int, b: int, rng: random.Random) -> tuple[int, int]:
    if b <= a:
        return a, b
    lo = rng.randint(a, b - 1)
    hi = rng.randint(lo + 1, b)
    return lo, hi


def _rul_sample(depth, causal_channels, a0, b0, rng, allow_until, force_temporal_root):
    if depth <= 0:
        return Predicate(rng.choice(PRED_FAMILIES), rng.choice(causal_channels))
    temporal_ops = ["eventually", "always"] + (["until"] if allow_until else [])
    boolean_ops = ["and", "or"]
    if force_temporal_root and rng.random() < 0.8:
        ops = temporal_ops
    else:
        ops = temporal_ops + boolean_ops
    op = rng.choice(ops)
    if op in ("eventually", "always"):
        lo, hi = _rand_subinterval(a0, b0, rng)
        child = _rul_sample(depth - 1, causal_channels, a0, b0, rng, allow_until, False)
        return (Eventually if op == "eventually" else Always)(lo, hi, child)
    if op == "until":
        lo, hi = _rand_subinterval(a0, b0, rng)
        left = _rul_sample(depth - 1, causal_channels, a0, b0, rng, allow_until, False)
        right = _rul_sample(depth - 1, causal_channels, a0, b0, rng, allow_until, False)
        return Until(lo, hi, left, right)
    left = _rul_sample(depth - 1, causal_channels, a0, b0, rng, allow_until, False)
    right = _rul_sample(depth - 1, causal_channels, a0, b0, rng, allow_until, False)
    return (And if op == "and" else Or)(left, right)


def sample_rul_program(
    depth: int,
    causal_channels: list[int],
    T: int,
    rng: random.Random,
    rul_window,
    k: float,
    allow_until: bool = False,
) -> Optional[Node]:
    """A depth-``d`` program whose temporal windows lie in the RUL<=k region.

    Returns None when the window has no near-failure region (so the caller skips
    it). The outermost operator is temporal (it must reason over the approach to
    failure), exactly like ``sample_program``'s ``force_temporal_root``.
    """
    nf = near_failure_interval(rul_window, k)
    if nf is None:
        return None
    a0, b0 = nf
    if b0 - a0 < 1:                      # need at least a 2-step window for a temporal op
        return None
    return _rul_sample(depth, causal_channels, a0, b0, rng, allow_until, True)


# --------------------------------------------------------------------------- #
# planting: crisp per-channel profiles -> (mu_star truths, latent level)
# --------------------------------------------------------------------------- #

def _channel_profile(
    T: int, rng: random.Random, hi: float = 2.0, lo: float = -2.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crisp truths (high, low, rising, falling) and a clean latent level.

    The channel is a sequence of 2-4 segments; each segment has a level class in
    {high, mid, low} and a trend class in {flat, rising, falling}. Truths are
    read off the CHOSEN classes (exact, noise-independent); the latent level is
    constructed to realise them. X is this latent plus noise (added by caller),
    so a learned perception must read X to recover the truths.
    """
    high = torch.zeros(T)
    low = torch.zeros(T)
    rising = torch.zeros(T)
    falling = torch.zeros(T)
    level = torch.zeros(T)

    n_seg = rng.randint(2, 4)
    cuts = sorted(rng.sample(range(1, T), n_seg - 1)) if n_seg > 1 else []
    bounds = [0] + cuts + [T]

    for s in range(n_seg):
        t0, t1 = bounds[s], bounds[s + 1]
        L = t1 - t0
        level_cls = rng.choice(["high", "mid", "low"])
        trend_cls = rng.choice(["flat", "rising", "falling"])
        base = {"high": hi, "mid": 0.0, "low": lo}[level_cls]
        slope = {"flat": 0.0, "rising": 0.6, "falling": -0.6}[trend_cls]
        local = torch.arange(L, dtype=torch.float32)
        level[t0:t1] = base + slope * (local - local.mean())
        if level_cls == "high":
            high[t0:t1] = 1.0
        elif level_cls == "low":
            low[t0:t1] = 1.0
        if trend_cls == "rising":
            rising[t0:t1] = 1.0
        elif trend_cls == "falling":
            falling[t0:t1] = 1.0
    return high, low, rising, falling, level


def _plant_causal(
    n_causal: int, T: int, rng: random.Random
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build planted crisp truths mu_causal:[T,4*n_causal] and latent:[T,n_causal]."""
    highs, lows, risings, fallings, levels = [], [], [], [], []
    for _ in range(n_causal):
        h, lo_, r, f, lv = _channel_profile(T, rng)
        highs.append(h); lows.append(lo_); risings.append(r); fallings.append(f)
        levels.append(lv)
    # family-major: [high(0..C-1), low(0..C-1), rising(...), falling(...)]
    mu = torch.cat(
        [torch.stack(highs, 1), torch.stack(lows, 1),
         torch.stack(risings, 1), torch.stack(fallings, 1)], dim=1
    )  # [T, 4*n_causal]
    latent = torch.stack(levels, dim=1)  # [T, n_causal]
    return mu, latent


def _spurious_truths(level_value: float, T: int, hi: float = 2.0, lo: float = -2.0):
    """Crisp truths for a constant-level spurious channel (trend flat)."""
    high = torch.full((T,), 1.0 if level_value >= hi else 0.0)
    low = torch.full((T,), 1.0 if level_value <= lo else 0.0)
    flat = torch.zeros(T)
    return high, low, flat.clone(), flat.clone()


# --------------------------------------------------------------------------- #
# benchmark assembly
# --------------------------------------------------------------------------- #

def _realize_causal(
    latent_causal: torch.Tensor, causal_noise: float, rng: random.Random
) -> torch.Tensor:
    """Add observation noise to the clean causal latent -> X_causal:[T,n_causal]."""
    T, n_causal = latent_causal.shape
    g = torch.Generator().manual_seed(rng.randint(0, 2**31 - 1))
    return latent_causal + causal_noise * torch.randn(T, n_causal, generator=g)


def _attach_spurious(
    x_causal: torch.Tensor,
    mu_causal: torch.Tensor,
    answer: bool,
    n_causal: int,
    T: int,
    rng: random.Random,
    spurious_gap: float,
    spurious_noise: float,
    regime: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X:[T,C], mu_star:[T,4C]) given a PRE-REALISED causal signal.

    Only the spurious channel is generated here, so two regimes can share the
    exact same causal content and differ ONLY in the spurious channel.

    regime:
      "train"/"indist" -> spurious level = +gap if answer else -gap (shortcut).
      "shift_decorr"   -> spurious level sign is INDEPENDENT of the answer.
      "shift_flip"     -> spurious level sign is INVERTED w.r.t. the answer.
    """
    C = n_causal + 1
    g = torch.Generator().manual_seed(rng.randint(0, 2**31 - 1))

    if regime in ("train", "indist"):
        sign = 1.0 if answer else -1.0
    elif regime == "shift_decorr":
        sign = rng.choice([1.0, -1.0])
    elif regime == "shift_flip":
        sign = -1.0 if answer else 1.0
    else:
        raise ValueError(f"unknown regime {regime!r}")
    sp_level = sign * spurious_gap

    X = torch.empty(T, C)
    X[:, :n_causal] = x_causal
    X[:, n_causal] = sp_level + spurious_noise * torch.randn(T, generator=g)

    # planted truths for spurious channel (do not affect the answer)
    sh, sl, sr, sf = _spurious_truths(sp_level, T)
    # rebuild mu_star in family-major order across ALL C channels
    H = torch.cat([mu_causal[:, 0 * n_causal:1 * n_causal], sh.unsqueeze(1)], 1)
    L = torch.cat([mu_causal[:, 1 * n_causal:2 * n_causal], sl.unsqueeze(1)], 1)
    R = torch.cat([mu_causal[:, 2 * n_causal:3 * n_causal], sr.unsqueeze(1)], 1)
    F = torch.cat([mu_causal[:, 3 * n_causal:4 * n_causal], sf.unsqueeze(1)], 1)
    mu_star = torch.cat([H, L, R, F], dim=1)  # [T, 4C]
    return X, mu_star


def _base_instances(
    n_per_depth: int,
    depths: list[int],
    n_causal: int,
    T: int,
    rng: random.Random,
    allow_until: bool,
    balance_tol: float,
    over_factor: int,
) -> list[dict]:
    """Generate balanced (phi_star, mu_causal, latent, answer, depth) tuples.

    Class balance is enforced per depth by rejection: we over-generate and keep
    a ~50/50 yes/no subset so neither the LSTM nor the symbolic path can win by
    predicting a constant.
    """
    causal_channels = list(range(n_causal))
    P = 4 * (n_causal + 1)
    pidx = predicate_index(n_causal + 1)
    out: list[dict] = []

    for d in depths:
        pool_yes, pool_no = [], []
        target = n_per_depth // 2
        attempts = 0
        while (len(pool_yes) < target or len(pool_no) < target) and attempts < n_per_depth * over_factor:
            attempts += 1
            phi = sample_program(d, causal_channels, T, rng, allow_until=allow_until)
            try:
                validate(phi, n_predicates=P)
            except Exception:
                continue
            mu_causal, latent = _plant_causal(n_causal, T, rng)
            # pad mu_causal to all-C width with zero spurious truths just to evaluate
            zeros = torch.zeros(T, 4)
            mu_eval = torch.cat([
                torch.cat([mu_causal[:, 0 * n_causal:1 * n_causal], zeros[:, 0:1]], 1),
                torch.cat([mu_causal[:, 1 * n_causal:2 * n_causal], zeros[:, 1:2]], 1),
                torch.cat([mu_causal[:, 2 * n_causal:3 * n_causal], zeros[:, 2:3]], 1),
                torch.cat([mu_causal[:, 3 * n_causal:4 * n_causal], zeros[:, 3:4]], 1),
            ], dim=1)
            rho, _ = evaluate(mu_eval, phi, pidx, anchor=0)
            # Reject VACUOUS instances: a non-finite robustness means an empty
            # (geometry-clamped) temporal window governs the verdict, so the
            # answer does not depend on the planted causal truths. Such an
            # instance is answerable without reading the signal -> not learnable
            # perception, and a leak the LSTM could exploit via the program
            # encoding alone. This mirrors the enforce_learnable guard in
            # benchmark/synthetic.py.
            if not math.isfinite(rho):
                continue
            ans = rho > 0.0
            rec = {"phi": phi, "mu_causal": mu_causal, "latent": latent,
                   "answer": bool(ans), "rho": float(rho), "depth": d}
            if ans and len(pool_yes) < target:
                pool_yes.append(rec)
            elif (not ans) and len(pool_no) < target:
                pool_no.append(rec)
        keep = pool_yes + pool_no
        rng.shuffle(keep)
        out.extend(keep)

    return out


def generate_spurious_benchmark(
    n_train: int = 2400,
    n_test: int = 800,
    n_causal: int = 3,
    T: int = 32,
    depths: Optional[list[int]] = None,
    spurious_gap: float = 3.0,
    causal_noise: float = 0.3,
    spurious_noise: float = 0.3,
    shift_mode: str = "shift_decorr",   # or "shift_flip"
    allow_until: bool = False,
    seed: int = 0,
) -> dict:
    """Build the three populations for the necessity experiment.

    Returns a dict with:
      "train"        : list[SpuriousInstance]   (spurious correlated)
      "test_indist"  : list[SpuriousInstance]   (spurious correlated, held-out rules)
      "test_shift"   : list[SpuriousInstance]   (SAME causal content as indist,
                                                 spurious correlation broken)
      "meta"         : dict of generation settings

    test_indist and test_shift share identical causal signals; they differ ONLY
    in the spurious channel, so any accuracy gap between them is attributable to
    spurious reliance.
    """
    depths = depths or [1, 2, 3, 4]
    rng = random.Random(seed)

    n_per_depth_tr = max(2, n_train // len(depths))
    n_per_depth_te = max(2, n_test // len(depths))

    base_tr = _base_instances(n_per_depth_tr, depths, n_causal, T, rng,
                              allow_until, balance_tol=0.05, over_factor=40)
    base_te = _base_instances(n_per_depth_te, depths, n_causal, T, rng,
                              allow_until, balance_tol=0.05, over_factor=40)

    def make(rec: dict, regime: str, x_causal: torch.Tensor) -> SpuriousInstance:
        X, mu_star = _attach_spurious(
            x_causal, rec["mu_causal"], rec["answer"], n_causal, T, rng,
            spurious_gap, spurious_noise,
            regime if regime != "shift" else shift_mode,
        )
        q = Question(template=f"spurious_d{rec['depth']}", program=rec["phi"],
                     bindings={"depth": rec["depth"]}, text=None)
        return SpuriousInstance(
            X=X, mu_star=mu_star, question=q, phi_star=rec["phi"],
            answer_star=rec["answer"], rho_star=rec["rho"], depth=rec["depth"],
            regime=regime, spurious_channel=n_causal, n_causal=n_causal,
        )

    # train: fresh causal realisation per record
    train = [make(r, "train", _realize_causal(r["latent"], causal_noise, rng))
             for r in base_tr]
    # test: realise causal ONCE per record, reuse for both indist and shift so
    # the only difference between the two splits is the spurious channel.
    test_indist, test_shift = [], []
    for r in base_te:
        x_causal = _realize_causal(r["latent"], causal_noise, rng)
        test_indist.append(make(r, "indist", x_causal))
        test_shift.append(make(r, "shift", x_causal))

    meta = {
        "n_train": len(train), "n_test": len(test_indist),
        "n_causal": n_causal, "n_channels": n_causal + 1, "T": T,
        "depths": depths, "spurious_gap": spurious_gap,
        "causal_noise": causal_noise, "spurious_noise": spurious_noise,
        "shift_mode": shift_mode, "allow_until": allow_until, "seed": seed,
        "predicate_dim": 4 * (n_causal + 1),
    }
    return {"train": train, "test_indist": test_indist,
            "test_shift": test_shift, "meta": meta}


def balance_report(instances: list[SpuriousInstance]) -> dict:
    """Yes/no balance overall and per depth, plus spurious-shortcut accuracy.

    ``spurious_acc`` is the accuracy of the trivial classifier 'predict yes iff
    the spurious channel level is high' -- it should be ~1.0 in train/indist and
    ~0.5 (decorr) or ~0.0 (flip) in shift, which is the whole point.
    """
    n = len(instances)
    yes = sum(i.answer_star for i in instances)
    per_depth: dict[int, dict] = {}
    sp_correct = 0
    for i in instances:
        sp_level = float(i.X[:, i.spurious_channel].mean())
        sp_pred = sp_level >= 0.0
        sp_correct += int(sp_pred == i.answer_star)
        d = i.depth
        per_depth.setdefault(d, {"n": 0, "yes": 0})
        per_depth[d]["n"] += 1
        per_depth[d]["yes"] += int(i.answer_star)
    return {
        "n": n, "yes": yes, "no": n - yes,
        "yes_frac": (yes / n if n else 0.0),
        "spurious_shortcut_acc": (sp_correct / n if n else 0.0),
        "per_depth": {d: {"n": v["n"], "yes_frac": v["yes"] / max(1, v["n"])}
                      for d, v in sorted(per_depth.items())},
    }