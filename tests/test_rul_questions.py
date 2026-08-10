"""RUL-grounded question family: anchoring, non-circularity, balance."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.adapters import DatasetAdapter, Series
from benchmark.realdata import build_real_benchmark, real_balance_report
from benchmark.spurious import (near_failure_interval, sample_program,
                                sample_rul_program)
from executor.grammar import Always, And, Eventually, Not, Or, Predicate, Until
from executor.hard_logic import evaluate
from perception.grounding import predicate_index
import random

T, C, RUL_K = 16, 3, 16


def _temporal_intervals(node):
    """All (a,b) windows of temporal operators in a program tree."""
    out = []
    if isinstance(node, (Eventually, Always)):
        out.append((node.a, node.b))
        out += _temporal_intervals(node.child)
    elif isinstance(node, Until):
        out.append((node.a, node.b))
        out += _temporal_intervals(node.left) + _temporal_intervals(node.right)
    elif isinstance(node, Not):
        out += _temporal_intervals(node.child)
    elif isinstance(node, (And, Or)):
        out += _temporal_intervals(node.left) + _temporal_intervals(node.right)
    return out


def test_near_failure_interval():
    rul = list(range(20, 0, -1))                 # 20,19,...,1 over T=20
    assert near_failure_interval(rul, 5) == (15, 19)   # last 5 steps have RUL<=5
    assert near_failure_interval(rul, 100) == (0, 19)
    assert near_failure_interval([50, 40, 30], 5) is None


def test_rul_program_windows_inside_near_failure():
    rng = random.Random(0)
    rul_window = list(range(T, 0, -1))           # T..1
    a0, b0 = near_failure_interval(rul_window, RUL_K)
    for _ in range(200):
        for d in (1, 2, 3):
            phi = sample_rul_program(d, list(range(C)), T, rng, rul_window, RUL_K)
            if phi is None:
                continue
            for (a, b) in _temporal_intervals(phi):
                assert a0 <= a <= b <= b0, f"interval [{a},{b}] outside near-failure [{a0},{b0}]"


def test_rul_program_none_for_early_window():
    rng = random.Random(0)
    rul_window = list(range(100, 100 - T, -1))   # all RUL >> k -> early life
    assert sample_rul_program(2, list(range(C)), T, rng, rul_window, RUL_K) is None


# --- end-to-end build over a tiny in-memory adapter (no vibration/FFT) --------- #

class _SynthAdapter(DatasetAdapter):
    name = "synth"

    def load(self):
        g = torch.Generator().manual_seed(0)
        series = []
        for cond in ("A", "B"):
            for u in range(4):
                L = 64
                vals = torch.cumsum(0.3 * torch.randn(L, C, generator=g), dim=0)
                series.append(Series(
                    values=vals, unit_id=f"{cond}:u{u}", dataset="synth",
                    channel_names=[f"s{c}" for c in range(C)], condition=cond,
                    meta={"rul": torch.arange(L, 0, -1, dtype=torch.float32)}))
        return series


def _build_rul():
    return build_real_benchmark(
        _SynthAdapter(), T=T, stride=2, depths=(1, 2),
        shift="condition", train_conditions=("A",), test_conditions=("B",),
        indist_holdout_frac=0.34, n_train_per_depth=24, n_test_per_depth=24,
        over_factor=400, seed=0, question_family="rul", rul_k=RUL_K,
    )


def test_rul_benchmark_balanced_and_non_circular():
    bm = _build_rul()
    assert bm["meta"]["question_family"] == "rul"
    C_ = bm["meta"]["n_channels"]
    pidx = predicate_index(C_)
    for split in ("train", "test_indist", "test_shift"):
        insts = bm[split]
        assert insts, f"{split} empty"
        rep = real_balance_report(insts)
        assert 0.40 <= rep["yes_frac"] <= 0.60        # balanced (loose for small n)
        for i in insts:
            rho, _ = evaluate(i.mu_star, i.phi_star, pidx, 0)
            assert bool(rho > 0) == bool(i.answer_star)   # non-circular label
            assert i.provenance["family"] == "rul"
