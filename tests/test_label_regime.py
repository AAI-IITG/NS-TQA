"""Invariant tests for the label_regime switch in build_real_benchmark (WO-1A).

Self-contained: a tiny synthetic 2-condition adapter (no raw data). Checks that
BOTH regimes preserve the core invariants (balance, non-circularity / oracle 1.000)
and that L-tgt actually redefines the shifted-test labels via a test-condition-fit
grounding while keeping the model input on the train normalizer.
See docs/labeling_regimes.md.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.adapters import DatasetAdapter, Series
from benchmark.necessity import oracle_accuracy
from benchmark.realdata import build_real_benchmark, real_balance_report
from perception.grounding import predicate_index

BUILD = dict(
    T=32, stride=4, depths=(1, 2, 3, 4), shift="condition",
    train_conditions=("A",), test_conditions=("B",), indist_holdout_frac=0.34,
    n_train_per_depth=24, n_test_per_depth=16, over_factor=40,
    max_windows_per_unit=30, seed=1, label_fit_frac=0.5,
)


class _StubAdapter(DatasetAdapter):
    """Two operating conditions; condition B is scaled+offset so a train-A grounding
    labels it differently than a B-fit grounding would."""
    name = "stub"

    def load(self):
        g = torch.Generator().manual_seed(0)
        names = [f"ch{i}" for i in range(4)]
        out = []
        for cond, scale, off, n in [("A", 1.0, 0.0, 6), ("B", 1.4, 0.4, 6)]:
            for u in range(n):
                t = torch.linspace(0, 1, 160).unsqueeze(1)
                base = torch.sin(2 * torch.pi * (t + 0.1 * u)) + 0.5 * t
                vals = off + scale * (base.repeat(1, 4) + 0.1 * torch.randn(160, 4, generator=g))
                out.append(Series(values=vals, unit_id=f"{cond}:u{u}", dataset="stub",
                                  channel_names=names, condition=cond,
                                  meta={"rul": torch.arange(160, 0, -1).float()}))
        return out


def _build(regime):
    return build_real_benchmark(_StubAdapter(), label_regime=regime, **BUILD)


def _invariants(bm, regime):
    C = bm["meta"]["n_channels"]
    pidx = predicate_index(C)
    assert bm["meta"]["label_regime"] == regime
    for split in ("train", "test_indist", "test_shift"):
        insts = bm[split]
        assert insts, f"{regime}/{split} empty"
        rep = real_balance_report(insts)
        assert 0.45 <= rep["yes_frac"] <= 0.55, (regime, split, rep["yes_frac"])
        # non-circularity: executor on stored mu_star reproduces answer_star exactly
        assert oracle_accuracy(insts, pidx)["answer_accuracy"] == 1.0, (regime, split)


def test_lsrc_invariants():
    _invariants(_build("L-src"), "L-src")


def test_ltgt_invariants():
    _invariants(_build("L-tgt"), "L-tgt")


def test_lsrc_is_default():
    bm = build_real_benchmark(_StubAdapter(), **BUILD)  # no label_regime -> L-src
    assert bm["meta"]["label_regime"] == "L-src"


def test_ltgt_relabels_shift_pool():
    """L-tgt must change the shifted-test labels vs L-src (different grounding)."""
    src, tgt = _build("L-src"), _build("L-tgt")

    def mu_mean(bm):
        xs = [i.mu_star.float().mean().item() for i in bm["test_shift"]]
        return sum(xs) / len(xs)

    # the two regimes ground the shift pool with different calibrators -> different mu_star
    assert abs(mu_mean(src) - mu_mean(tgt)) > 1e-3
    # in-dist (same condition as train) is unaffected by the regime
    assert src["meta"]["n_indist"] == tgt["meta"]["n_indist"]


def test_invalid_regime_rejected():
    import pytest
    with pytest.raises(ValueError, match="label_regime"):
        build_real_benchmark(_StubAdapter(), label_regime="nonsense", **BUILD)
