"""Invariant tests for input-side sensor degradation (WO-1B). No raw data needed."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.degrade import (add_noise, calibration_drift, degrade_instances,
                               drop_samples, resample_jitter)


def _X():
    torch.manual_seed(0)
    return torch.randn(40, 8)


def test_shape_and_finite_all_families():
    X = _X()
    for Y in [add_noise(X, 10), drop_samples(X, 0.2, "zero"), drop_samples(X, 0.2, "hold"),
              drop_samples(X, 0.2, "interp"), calibration_drift(X, 1.1, 0.2),
              resample_jitter(X, 1.1)]:
        assert Y.shape == X.shape
        assert torch.isfinite(Y).all()


def test_determinism_and_seed_sensitivity():
    X = _X()
    assert torch.equal(add_noise(X, 8, seed=1), add_noise(X, 8, seed=1))
    assert not torch.equal(add_noise(X, 8, seed=1), add_noise(X, 8, seed=2))


def test_noise_monotone_in_snr():
    X = _X()
    mses = [(add_noise(X, s, seed=1) - X).pow(2).mean().item() for s in (20, 10, 5, 0)]
    assert mses[0] < mses[1] < mses[2] < mses[3]


def test_drift_is_exact_affine():
    X = _X()
    assert torch.allclose(calibration_drift(X, gain=2.0, offset=1.0), X * 2 + 1)


def test_drop_zero_actually_zeros_and_hold_carries():
    X = _X()
    Yz = drop_samples(X, 0.5, "zero", seed=3)
    assert (Yz == 0).any()                        # some samples were zeroed
    Yh = drop_samples(X, 0.5, "hold", seed=3)
    assert torch.isfinite(Yh).all()              # no NaNs from leading gaps


def test_p_out_of_range_raises():
    with pytest.raises(ValueError):
        drop_samples(_X(), 1.5, "zero")


class _Inst:
    def __init__(self, X):
        self.X = X
        self.answer_star = True
        self.mu_star = torch.ones(X.shape[0], 4 * X.shape[1])
        self.depth = 2


def test_degrade_instances_preserves_labels_and_clean():
    X = _X()
    insts = [_Inst(X.clone()) for _ in range(4)]
    deg = degrade_instances(insts, "noise", seed=0, snr_db=5)
    assert len(deg) == 4
    # labels/mu_star carried over (shared object), X degraded, per-instance variety
    assert all(d.answer_star == o.answer_star and d.mu_star is o.mu_star
               for d, o in zip(deg, insts))
    assert not torch.equal(deg[0].X, insts[0].X)
    assert not torch.equal(deg[0].X, deg[1].X)
    # original instances untouched (no in-place corruption of the clean benchmark)
    assert torch.equal(insts[0].X, X)


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        degrade_instances([_Inst(_X())], "bogus", snr_db=1)
