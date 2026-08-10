"""Perception tests: fuzzy truths stay in [0,1], shapes correct, trends sensible."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from perception.grounding import (
    Calibrator,
    ground,
    local_slope,
    predicate_index,
    smooth,
)


def test_grounding_in_unit_interval():
    torch.manual_seed(0)
    X = torch.randn(8, 20, 3)
    cal = Calibrator.fit(X)
    mu, _ = ground(X[0], cal)
    assert mu.min() >= 0.0 and mu.max() <= 1.0


def test_grounding_shape_and_index():
    X = torch.randn(4, 16, 5)
    cal = Calibrator.fit(X)
    mu, pidx = ground(X[0], cal)
    assert mu.shape == (16, 20)  # 4 families * 5 channels
    assert len(pidx) == 20
    assert predicate_index(5) == pidx


def test_rising_detected_for_uptrend():
    X = torch.zeros(4, 20, 1)
    ramp = torch.linspace(0, 5, 20).view(1, 20, 1)
    X = X + ramp
    cal = Calibrator.fit(X)
    mu, pidx = ground(X[0], cal)
    rising = mu[:, pidx[("rising", 0)]].mean()
    falling = mu[:, pidx[("falling", 0)]].mean()
    assert rising > falling


def test_smooth_preserves_length():
    x = torch.randn(20, 3)
    assert smooth(x, k=5).shape == x.shape


def test_slope_zero_for_constant():
    x = torch.ones(20, 2)
    d = local_slope(x, k=3)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-5)
