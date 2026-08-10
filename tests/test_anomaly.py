"""Learned `anomalous` predicate: privileged healthy-baseline target behaves, and
the neural head learns it (where a fixed threshold cannot)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perception.anomaly import (HealthyBaseline, predict_anomaly,
                                train_anomaly_head)


def _healthy(n, T, C, scale, seed):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(T * 0 + T, C, generator=g) if False else \
        scale * torch.randn(n, T, C, generator=g)


def test_target_fires_on_deviation_not_on_healthy():
    T, C = 24, 2
    healthy = 0.3 * torch.randn(40, T, C, generator=torch.Generator().manual_seed(0))
    base = HealthyBaseline.fit(healthy, q=0.99, a_anom=4.0, smooth_k=3)
    # a window matching the healthy scale -> low anomaly
    calm = 0.3 * torch.randn(T, C, generator=torch.Generator().manual_seed(1))
    # a window pushed far outside the healthy band on ch0 -> high anomaly there
    bad = calm.clone()
    bad[:, 0] += 6.0
    assert base.target(calm).mean() < 0.3
    assert base.target(bad)[:, 0].mean() > 0.8
    # deterministic
    assert torch.allclose(base.target(bad), base.target(bad))


def test_head_learns_the_target():
    T, C = 24, 3
    g = torch.Generator().manual_seed(0)
    healthy = 0.3 * torch.randn(60, T, C, generator=g)
    base = HealthyBaseline.fit(healthy, q=0.99, smooth_k=3)
    # mixed windows: half calm, half with a random channel deviated
    wins = []
    for k in range(80):
        w = 0.3 * torch.randn(T, C, generator=g)
        if k % 2 == 0:
            w[:, k % C] += 5.0
        wins.append(w)
    X = torch.stack(wins)
    head = train_anomaly_head(X, base, hidden=16, epochs=40, seed=0)
    gold = torch.stack([base.target(x) for x in X]) > 0.5
    pred = torch.stack([predict_anomaly(head, x) for x in X]) > 0.5
    tp = float((pred & gold).sum()); fp = float((pred & ~gold).sum()); fn = float((~pred & gold).sum())
    p = tp / max(1.0, tp + fp); r = tp / max(1.0, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    assert f1 > 0.75, f"head failed to learn the anomaly target (F1={f1:.3f})"
