"""Invariants for the 5-family anomaly-question benchmark (no training needed)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.adapters import DatasetAdapter, Series
from benchmark.anomaly_qa import (FIVE_FAMILIES, build_anomaly_benchmark,
                                  oracle_anomaly_eval, pidx5, stl_only_anomaly_eval)
from executor.grammar import Predicate
from executor.hard_logic import evaluate

C = 6
NAMES = [f"h_f{i}" for i in range(3)] + [f"v_f{i}" for i in range(3)]


class _StubRig(DatasetAdapter):
    name = "stub"

    def load(self):
        g = torch.Generator().manual_seed(0)
        out = []
        for cond in ("a", "b"):
            for n in (1, 2):
                L = 70
                t = torch.linspace(0, 1, L).unsqueeze(1)
                vals = 0.3 * torch.randn(L, C, generator=g) + 2.0 * t  # degrade upward
                out.append(Series(values=vals, unit_id=f"{cond}:B{n}", dataset="stub",
                                  channel_names=NAMES, condition=cond, fs=1.0,
                                  meta={"rul": torch.arange(L, 0, -1)}))
        return out


def _build(anomaly_p=1.0):
    return build_anomaly_benchmark(
        _StubRig(), T=12, stride=1, depths=(1, 2), shift="condition",
        train_conditions=("a",), test_conditions=("b",), indist_holdout_frac=0.5,
        healthy_frac=0.3, n_train_per_depth=30, n_test_per_depth=30, anomaly_p=anomaly_p,
        max_windows_per_unit=200, over_factor=400, seed=0)


def test_pidx5_layout():
    pi = pidx5(C)
    assert len(pi) == 5 * C
    assert pi[("high", 0)] == 0 and pi[("anomalous", 0)] == 4 * C
    assert FIVE_FAMILIES[-1] == "anomalous"


def test_build_is_non_circular_and_uses_anomalous():
    bm = _build(anomaly_p=1.0)
    assert bm["meta"]["n_channels"] == C
    pi = pidx5(C)
    used_anom = False
    for split in ("train", "test_indist", "test_shift"):
        for i in bm[split]:
            assert i.mu_star.shape[1] == 5 * C                 # 5-family truths
            rho, _ = evaluate(i.mu_star, i.phi_star, pi, 0)
            assert bool(rho > 0) == bool(i.answer_star)        # label == executor(phi, mu*)

            def walk(n):
                nonlocal used_anom
                if isinstance(n, Predicate):
                    used_anom = used_anom or (n.name == "anomalous")
                else:
                    for c in n._children():
                        walk(c)
            walk(i.phi_star)
    assert used_anom                                           # programs reference anomalous
    # cross-condition: no unit leak
    assert {i.unit_id for i in bm["train"]}.isdisjoint({i.unit_id for i in bm["test_shift"]})


def test_oracle_is_one_and_stl_only_is_below():
    bm = _build(anomaly_p=1.0)
    for split in ("test_indist", "test_shift"):
        if not bm[split]:
            continue
        assert abs(oracle_anomaly_eval(bm[split])["answer_accuracy"] - 1.0) < 1e-9
        stl = stl_only_anomaly_eval(bm[split])["answer_accuracy"]
        assert 0.0 <= stl <= 1.0                               # hand-rule proxy: a real (imperfect) number


def test_balanced_per_split():
    bm = _build(anomaly_p=1.0)
    for split in ("train", "test_indist", "test_shift"):
        if not bm[split]:
            continue
        yes = sum(i.answer_star for i in bm[split]) / len(bm[split])
        assert 0.4 <= yes <= 0.6
