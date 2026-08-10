"""Invariant tests for the FEMTO/PRONOSTIA adapter.

Self-contained: synthesizes a tiny fake PRONOSTIA tree in ``tmp_path`` (two
conditions, ``acc_*.csv`` snapshots, one bearing written with ';' delimiters to
exercise the mixed-delimiter reader) so the tests need neither the raw download
nor the cached HI trajectories.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.adapters_femto import FEMTOAdapter
from benchmark.necessity import oracle_accuracy
from benchmark.realdata import build_real_benchmark, real_balance_report
from executor.hard_logic import evaluate
from perception.grounding import predicate_index
from perception.vibration import feature_names

N_BANDS = 2
N_SAMPLES = 512


def _snap(sev: float, rng) -> np.ndarray:
    t = np.arange(N_SAMPLES)
    base = 0.5 * np.sin(2 * np.pi * 0.05 * t)
    noise = (0.2 + 1.5 * sev) * rng.standard_normal(N_SAMPLES)
    imp = np.zeros(N_SAMPLES)
    imp[::32] = (3.0 * sev) * rng.standard_normal(len(imp[::32]))
    return np.stack([base + noise + imp, 0.8 * base + noise + 0.7 * imp], axis=1)


def _write_bearing(bdir: Path, n: int, seed: int, delim: str = ",") -> None:
    bdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for k in range(1, n + 1):
        arr = _snap((k - 1) / max(1, n - 1), rng)
        with open(bdir / f"acc_{k:05d}.csv", "w") as fh:
            for h, v in arr:
                # 6 columns: hour, minute, second, microsecond, horiz, vert
                fh.write(delim.join(["0", "0", "0", str(k), f"{h:.5f}", f"{v:.5f}"]) + "\n")


def _fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    # condition 1: two bearings in Learning_set (one comma, one semicolon)
    _write_bearing(root / "Learning_set" / "Bearing1_1", 40, 1, delim=",")
    _write_bearing(root / "Learning_set" / "Bearing1_2", 45, 2, delim=";")
    # condition 2: two bearings in Full_Test_Set
    _write_bearing(root / "Full_Test_Set" / "Bearing2_1", 38, 3, delim=",")
    _write_bearing(root / "Full_Test_Set" / "Bearing2_2", 42, 4, delim=",")
    # a non-acc folder that must be ignored
    (root / "Full_Test_Set" / "NotABearing").mkdir(parents=True, exist_ok=True)
    return root


def _adapter(tmp_path):
    return FEMTOAdapter(root=_fake_root(tmp_path), n_bands=N_BANDS,
                        cache_path=tmp_path / "femto.pkl")


def test_loads_consistent_channels_conditions_and_rul(tmp_path):
    series = _adapter(tmp_path).load()
    assert len(series) == 4
    C = series[0].C
    assert C == 2 * (4 + N_BANDS)
    assert series[0].channel_names == feature_names(N_BANDS)
    by_id = {s.unit_id: s for s in series}
    assert set(by_id) == {"femto:1:Bearing1_1", "femto:1:Bearing1_2",
                          "femto:2:Bearing2_1", "femto:2:Bearing2_2"}
    assert sorted({s.condition for s in series}) == ["1", "2"]
    for s in series:
        assert s.C == C and s.dataset == "femto"
        assert torch.isfinite(s.values).all()
        assert s.meta["rul"][0] == s.L and s.meta["rul"][-1] == 1
    # the ';'-delimited bearing parsed to the same width
    assert by_id["femto:1:Bearing1_2"].C == C


def test_cache_is_deterministic(tmp_path):
    root = _fake_root(tmp_path)
    cache = tmp_path / "femto.pkl"
    s1 = FEMTOAdapter(root=root, n_bands=N_BANDS, cache_path=cache).load()
    assert cache.exists()
    s2 = FEMTOAdapter(root=root, n_bands=N_BANDS, cache_path=cache).load()
    a = {s.unit_id: s.values for s in s1}
    for s in s2:
        assert torch.equal(a[s.unit_id], s.values)


def test_cross_condition_build_balanced_and_non_circular(tmp_path):
    adapter = _adapter(tmp_path)
    bm = build_real_benchmark(
        adapter, T=12, stride=1, depths=(1, 2), shift="condition",
        train_conditions=("1",), test_conditions=("2",),
        indist_holdout_frac=0.5, n_train_per_depth=40, n_test_per_depth=40,
        hi_q=0.85, lo_q=0.15, smooth_k=5, a_level=4.0, over_factor=300, seed=0,
    )
    C = bm["meta"]["n_channels"]
    pidx = predicate_index(C)
    train_u = {i.unit_id for i in bm["train"]}
    shift_u = {i.unit_id for i in bm["test_shift"]}
    assert all(u.startswith("femto:1:") for u in train_u)
    assert all(u.startswith("femto:2:") for u in shift_u)        # cross-condition, no leak
    for split in ("train", "test_indist", "test_shift"):
        if not bm[split]:
            continue
        rep = real_balance_report(bm[split])
        assert 0.45 <= rep["yes_frac"] <= 0.55
        for i in bm[split]:
            rho, _ = evaluate(i.mu_star, i.phi_star, pidx, 0)
            assert bool(rho > 0) == bool(i.answer_star)
        assert abs(oracle_accuracy(bm[split], pidx)["answer_accuracy"] - 1.0) < 1e-9
