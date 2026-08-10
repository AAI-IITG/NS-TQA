"""Invariant tests for the XJTU vibration front-end + adapter.

Self-contained: synthesizes a tiny fake XJTU directory tree in the pytest
``tmp_path`` (two bearings of short degrading snapshots) so the tests are fast
and need neither the multi-GB raw download nor the cached HI trajectories.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.adapters_xjtu import XJTUAdapter
from benchmark.necessity import oracle_accuracy
from benchmark.realdata import build_real_benchmark, real_balance_report
from benchmark.spurious_real import attach_spurious_channel
from executor.grammar import Predicate
from executor.hard_logic import evaluate
from perception.grounding import predicate_index
from perception.vibration import (bearing_trajectory, feature_names,
                                  snapshot_features)

N_BANDS = 2
N_SAMPLES = 1024


def _fake_snapshot(severity: float, rng: np.random.Generator) -> np.ndarray:
    """One [N,2] snapshot whose energy/impulsiveness grows with severity."""
    t = np.arange(N_SAMPLES)
    base = 0.5 * np.sin(2 * np.pi * 0.05 * t)
    noise = (0.2 + 1.5 * severity) * rng.standard_normal(N_SAMPLES)
    impulses = np.zeros(N_SAMPLES)
    impulses[::64] = (3.0 * severity) * rng.standard_normal(len(impulses[::64]))
    h = base + noise + impulses
    v = 0.8 * base + noise + 0.7 * impulses
    return np.stack([h, v], axis=1)


def _write_bearing(bdir: Path, indices: list[int], seed: int) -> None:
    bdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = len(indices)
    for pos, k in enumerate(indices):
        sev = pos / max(1, n - 1)
        arr = _fake_snapshot(sev, rng)
        with open(bdir / f"{k}.csv", "w") as fh:
            fh.write("Horizontal_vibration_signals,Vertical_vibration_signals\n")
            np.savetxt(fh, arr, delimiter=",")


def _fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "XJTU"
    # Bearing3_4: 60 snapshots with a GAP (no index 1, jumps) -> tests numeric order.
    idx4 = [k for k in range(2, 80) if k % 13 != 0][:60]
    _write_bearing(root / "40Hz10kN" / "Bearing3_4", idx4, seed=1)
    # Bearing3_5: 50 contiguous snapshots.
    _write_bearing(root / "40Hz10kN" / "Bearing3_5", list(range(1, 51)), seed=2)
    # An empty bearing folder that must be skipped.
    (root / "40Hz10kN" / "Bearing3_1").mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------- #
# front-end
# --------------------------------------------------------------------------- #

def test_front_end_deterministic_and_shaped():
    rng = np.random.default_rng(0)
    snap = rng.standard_normal((N_SAMPLES, 2))
    f1, names = snapshot_features(snap, n_bands=N_BANDS)
    f2, _ = snapshot_features(snap, n_bands=N_BANDS)
    assert np.array_equal(f1, f2)                  # deterministic
    assert names == feature_names(N_BANDS)
    assert f1.shape == (2 * (4 + N_BANDS),)        # H ++ V


def test_front_end_trends_up_on_degradation(tmp_path):
    bdir = tmp_path / "B"
    _write_bearing(bdir, list(range(1, 26)), seed=3)
    vals, names = bearing_trajectory(bdir, n_bands=N_BANDS)
    assert vals.shape == (25, len(names))
    for ch in ("h_rms", "h_kurtosis"):
        y = vals[:, names.index(ch)]
        slope = np.polyfit(np.arange(len(y)), y, 1)[0]
        assert slope > 0, f"{ch} should trend upward on a degrading signal"


# --------------------------------------------------------------------------- #
# adapter
# --------------------------------------------------------------------------- #

def test_adapter_loads_consistent_channels_and_rul(tmp_path):
    root = _fake_root(tmp_path)
    cache = tmp_path / "cache.pkl"
    series = XJTUAdapter(root=root, n_bands=N_BANDS, cache_path=cache).load()

    assert len(series) == 2                         # empty Bearing3_1 skipped
    C = series[0].C
    assert C == 2 * (4 + N_BANDS)
    by_id = {s.unit_id: s for s in series}
    assert set(by_id) == {"40Hz10kN:Bearing3_4", "40Hz10kN:Bearing3_5"}
    for s in series:
        assert s.C == C and s.channel_names == series[0].channel_names  # consistent set
        assert s.condition == "40Hz10kN" and s.dataset == "xjtu" and s.fs == 25600.0
        assert torch.isfinite(s.values).all()
        rul = s.meta["rul"]
        assert rul[0] == s.L and rul[-1] == 1       # RUL = n..1
        assert int(rul[0]) == s.meta["n_snapshots"]
    assert by_id["40Hz10kN:Bearing3_4"].L == 60
    assert by_id["40Hz10kN:Bearing3_5"].L == 50


def test_adapter_cache_is_deterministic(tmp_path):
    root = _fake_root(tmp_path)
    cache = tmp_path / "cache.pkl"
    s1 = XJTUAdapter(root=root, n_bands=N_BANDS, cache_path=cache).load()
    assert cache.exists()                           # cache written
    s2 = XJTUAdapter(root=root, n_bands=N_BANDS, cache_path=cache).load()  # cache hit
    a = {s.unit_id: s.values for s in s1}
    b = {s.unit_id: s.values for s in s2}
    for uid in a:
        assert torch.equal(a[uid], b[uid])


# --------------------------------------------------------------------------- #
# non-circular benchmark invariants
# --------------------------------------------------------------------------- #

def _build(tmp_path):
    root = _fake_root(tmp_path)
    adapter = XJTUAdapter(root=root, n_bands=N_BANDS, cache_path=tmp_path / "c.pkl")
    return build_real_benchmark(
        adapter, T=12, stride=1, depths=(1, 2),
        shift="unit", n_train_per_depth=40, n_test_per_depth=40,
        hi_q=0.85, lo_q=0.15, smooth_k=5, a_level=4.0,
        over_factor=300, seed=0,
    )


def test_benchmark_balanced_per_split_and_depth(tmp_path):
    bm = _build(tmp_path)
    for split in ("train", "test_indist"):
        rep = real_balance_report(bm[split])
        assert 0.45 <= rep["yes_frac"] <= 0.55
        for d, v in rep["per_depth"].items():
            assert 0.45 <= v["yes_frac"] <= 0.55, f"{split} depth {d} imbalanced"


def test_benchmark_non_circular_and_oracle_exact(tmp_path):
    bm = _build(tmp_path)
    C = bm["meta"]["n_channels"]
    pidx = predicate_index(C)
    # held-out bearing is a DIFFERENT unit than train (grouped split, no leak)
    train_u = {i.unit_id for i in bm["train"]}
    test_u = {i.unit_id for i in bm["test_indist"]}
    assert train_u and test_u and train_u.isdisjoint(test_u)
    for split in ("train", "test_indist"):
        for i in bm[split]:
            rho, _ = evaluate(i.mu_star, i.phi_star, pidx, 0)
            assert bool(rho > 0) == bool(i.answer_star)   # label == executor(phi, mu*)
        assert abs(oracle_accuracy(bm[split], pidx)["answer_accuracy"] - 1.0) < 1e-9


def test_spurious_bridge_keeps_causal_program_off_spurious_channel(tmp_path):
    bm = _build(tmp_path)
    C = bm["meta"]["n_channels"]
    sbm = attach_spurious_channel(bm, shift_mode="shift_decorr", strength=3.0, seed=0)
    assert sbm["meta"]["n_channels"] == C + 1
    sp = sbm["meta"]["spurious_channel"]
    assert sp == C

    def channels_used(node):
        if isinstance(node, Predicate):
            return {node.channel}
        out = set()
        for c in node._children():
            out |= channels_used(c)
        return out

    for i in sbm["train"]:
        assert i.X.shape[1] == C + 1 and i.mu_star.shape[1] == 4 * (C + 1)
        # causal program references only REAL channels (< C), never the shortcut
        assert all(ch < C for ch in channels_used(i.phi_star))
