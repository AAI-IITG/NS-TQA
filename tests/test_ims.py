"""Invariant tests for the NASA IMS adapter.

Self-contained: synthesizes a tiny fake IMS tree in ``tmp_path`` with both column
layouts — an 8-column test (4 bearings x 2 axes) and a 4-column test (4 bearings x
1 axis, which the adapter duplicates to 2) — so both code paths are exercised
without the multi-GB raw download.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.adapters_ims import IMSAdapter
from benchmark.necessity import oracle_accuracy
from benchmark.realdata import build_real_benchmark, real_balance_report
from executor.hard_logic import evaluate
from perception.grounding import predicate_index
from perception.vibration import feature_names

N_BANDS = 2
N_SAMPLES = 256


def _write_test(test_dir: Path, n_cols: int, n_snaps: int, seed: int) -> None:
    test_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES)
    for k in range(n_snaps):
        sev = k / max(1, n_snaps - 1)
        cols = []
        for _ in range(n_cols):
            base = 0.5 * np.sin(2 * np.pi * 0.05 * t)
            noise = (0.2 + 1.5 * sev) * rng.standard_normal(N_SAMPLES)
            cols.append(base + noise)
        arr = np.stack(cols, axis=1)                        # [N, n_cols]
        # timestamp-like, zero-padded so lexicographic order == chronological
        np.savetxt(test_dir / f"2004.02.12.{k:06d}", arr, delimiter="\t")


def _fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "IMS"
    _write_test(root / "1st_test", n_cols=8, n_snaps=40, seed=1)   # 2 axes/bearing
    _write_test(root / "2nd_test", n_cols=4, n_snaps=44, seed=2)   # 1 axis/bearing -> dup
    return root


def _adapter(tmp_path, tests=("ims1", "ims2")):
    return IMSAdapter(root=_fake_root(tmp_path), tests=tests, n_bands=N_BANDS,
                      cache_path=tmp_path / "ims.pkl")


def test_both_column_layouts_load_to_14_like_vocabulary(tmp_path):
    series = _adapter(tmp_path).load()
    assert len(series) == 8                                  # 4 bearings x 2 tests
    C = series[0].C
    assert C == 2 * (4 + N_BANDS)                            # same shape for 8-col and 4-col tests
    assert series[0].channel_names == feature_names(N_BANDS)
    by_id = {s.unit_id: s for s in series}
    assert "ims:ims1:Bearing4" in by_id and "ims:ims2:Bearing1" in by_id
    for s in series:
        assert s.C == C and s.dataset == "ims"
        assert torch.isfinite(s.values).all()
        assert s.meta["rul"][0] == s.L and s.meta["rul"][-1] == 1
        assert s.meta["rul_is_proxy"] is True
    # known-failure flags wired from IMS_TESTS
    assert by_id["ims:ims1:Bearing3"].meta["fails"] is True   # run 1: B3, B4
    assert by_id["ims:ims1:Bearing1"].meta["fails"] is False
    assert by_id["ims:ims2:Bearing1"].meta["fails"] is True   # run 2: B1


def test_cache_deterministic(tmp_path):
    root = _fake_root(tmp_path)
    cache = tmp_path / "ims.pkl"
    s1 = IMSAdapter(root=root, n_bands=N_BANDS, cache_path=cache).load()
    assert cache.exists()
    s2 = IMSAdapter(root=root, n_bands=N_BANDS, cache_path=cache).load()
    a = {s.unit_id: s.values for s in s1}
    for s in s2:
        assert torch.equal(a[s.unit_id], s.values)


def test_cross_condition_build_balanced_and_non_circular(tmp_path):
    adapter = _adapter(tmp_path)
    bm = build_real_benchmark(
        adapter, T=12, stride=1, depths=(1, 2), shift="condition",
        train_conditions=("ims1",), test_conditions=("ims2",),
        indist_holdout_frac=0.5, n_train_per_depth=40, n_test_per_depth=40,
        hi_q=0.85, lo_q=0.15, smooth_k=5, a_level=4.0, over_factor=300, seed=0,
    )
    C = bm["meta"]["n_channels"]
    pidx = predicate_index(C)
    assert all(i.unit_id.startswith("ims:ims1:") for i in bm["train"])
    assert all(i.unit_id.startswith("ims:ims2:") for i in bm["test_shift"])   # cross-rig, no leak
    for split in ("train", "test_indist", "test_shift"):
        if not bm[split]:
            continue
        assert 0.45 <= real_balance_report(bm[split])["yes_frac"] <= 0.55
        for i in bm[split]:
            rho, _ = evaluate(i.mu_star, i.phi_star, pidx, 0)
            assert bool(rho > 0) == bool(i.answer_star)
        assert abs(oracle_accuracy(bm[split], pidx)["answer_accuracy"] - 1.0) < 1e-9
