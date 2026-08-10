"""WO-1C: leakage-safe anomaly-family benchmark invariants. Uses a tiny synthetic
adapter (no raw data). Also a regression guard that the 5-family path leaves the
global 4-family grounding untouched."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.adapters import DatasetAdapter, Series
from benchmark.anomaly_questions import (build_safe_anomaly_benchmark, heldout_leakage,
                                         is_composition_safe, _leaf_families,
                                         sample_safe_anomaly_program)
from benchmark.anomaly_qa import oracle_anomaly_eval, pidx5
from executor.grammar import Predicate
from perception.grounding import PRED_FAMILIES, predicate_index


class _StubBearings(DatasetAdapter):
    """A few synthetic 'bearings' per condition with an upward-trending HI signal."""
    name = "stub"

    def load(self):
        names = [f"h_f{i}" for i in range(7)] + [f"v_f{i}" for i in range(7)]
        out = []
        g = torch.Generator().manual_seed(0)
        for cond in ("35Hz12kN", "37.5Hz11kN", "40Hz10kN"):
            for bnum in range(4):
                n = 160
                trend = torch.linspace(0, 3, n).unsqueeze(1)      # degradation ramp
                vals = trend + 0.3 * torch.randn(n, 14, generator=g)
                out.append(Series(values=vals, unit_id=f"{cond}:B{bnum}", dataset="stub",
                                  channel_names=names, condition=cond, fs=1.0,
                                  meta={"bearing": f"B{bnum}"}))
        return out


def _bm(**over):
    kw = dict(T=24, stride=2, depths=(1, 2, 3, 4), indist_holdout_frac=0.5,
              healthy_frac=0.3, n_train_per_depth=60, n_test_per_depth=120,
              anomaly_p=0.4, max_windows_per_unit=60, over_factor=150, seed=0,
              leak_max=0.55, max_leak_retries=8)
    kw.update(over)
    return build_safe_anomaly_benchmark(_StubBearings(), **kw)


def test_pidx5_is_five_families_and_global_untouched():
    C = 14
    assert len(pidx5(C)) == 5 * C
    # global 4-family grounding is unchanged (we did NOT reindex it)
    assert PRED_FAMILIES == ["high", "low", "rising", "falling"]
    assert len(predicate_index(C)) == 4 * C


def test_composition_constraint_holds_at_depth_ge_2():
    bm = _bm()
    viol = [i for i in bm["test_shift"] + bm["test_indist"]
            if not is_composition_safe(i.phi_star, i.depth)]
    assert not viol, f"{len(viol)} depth>=2 programs decidable from anomalous alone"


def test_sampler_rejects_anomaly_only_deep_programs():
    import random
    rng = random.Random(0)
    for _ in range(50):
        phi = sample_safe_anomaly_program(3, list(range(14)), 24, rng, anomaly_p=1.0)
        if phi is None:
            continue
        fams = _leaf_families(phi)
        # with anomaly_p=1 the sampler must still inject a non-anomaly leaf to be safe
        assert ("anomalous" not in fams) or any(f in ("high", "low", "rising", "falling") for f in fams)


def test_balanced_and_oracle_perfect():
    bm = _bm()
    for split in ("train", "test_indist", "test_shift"):
        insts = bm[split]
        yes = sum(i.answer_star for i in insts)
        assert 0.4 <= yes / len(insts) <= 0.6, f"{split} imbalance {yes/len(insts):.2f}"
    for pool in ("test_indist", "test_shift"):
        assert oracle_anomaly_eval(bm[pool])["answer_accuracy"] > 0.999


def test_mu_star_is_five_family_shape():
    bm = _bm()
    C = bm["meta"]["n_channels"]
    i = bm["test_shift"][0]
    assert i.mu_star.shape[1] == 5 * C


def test_heldout_leakage_below_optimistic():
    bm = _bm()
    shift = bm["test_shift"]
    # held-out probe should not exceed the trivial "always majority" by a wild margin
    ho = heldout_leakage(shift, seed=1)
    assert 0.4 <= ho <= 0.85
