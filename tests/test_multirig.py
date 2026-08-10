"""Unit tests for MultiRigAdapter (no raw data needed — uses tiny stub adapters)."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.adapters import DatasetAdapter, Series
from benchmark.adapters_multirig import MultiRigAdapter


class _StubRig(DatasetAdapter):
    """Returns two hand-built bearings tagged with a given dataset/rig name."""

    def __init__(self, rig: str, names: list[str], condition: str = "c0"):
        self._rig, self._names, self._cond = rig, names, condition

    def load(self) -> list[Series]:
        out = []
        for n in (1, 2):
            out.append(Series(
                values=torch.randn(20, len(self._names)),
                unit_id=f"{self._rig}:{self._cond}:B{n}",
                dataset=self._rig, channel_names=list(self._names),
                condition=self._cond, fs=1.0, meta={"bearing": f"B{n}"}))
        return out


NAMES = [f"h_f{i}" for i in range(7)] + [f"v_f{i}" for i in range(7)]


def test_concatenates_and_retags_condition_to_rig():
    adp = MultiRigAdapter([_StubRig("xjtu", NAMES, "35Hz"),
                           _StubRig("femto", NAMES, "1"),
                           _StubRig("ims", NAMES, "ims2")])
    series = adp.load()
    assert len(series) == 6
    # condition is now the RIG name; unit_id and orig_condition preserved
    assert sorted({s.condition for s in series}) == ["femto", "ims", "xjtu"]
    assert adp.conditions(series) == ["femto", "ims", "xjtu"]
    by_rig = {}
    for s in series:
        by_rig.setdefault(s.condition, []).append(s)
        assert s.meta["rig"] == s.condition == s.dataset
        assert s.meta["orig_condition"] in ("35Hz", "1", "ims2")
        assert s.unit_id.startswith(f"{s.condition}:")
    assert {r: len(v) for r, v in by_rig.items()} == {"xjtu": 2, "femto": 2, "ims": 2}
    assert all(s.channel_names == NAMES for s in series)


def test_rejects_channel_vocabulary_mismatch():
    other = [f"h_f{i}" for i in range(5)] + [f"v_f{i}" for i in range(5)]  # 10 != 14
    adp = MultiRigAdapter([_StubRig("xjtu", NAMES), _StubRig("femto", other)])
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        adp.load()


def test_empty_children_rejected():
    with pytest.raises(ValueError):
        MultiRigAdapter([])
