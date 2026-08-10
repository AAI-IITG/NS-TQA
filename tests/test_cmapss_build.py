import importlib.util
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.templates import sample_question
from perception.grounding import Calibrator


def _load_build_script():
    path = ROOT / "scripts" / "02_build_cmapss_qa.py"
    spec = importlib.util.spec_from_file_location("build_cmapss_qa", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cmapss_build_instance_pickles_compact_window(tmp_path):
    build = _load_build_script()
    X = torch.randn(100, 48, 3)
    cal = Calibrator.fit(X, smooth_k=1)
    question = sample_question(n_channels=3, T=48, seed=0)

    instance = build.build_instance(X[0], 12.0, question, cal, engine_id=7)
    out = tmp_path / "instance.pkl"
    with open(out, "wb") as f:
        pickle.dump([instance], f)

    assert instance["X"].shape == (48, 3)
    assert instance["engine_id"] == 7
    assert instance["X"].untyped_storage().data_ptr() != X.untyped_storage().data_ptr()
    assert out.stat().st_size < 100_000
