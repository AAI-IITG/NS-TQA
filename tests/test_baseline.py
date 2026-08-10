import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from benchmark.baseline import (
    baseline_collate,
    encode_question,
    group_split_indices,
    query_dim,
    split_indices,
)
from benchmark.templates import generate_questions, sample_question
from models.baselines import LSTMBaseline


def _instance(question, answer=False, bearing_id=None, engine_id=None):
    return {
        "X": torch.randn(8, 3),
        "question": question,
        "phi_star": question.program,
        "answer_star": answer,
        "engine_id": engine_id,
        "bearing_id": bearing_id,
    }


def test_encode_question_is_fixed_and_deterministic():
    q = sample_question(n_channels=3, T=8, seed=0)
    a = encode_question(q, n_channels=3, T=8)
    b = encode_question(q, n_channels=3, T=8)
    assert a.shape == (query_dim(),)
    assert torch.allclose(a, b)


def test_encode_question_changes_with_template_or_slots():
    questions = generate_questions(20, n_channels=4, T=12, seed=2)
    encodings = [encode_question(q, n_channels=4, T=12) for q in questions]
    assert any(not torch.allclose(encodings[0], e) for e in encodings[1:])


def test_baseline_collate_shapes_and_metadata():
    questions = generate_questions(2, n_channels=3, T=8, seed=1)
    batch = [
        _instance(questions[0], answer=True, bearing_id="BearingA", engine_id=1),
        _instance(questions[1], answer=False, bearing_id="BearingB", engine_id=2),
    ]
    out = baseline_collate(batch, n_channels=3, T=8)
    assert out["X"].shape == (2, 8, 3)
    assert out["q"].shape == (2, query_dim())
    assert out["answer_star"].shape == (2,)
    assert out["answer_star"].dtype == torch.float32
    assert out["template"] == [questions[0].template, questions[1].template]
    assert out["engine_id"] == [1, 2]
    assert out["bearing_id"] == ["BearingA", "BearingB"]


def test_split_indices_is_deterministic_and_complete():
    a = split_indices(20, train_frac=0.7, val_frac=0.15, seed=0)
    b = split_indices(20, train_frac=0.7, val_frac=0.15, seed=0)
    assert a == b
    merged = a[0] + a[1] + a[2]
    assert sorted(merged) == list(range(20))
    assert len(a[0]) == 14
    assert len(a[1]) == 3
    assert len(a[2]) == 3


def test_group_split_indices_is_deterministic_and_group_disjoint():
    q = sample_question(n_channels=3, T=8, seed=0)
    instances = []
    for group in ["a", "b", "c", "d", "e"]:
        for _ in range(3):
            instances.append(_instance(q, bearing_id=group))

    a = group_split_indices(instances, "bearing_id", 0.6, 0.2, seed=7)
    b = group_split_indices(instances, "bearing_id", 0.6, 0.2, seed=7)
    assert a == b

    train, val, test, meta = a
    assert sorted(train + val + test) == list(range(len(instances)))

    def groups(indices):
        return {instances[i]["bearing_id"] for i in indices}

    assert groups(train).isdisjoint(groups(val))
    assert groups(train).isdisjoint(groups(test))
    assert groups(val).isdisjoint(groups(test))
    assert meta["group_key"] == "bearing_id"


def test_lstm_baseline_forward_and_training_step():
    torch.manual_seed(0)
    model = LSTMBaseline(n_channels=3, query_dim=query_dim(), hidden=8)
    X = torch.randn(4, 8, 3)
    q = torch.randn(4, query_dim())
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    logits = model(X, q)
    assert logits.shape == (4,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
