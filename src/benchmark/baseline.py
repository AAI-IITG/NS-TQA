"""Utilities for end-to-end QA baselines.

These helpers are for ablation models that answer directly from
``(signal, encoded structured question)``. They must not be used on the
faithful symbolic answer path, where the STL executor remains the source of
truth.
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any

import torch

from benchmark.templates import Question
from executor.grammar import Always, And, Eventually, Predicate, Until

TEMPLATE_NAMES = ["existence", "persistence", "temporal_order", "conjunction"]
PREDICATE_NAMES = ["high", "low", "rising", "falling"]


def query_dim() -> int:
    """Fixed question encoding dimension."""
    # template one-hot + two predicate one-hots + c1,c2,a1,b1,a2,b2
    return len(TEMPLATE_NAMES) + 2 * len(PREDICATE_NAMES) + 6


def _one_hot(value: str, names: list[str]) -> list[float]:
    out = [0.0] * len(names)
    if value in names:
        out[names.index(value)] = 1.0
    return out


def _norm_channel(channel: int | None, n_channels: int) -> float:
    if channel is None:
        return 0.0
    denom = max(1, n_channels - 1)
    return float(channel) / denom


def _norm_time(t: int | None, T: int) -> float:
    if t is None:
        return 0.0
    denom = max(1, T - 1)
    return float(t) / denom


def _extract_slots(question: Question) -> dict[str, Any]:
    """Extract predicate/channel/interval slots from the executable program."""
    phi = question.program

    if question.template == "existence" and isinstance(phi, Eventually):
        child = phi.child
        if isinstance(child, Predicate):
            return {"p1": child.name, "c1": child.channel, "a1": phi.a, "b1": phi.b}

    if question.template == "persistence" and isinstance(phi, Always):
        child = phi.child
        if isinstance(child, Predicate):
            return {"p1": child.name, "c1": child.channel, "a1": phi.a, "b1": phi.b}

    if question.template == "temporal_order" and isinstance(phi, Until):
        left, right = phi.left, phi.right
        if isinstance(left, Predicate) and isinstance(right, Predicate):
            return {
                "p1": left.name,
                "c1": left.channel,
                "p2": right.name,
                "c2": right.channel,
                "a1": phi.a,
                "b1": phi.b,
            }

    if question.template == "conjunction" and isinstance(phi, And):
        left, right = phi.left, phi.right
        if isinstance(left, Always) and isinstance(right, Eventually):
            lchild, rchild = left.child, right.child
            if isinstance(lchild, Predicate) and isinstance(rchild, Predicate):
                return {
                    "p1": lchild.name,
                    "c1": lchild.channel,
                    "a1": left.a,
                    "b1": left.b,
                    "p2": rchild.name,
                    "c2": rchild.channel,
                    "a2": right.a,
                    "b2": right.b,
                }

    raise ValueError(f"cannot encode question template/program: {question}")


def encode_question(question: Question, n_channels: int, T: int) -> torch.Tensor:
    """Encode one structured question as a fixed-length float tensor."""
    slots = _extract_slots(question)
    values = []
    values += _one_hot(question.template, TEMPLATE_NAMES)
    values += _one_hot(slots.get("p1"), PREDICATE_NAMES)
    values += _one_hot(slots.get("p2"), PREDICATE_NAMES)
    values += [
        _norm_channel(slots.get("c1"), n_channels),
        _norm_channel(slots.get("c2"), n_channels),
        _norm_time(slots.get("a1"), T),
        _norm_time(slots.get("b1"), T),
        _norm_time(slots.get("a2"), T),
        _norm_time(slots.get("b2"), T),
    ]
    return torch.tensor(values, dtype=torch.float32)


def _value(instance: Any, key: str, default: Any = None) -> Any:
    if isinstance(instance, dict):
        return instance.get(key, default)
    return getattr(instance, key, default)


def baseline_collate(batch: list[dict], n_channels: int, T: int) -> dict:
    """Collate QA instances for direct-answer baselines."""
    questions = [_value(b, "question") for b in batch]
    return {
        "X": torch.stack([_value(b, "X") for b in batch]),
        "q": torch.stack([encode_question(q, n_channels, T) for q in questions]),
        "answer_star": torch.tensor(
            [float(_value(b, "answer_star")) for b in batch], dtype=torch.float32
        ),
        "question": questions,
        "phi_star": [_value(b, "phi_star") for b in batch],
        "template": [q.template for q in questions],
        "engine_id": [_value(b, "engine_id") for b in batch],
        "bearing_id": [_value(b, "bearing_id") for b in batch],
    }


def load_qa_pickle(path: str | Path) -> dict:
    """Load a QA pickle artifact and normalize list-only artifacts."""
    path = Path(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "instances" in obj:
        return obj
    return {"instances": obj}


def split_indices(
    n: int, train_frac: float, val_frac: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Deterministically shuffle and split indices into train/val/test."""
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1)")
    if not (0.0 <= val_frac < 1.0):
        raise ValueError("val_frac must be in [0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = indices[:n_train]
    val = indices[n_train : n_train + n_val]
    test = indices[n_train + n_val :]
    return train, val, test


def group_split_indices(
    instances: list,
    group_key: str,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[int], list[int], list[int], dict]:
    """Deterministically split by group so groups do not cross splits."""
    groups: dict[str, list[int]] = {}
    for i, instance in enumerate(instances):
        value = _value(instance, group_key)
        if value is None:
            raise ValueError(f"instance {i} missing group key {group_key!r}")
        groups.setdefault(str(value), []).append(i)

    group_names = list(groups)
    rng = random.Random(seed)
    order = list(range(len(group_names)))
    rng.shuffle(order)
    n_groups = len(order)
    n_train = int(n_groups * train_frac)
    n_val = int(n_groups * val_frac)
    if val_frac > 0.0 and n_groups >= 3:
        n_val = max(1, n_val)
    n_test = n_groups - n_train - n_val
    if n_test <= 0 and n_groups >= 3:
        n_test = 1
        n_train = max(1, n_groups - n_val - n_test)
    if n_train <= 0:
        raise ValueError("not enough groups for requested split")
    train_g = order[:n_train]
    val_g = order[n_train : n_train + n_val]
    test_g = order[n_train + n_val :]
    train_groups = [group_names[i] for i in train_g]
    val_groups = [group_names[i] for i in val_g]
    test_groups = [group_names[i] for i in test_g]

    def expand(names: list[str]) -> list[int]:
        out = []
        for name in names:
            out.extend(groups[name])
        return out

    meta = {
        "group_key": group_key,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "test_groups": test_groups,
    }
    return expand(train_groups), expand(val_groups), expand(test_groups), meta


def select_device(preference: str = "cpu") -> torch.device:
    """Choose a device, validating CUDA with a real tiny kernel."""
    if preference != "cuda":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        x = torch.ones(1, device="cuda")
        y = x + 1.0
        torch.cuda.synchronize()
        if float(y.item()) == 2.0:
            return torch.device("cuda")
    except Exception:
        return torch.device("cpu")
    return torch.device("cpu")
