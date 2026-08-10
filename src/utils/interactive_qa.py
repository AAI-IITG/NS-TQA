"""Notebook helpers for interactive QA artifact inspection.

These helpers keep notebooks focused on exploration while preserving the core
NS-TQA contract: oracle answers are produced by grounding with the saved
calibrator and executing ``phi_star`` through ``NSTQA``.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch

from benchmark.baseline import encode_question, load_qa_pickle, query_dim, select_device
from models.baselines import LSTMBaseline
from models.nstqa import NSTQA


def dataset_tag(path: str | Path) -> str:
    """Return the stable dataset tag used in run directories."""
    name = Path(path).stem
    return name[:-3] if name.endswith("_qa") else name


def load_artifact(path: str | Path) -> dict:
    """Load a QA pickle artifact."""
    return load_qa_pickle(path)


def instances(artifact: dict) -> list[dict]:
    """Return the artifact's QA instances."""
    return artifact["instances"]


def instance_value(instance: Any, key: str, default: Any = None) -> Any:
    """Read a dict or dataclass-style QA field."""
    if isinstance(instance, dict):
        return instance.get(key, default)
    return getattr(instance, key, default)


def artifact_summary(artifact: dict) -> dict:
    """Compact summary for display in notebooks."""
    rows = instances(artifact)
    first = rows[0] if rows else {}
    X = instance_value(first, "X")
    templates = Counter(instance_value(x, "question").template for x in rows)
    engine_ids = sorted(
        {str(instance_value(x, "engine_id")) for x in rows if instance_value(x, "engine_id") is not None}
    )
    bearing_ids = sorted(
        {str(instance_value(x, "bearing_id")) for x in rows if instance_value(x, "bearing_id") is not None}
    )
    return {
        "n_instances": len(rows),
        "window_shape": tuple(X.shape) if X is not None else None,
        "yes_frac": sum(bool(instance_value(x, "answer_star")) for x in rows) / max(1, len(rows)),
        "templates": dict(sorted(templates.items())),
        "n_engines": len(engine_ids),
        "n_bearings": len(bearing_ids),
        "engine_ids_sample": engine_ids[:10],
        "bearing_ids_sample": bearing_ids[:10],
        "channel_names": artifact.get("channel_names"),
    }


def select_indices(
    artifact: dict,
    template: str | None = None,
    answer: bool | None = None,
    engine_id: str | int | None = None,
    bearing_id: str | None = None,
    limit: int = 20,
) -> list[int]:
    """Find instance indices matching optional metadata filters."""
    out = []
    for i, row in enumerate(instances(artifact)):
        question = instance_value(row, "question")
        if template is not None and question.template != template:
            continue
        if answer is not None and bool(instance_value(row, "answer_star")) != bool(answer):
            continue
        if engine_id is not None and str(instance_value(row, "engine_id")) != str(engine_id):
            continue
        if bearing_id is not None and str(instance_value(row, "bearing_id")) != str(bearing_id):
            continue
        out.append(i)
        if len(out) >= limit:
            break
    return out


def describe_instance(instance: dict) -> dict:
    """Human-readable fields for one QA instance."""
    question = instance_value(instance, "question")
    return {
        "template": question.template,
        "question_text": question.text,
        "bindings": question.bindings,
        "program": repr(instance_value(instance, "phi_star")),
        "answer_star": bool(instance_value(instance, "answer_star")),
        "rho_star": float(instance_value(instance, "rho_star")),
        "critical_t": instance_value(instance, "critical_t"),
        "rul": instance_value(instance, "rul"),
        "engine_id": instance_value(instance, "engine_id"),
        "bearing_id": instance_value(instance, "bearing_id"),
        "X_shape": tuple(instance_value(instance, "X").shape),
    }


def answer_symbolic(artifact: dict, index: int) -> dict:
    """Execute the faithful symbolic oracle for one QA instance."""
    row = instances(artifact)[index]
    oracle = NSTQA(artifact["calibrator"])
    pred = oracle.answer(instance_value(row, "X"), instance_value(row, "phi_star"))
    return {
        "index": index,
        "pred_answer": bool(pred["answer"]),
        "gold_answer": bool(instance_value(row, "answer_star")),
        "pred_rho": float(pred["rho"]),
        "gold_rho": float(instance_value(row, "rho_star")),
        "pred_critical_t": pred["critical_t"],
        "gold_critical_t": instance_value(row, "critical_t"),
        "answer_correct": bool(pred["answer"]) == bool(instance_value(row, "answer_star")),
        "rho_sign_correct": bool(pred["rho"] > 0)
        == bool(float(instance_value(row, "rho_star")) > 0),
        "critical_t_exact": pred["critical_t"] == instance_value(row, "critical_t"),
    }


def load_baseline(checkpoint_path: str | Path, device: str = "cpu") -> tuple[LSTMBaseline, dict, torch.device]:
    """Load a trained direct-answer LSTM baseline checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    dev = select_device(device)
    model = LSTMBaseline(
        n_channels=ckpt["n_channels"],
        query_dim=query_dim(),
        hidden=ckpt["hidden"],
        dropout=ckpt.get("dropout", 0.0),
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt, dev


def answer_baseline(
    artifact: dict,
    index: int,
    model: LSTMBaseline,
    ckpt: dict,
    device: torch.device,
) -> dict:
    """Predict one direct-answer baseline result for one QA instance."""
    row = instances(artifact)[index]
    X = instance_value(row, "X").unsqueeze(0).to(device)
    q = encode_question(
        instance_value(row, "question"),
        n_channels=ckpt["n_channels"],
        T=ckpt["T"],
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = float(model(X, q).item())
    prob = float(torch.sigmoid(torch.tensor(logit)).item())
    pred = prob >= 0.5
    gold = bool(instance_value(row, "answer_star"))
    return {
        "index": index,
        "prob_yes": prob,
        "pred_answer": pred,
        "gold_answer": gold,
        "answer_correct": pred == gold,
        "checkpoint_dataset": ckpt.get("dataset_path"),
        "split_meta": ckpt.get("split_meta") or {"split_mode": "random"},
    }


def plot_window(instance: dict, channel_names: list[str] | None = None, channels: list[int] | None = None):
    """Plot selected channels from one instance window."""
    import matplotlib.pyplot as plt

    X = instance_value(instance, "X").detach().cpu()
    T, C = X.shape
    if channels is None:
        channels = list(range(min(C, 6)))
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in channels:
        label = channel_names[c] if channel_names and c < len(channel_names) else f"ch{c}"
        ax.plot(range(T), X[:, c].numpy(), label=label)
    critical_t = instance_value(instance, "critical_t")
    if critical_t is not None:
        ax.axvline(int(critical_t), color="black", linestyle="--", linewidth=1, label="critical_t")
    ax.set_xlabel("timestep in window")
    ax.set_ylabel("value")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    return fig, ax
