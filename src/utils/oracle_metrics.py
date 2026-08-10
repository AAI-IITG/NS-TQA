"""Metrics for the faithful symbolic oracle evaluation path."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def accuracy(pred: list[bool], gold: list[bool]) -> float:
    """Boolean accuracy with empty-list protection."""
    return sum(p == g for p, g in zip(pred, gold)) / max(1, len(gold))


def binary_metrics(pred: list[bool], gold: list[bool]) -> dict:
    """Standard binary metrics for yes/no QA answers."""
    tp = sum(p and g for p, g in zip(pred, gold))
    tn = sum((not p) and (not g) for p, g in zip(pred, gold))
    fp = sum(p and (not g) for p, g in zip(pred, gold))
    fn = sum((not p) and g for p, g in zip(pred, gold))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "majority_no_accuracy": sum(not g for g in gold) / max(1, len(gold)),
    }


def group_accuracy(keys: list[Any], pred: list[bool], gold: list[bool]) -> dict:
    """Answer accuracy grouped by metadata key."""
    groups = defaultdict(lambda: {"correct": 0, "n": 0})
    for key, p, g in zip(keys, pred, gold):
        if key is None:
            continue
        groups[str(key)]["correct"] += int(p == g)
        groups[str(key)]["n"] += 1
    return {
        key: {"accuracy": val["correct"] / max(1, val["n"]), "n": val["n"]}
        for key, val in sorted(groups.items())
    }


def critical_within(
    pred_t: int | None,
    gold_t: int | None,
    tolerance: int = 2,
) -> bool:
    """Whether two critical timesteps match within tolerance."""
    if pred_t is None or gold_t is None:
        return pred_t == gold_t
    return abs(int(pred_t) - int(gold_t)) <= tolerance


def assemble_oracle_metrics(
    records: list[dict],
    dataset_path: str | None = None,
    dataset_tag: str | None = None,
) -> dict:
    """Assemble symbolic oracle metrics from per-instance prediction records."""
    pred_answers = [bool(r["pred_answer"]) for r in records]
    gold_answers = [bool(r["gold_answer"]) for r in records]
    rho_sign = [
        bool(r["pred_rho"] > 0) == bool(r["gold_rho"] > 0)
        for r in records
    ]
    critical_exact = [
        r.get("pred_critical_t") == r.get("gold_critical_t")
        for r in records
    ]
    critical_close = [
        critical_within(r.get("pred_critical_t"), r.get("gold_critical_t"), tolerance=2)
        for r in records
    ]
    conjunction = [
        p == g and c
        for p, g, c in zip(pred_answers, gold_answers, critical_exact)
    ]

    metrics = {
        "dataset_path": dataset_path,
        "dataset_tag": dataset_tag,
        "split": "all",
        "n": len(records),
        "answer_accuracy": accuracy(pred_answers, gold_answers),
        "rho_sign_accuracy": sum(rho_sign) / max(1, len(rho_sign)),
        "critical_t_exact": sum(critical_exact) / max(1, len(critical_exact)),
        "critical_t_within_2": sum(critical_close) / max(1, len(critical_close)),
        "program_exact_match": 1.0,
        "conjunction_score": sum(conjunction) / max(1, len(conjunction)),
        "gold_yes_frac": sum(gold_answers) / max(1, len(gold_answers)),
        "pred_yes_frac": sum(pred_answers) / max(1, len(pred_answers)),
        "binary": binary_metrics(pred_answers, gold_answers),
        "by_template": group_accuracy(
            [r.get("template") for r in records], pred_answers, gold_answers
        ),
        "by_engine_id": group_accuracy(
            [r.get("engine_id") for r in records], pred_answers, gold_answers
        ),
        "by_bearing_id": group_accuracy(
            [r.get("bearing_id") for r in records], pred_answers, gold_answers
        ),
    }
    return metrics
