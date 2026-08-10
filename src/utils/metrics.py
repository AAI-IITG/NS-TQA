"""Evaluation metrics for temporal QA.

The headline metric is the CONJUNCTION of answer correctness, program
correctness, and explanation correctness -- because a faithful system must be
right for the right reason (the "right answer, wrong program" failure is exactly
what this approach exists to catch).
"""
from __future__ import annotations

from executor.grammar import Node


def answer_accuracy(pred: list[bool], gold: list[bool]) -> float:
    """Fraction of correct boolean answers."""
    if not gold:
        return 0.0
    return sum(p == g for p, g in zip(pred, gold)) / len(gold)


def program_exact_match(pred: list[Node], gold: list[Node]) -> float:
    """Fraction where the predicted program's canonical form equals the gold's."""
    if not gold:
        return 0.0
    return sum(p.canonical() == g.canonical() for p, g in zip(pred, gold)) / len(gold)


def operator_accuracy(pred: list[Node], gold: list[Node]) -> float:
    """Token-level overlap of operator/predicate multisets (partial credit)."""
    if not gold:
        return 0.0

    def tokens(n: Node) -> list[str]:
        toks = [type(n).__name__]
        for c in n._children():
            toks += tokens(c)
        return toks

    scores = []
    for p, g in zip(pred, gold):
        tp, tg = tokens(p), tokens(g)
        # Jaccard on token multisets
        from collections import Counter
        cp, cg = Counter(tp), Counter(tg)
        inter = sum((cp & cg).values())
        union = sum((cp | cg).values())
        scores.append(inter / union if union else 0.0)
    return sum(scores) / len(scores)


def interval_overlap(pred_tau: list[int | None], gold_tau: list[int | None], tol: int = 2) -> float:
    """Fraction where predicted critical timestep is within tol of the gold."""
    if not gold_tau:
        return 0.0
    ok = 0
    for p, g in zip(pred_tau, gold_tau):
        if p is None or g is None:
            continue
        if abs(p - g) <= tol:
            ok += 1
    return ok / len(gold_tau)


def conjunction_score(
    ans_ok: list[bool], prog_ok: list[bool], expl_ok: list[bool]
) -> float:
    """Fraction of instances correct on ALL THREE: answer AND program AND explanation."""
    n = len(ans_ok)
    if n == 0:
        return 0.0
    return sum(a and p and e for a, p, e in zip(ans_ok, prog_ok, expl_ok)) / n


def report(
    pred_ans: list[bool],
    gold_ans: list[bool],
    pred_prog: list[Node] | None = None,
    gold_prog: list[Node] | None = None,
    pred_tau: list | None = None,
    gold_tau: list | None = None,
) -> dict:
    """Assemble a metrics dictionary; program/explanation metrics if provided."""
    out = {"answer_accuracy": answer_accuracy(pred_ans, gold_ans)}
    if pred_prog is not None and gold_prog is not None:
        out["program_exact_match"] = program_exact_match(pred_prog, gold_prog)
        out["operator_accuracy"] = operator_accuracy(pred_prog, gold_prog)
    if pred_tau is not None and gold_tau is not None:
        out["interval_overlap"] = interval_overlap(pred_tau, gold_tau)
    return out
