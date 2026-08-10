"""Faithful NS-TQA with LEARNED perception (drop-in for ``models.nstqa.NSTQA``).

Same ``answer(X, program) -> {answer, rho, critical_t}`` contract as the
calibrated ``NSTQA``, but perception is a trained ``LearnedPerception`` net
instead of fixed-quantile grounding. The executor is unchanged: the
deterministic, parameter-free ``hard_logic``. The ONLY difference from the
calibrated pipeline is how ``mu`` is produced -- so this class is exactly the
component that turns the project from "symbolic + a separate baseline" into a
genuine neuro-symbolic system whose neural part is load-bearing.

The answer is still obtainable ONLY by executing the program over the perceived
symbolic state, so a correct answer entails a correct program AND adequate
perception. ``explain`` exposes the perceived state, the robustness signal, and
the critical timestep for qualitative/faithfulness figures. ``evaluate_instances``
produces per-instance records (predicted vs planted-oracle) ready for the
existing ``utils.oracle_metrics`` assembler.
"""
from __future__ import annotations

from typing import Any

import torch

from executor.hard_logic import evaluate
from perception.grounding import predicate_index
from perception.learned import LearnedPerception, predict_mu


def _get(instance: Any, key: str, default: Any = None) -> Any:
    if isinstance(instance, dict):
        return instance.get(key, default)
    return getattr(instance, key, default)


class LearnedNSTQA:
    """Learned perception -> deterministic STL executor."""

    def __init__(self, model: LearnedPerception, n_channels: int):
        self.model = model
        self.n_channels = n_channels
        self.pidx = predicate_index(n_channels)

    @torch.no_grad()
    def perceive(self, X: torch.Tensor) -> torch.Tensor:
        """X:[T,C] -> perceived symbolic state mu_hat:[T,P]."""
        return predict_mu(self.model, X)

    def answer(self, X: torch.Tensor, program) -> dict:
        """Ground X via learned perception, execute program, return answer + evidence.

        Identical return contract to ``models.nstqa.NSTQA.answer``.
        """
        mu = self.perceive(X)
        rho, tr = evaluate(mu, program, self.pidx, anchor=0)
        return {"answer": bool(rho > 0), "rho": float(rho), "critical_t": tr.critical_t}

    def explain(self, X: torch.Tensor, program) -> dict:
        """Richer trace for qualitative/faithfulness figures.

        Returns the answer, robustness, critical timestep, the full robustness
        signal over time, the perceived symbolic state, and the program string.
        """
        mu = self.perceive(X)
        rho, tr = evaluate(mu, program, self.pidx, anchor=0)
        return {
            "answer": bool(rho > 0),
            "rho": float(rho),
            "critical_t": tr.critical_t,
            "rho_signal": tr.rho,          # [T]
            "mu_hat": mu,                  # [T, P] perceived truths
            "program": program.canonical(),
        }

    def evaluate_instances(self, instances: list) -> dict:
        """Run the faithful path over instances; return accuracy + records.

        Each record carries predicted answer/robustness/critical-step alongside
        the PLANTED-oracle answer/robustness/critical-step (computed from
        ``mu_star``), so the symbolic path's real error is separated from the
        oracle upper bound. Record keys are compatible with
        ``utils.oracle_metrics.assemble_oracle_metrics``.
        """
        records = []
        correct = 0
        for inst in instances:
            X = _get(inst, "X")
            phi = _get(inst, "phi_star")
            mu_star = _get(inst, "mu_star")
            gold_answer = bool(_get(inst, "answer_star"))

            mu_hat = self.perceive(X)
            rho, tr = evaluate(mu_hat, phi, self.pidx, anchor=0)
            pred_answer = bool(rho > 0.0)
            correct += int(pred_answer == gold_answer)

            # planted-oracle reference (upper bound)
            if mu_star is not None:
                gold_rho, gold_tr = evaluate(mu_star, phi, self.pidx, anchor=0)
                gold_rho_v = float(gold_rho)
                gold_crit = gold_tr.critical_t
            else:
                gold_rho_v = 1.0 if gold_answer else -1.0
                gold_crit = None

            records.append({
                "pred_answer": pred_answer,
                "gold_answer": gold_answer,
                "pred_rho": float(rho),
                "gold_rho": gold_rho_v,
                "pred_critical_t": tr.critical_t,
                "gold_critical_t": gold_crit,
                "template": _get(inst, "question").template
                if _get(inst, "question") is not None else None,
                "depth": _get(inst, "depth"),
                "regime": _get(inst, "regime"),
            })
        return {"answer_accuracy": correct / max(1, len(instances)), "records": records}