"""Master NS-TQA pipeline: perception -> (structured query) -> executor.

In the no-text design there is no learned parser: a structured Question already
carries its STL program. This pipeline grounds the signal and executes the
program deterministically, returning answer + evidence. Learned perception
(regime) can be swapped in as an ablation.
"""
from __future__ import annotations
import torch
from executor.hard_logic import evaluate
from perception.grounding import Calibrator, ground


class NSTQA:
    def __init__(self, calibrator: Calibrator):
        self.cal = calibrator

    def answer(self, X: torch.Tensor, program) -> dict:
        """Ground X:[T,C], execute program, return answer + evidence."""
        mu, pidx = ground(X, self.cal)
        rho, tr = evaluate(mu, program, pidx, anchor=0)
        return {"answer": bool(rho > 0), "rho": float(rho),
                "critical_t": tr.critical_t}
