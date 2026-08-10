"""Answer-only perception training learns via the soft executor (no mu_star)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark.adapters import RealInstance
from benchmark.templates import Question
from executor.grammar import Eventually, Predicate
from executor.hard_logic import evaluate
from perception.grounding import predicate_index
from perception.learned import predict_mu, train_perception_answer_only


def test_answer_only_learns_through_soft_executor():
    """A net trained on ANSWER labels only (no mu_star) should answer correctly
    under the HARD executor on a separable synthetic task."""
    torch.manual_seed(0)
    C, T = 2, 12
    pidx = predicate_index(C)
    phi = Eventually(0, T - 1, Predicate("high", 0))
    insts = []
    for k in range(60):
        ans = (k % 2 == 0)
        X = torch.randn(T, C) * 0.1
        if ans:
            X[5:8, 0] += 3.0                      # a clear "high" bump only when answer is True
        insts.append(RealInstance(
            X=X, question=Question("t", phi, {}, None), phi_star=phi,
            answer_star=ans, depth=1, unit_id="u", n_channels=C, mu_star=None))

    res = train_perception_answer_only(insts, n_channels=C, epochs=40,
                                       batch_size=16, seed=0, verbose=False)
    correct = 0
    for i in insts:
        mu = predict_mu(res.model, i.X)            # HARD executor at inference
        rho, _ = evaluate(mu, i.phi_star, pidx, 0)
        correct += int((rho > 0) == i.answer_star)
    assert correct / len(insts) > 0.8             # learned the task from answers alone
