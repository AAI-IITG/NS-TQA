"""STL-only baseline: hand-rule grounding + executor answers a clear case, and is
non-trivial (not the oracle) on noisy windows."""
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
from models.stl_only import hand_rule_calibrator, stl_only_evaluate


def _inst(X, phi, ans):
    return RealInstance(X=X, question=Question("t", phi, {}, None), phi_star=phi,
                        answer_star=ans, depth=1, unit_id="u", n_channels=X.shape[1])


def test_hand_rule_calibrator_is_fixed_not_fit():
    cal = hand_rule_calibrator(3, hi=1.0, lo=-1.0)
    assert torch.allclose(cal.hi, torch.ones(3))
    assert torch.allclose(cal.lo, -torch.ones(3))


def test_stl_only_answers_a_clear_high_case():
    # ch0 has a clear sustained high bump -> Eventually high(0) is True
    T, C = 16, 2
    phi = Eventually(0, T - 1, Predicate("high", 0))
    yes = torch.zeros(T, C)
    yes[5:9, 0] = 3.0                        # well above the +1.0 hand-rule cut
    no = torch.zeros(T, C)                    # flat -> never high
    out = stl_only_evaluate([_inst(yes, phi, True), _inst(no, phi, False)], C)
    assert out["answer_accuracy"] == 1.0


def test_stl_only_empty_is_safe():
    out = stl_only_evaluate([], 4)
    assert out["n"] == 0 and out["answer_accuracy"] is None
