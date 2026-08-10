"""Unit tests for explanation-faithfulness blame assignment + metrics."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from executor.grammar import Always, And, Eventually, Predicate
from executor.hard_logic import evaluate
from perception.grounding import predicate_index
from utils.faithfulness import (_counterfactual_flips, _iou, governing_leaf,
                                leakage_probe)

C, T = 2, 5
PIDX = predicate_index(C)            # high(0)=0, high(1)=1, low(0)=2, ...


def _mu_zeros():
    return torch.full((T, 4 * C), 0.5)   # neutral truths everywhere


def test_governing_leaf_eventually_picks_argmax_step():
    mu = _mu_zeros()
    mu[3, PIDX[("high", 0)]] = 1.0       # high(ch0) true only at t=3
    phi = Eventually(0, 4, Predicate("high", 0))
    leaf, t = governing_leaf(mu, phi, PIDX, 0)
    assert (leaf.name, leaf.channel) == ("high", 0)
    assert t == 3                        # decisive timestep is the satisfying one
    assert evaluate(mu, phi, PIDX, 0)[0] > 0


def test_governing_leaf_and_picks_binding_branch():
    mu = _mu_zeros()
    mu[0, PIDX[("high", 0)]] = 1.0        # high(0) strongly true
    mu[0, PIDX[("low", 1)]] = 0.0         # low(1) strongly false  -> binding (min)
    phi = And(Predicate("high", 0), Predicate("low", 1))
    leaf, t = governing_leaf(mu, phi, PIDX, 0)
    assert (leaf.name, leaf.channel) == ("low", 1)   # the smaller-robustness branch
    assert t == 0


def test_counterfactual_flips_the_answer():
    mu = _mu_zeros()
    mu[2, PIDX[("high", 0)]] = 1.0
    phi = Eventually(0, 4, Predicate("high", 0))
    leaf, t = governing_leaf(mu, phi, PIDX, 0)
    assert evaluate(mu, phi, PIDX, 0)[0] > 0          # currently satisfied
    assert _counterfactual_flips(mu, phi, PIDX, leaf, t) is True


def test_iou_helper():
    assert _iou((1, 3), (1, 3)) == 1.0
    assert _iou((0, 2), (2, 4)) == 1 / 5             # overlap {2}, union {0..4}
    assert _iou(None, (0, 1)) is None


def test_leakage_probe_runs_and_is_bounded():
    # two trivial depth-1 instances; probe returns a value in [0,1]
    class _Inst:
        def __init__(self, mu, ans, depth):
            self.mu_star, self.answer_star, self.depth = mu, ans, depth
    a = _mu_zeros(); a[0, PIDX[("high", 0)]] = 1.0
    b = _mu_zeros(); b[0, PIDX[("high", 0)]] = 0.0
    probe = leakage_probe([_Inst(a, True, 1), _Inst(b, False, 1)])
    assert 1 in probe and 0.0 <= probe[1] <= 1.0
