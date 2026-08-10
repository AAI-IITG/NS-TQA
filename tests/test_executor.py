"""Executor correctness tests: feed mock mu states, check exact robustness.

These pin the deterministic semantics against hand-computed values. If any of
these fail, the symbolic layer is untrustworthy and nothing downstream is valid.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from executor.grammar import (
    Always,
    And,
    Eventually,
    Not,
    Or,
    Predicate,
    Until,
    validate,
    STLSyntaxError,
)
from executor.hard_logic import evaluate, robustness, satisfied


def _mu(cols):
    """Build a [T, P] mu from a list of per-predicate time series."""
    return torch.tensor(cols, dtype=torch.float32).T


def test_predicate_leaf():
    # one predicate, truth rising from 0 to 1
    mu = _mu([[0.0, 0.5, 1.0]])  # T=3, P=1
    pidx = {("high", 0): 0}
    rho = robustness(mu, Predicate("high", 0), pidx)
    # signed margin = 2*mu - 1
    assert torch.allclose(rho, torch.tensor([-1.0, 0.0, 1.0]))


def test_not():
    mu = _mu([[0.0, 1.0]])
    pidx = {("high", 0): 0}
    rho = robustness(mu, Not(Predicate("high", 0)), pidx)
    assert torch.allclose(rho, torch.tensor([1.0, -1.0]))


def test_and_or():
    mu = _mu([[1.0, 0.0], [0.0, 1.0]])  # T=2, P=2
    pidx = {("a", 0): 0, ("b", 0): 1}
    a, b = Predicate("a", 0), Predicate("b", 0)
    # at t=0: a=+1, b=-1 ; and=min=-1, or=max=+1
    rho_and = robustness(mu, And(a, b), pidx)
    rho_or = robustness(mu, Or(a, b), pidx)
    assert torch.allclose(rho_and, torch.tensor([-1.0, -1.0]))
    assert torch.allclose(rho_or, torch.tensor([1.0, 1.0]))


def test_eventually():
    # truth low everywhere except a spike at t=2
    mu = _mu([[0.0, 0.0, 1.0, 0.0]])  # T=4
    pidx = {("high", 0): 0}
    # Eventually_[0,3] high : at anchor 0, max over [0,3] of (2mu-1) = +1
    rho, tr = evaluate(mu, Eventually(0, 3, Predicate("high", 0)), pidx, anchor=0)
    assert rho == 1.0
    assert tr.critical_t == 2  # the spike


def test_always():
    mu = _mu([[1.0, 1.0, 0.0, 1.0]])  # dips at t=2
    pidx = {("high", 0): 0}
    # Always_[0,3] high : min over window = at t=2, 2*0-1 = -1
    rho, tr = evaluate(mu, Always(0, 3, Predicate("high", 0)), pidx, anchor=0)
    assert rho == -1.0
    assert tr.critical_t == 2


def test_until():
    # phi1 = high(0) holds early, phi2 = high(1) holds at t=2
    mu = _mu([[1.0, 1.0, 1.0, 0.0],   # high(0)
              [0.0, 0.0, 1.0, 0.0]])  # high(1)
    pidx = {("high", 0): 0, ("high", 1): 1}
    phi = Until(0, 3, Predicate("high", 0), Predicate("high", 1))
    rho, _ = evaluate(mu, phi, pidx, anchor=0)
    # at tp=2: phi2=+1, phi1 holds on [0,2] (all +1) -> min=+1 -> val=+1
    assert rho == 1.0


def test_nested_before_pattern():
    # "high(0) stays high for [0,1] AND eventually rising(1)" -- a before-ish pattern
    mu = _mu([[1.0, 1.0, 0.0, 0.0],   # high(0): high early
              [0.0, 0.0, 1.0, 1.0]])  # rising(1): later
    pidx = {("high", 0): 0, ("rising", 1): 1}
    phi = And(
        Always(0, 1, Predicate("high", 0)),
        Eventually(0, 3, Predicate("rising", 1)),
    )
    rho, _ = evaluate(mu, phi, pidx, anchor=0)
    # Always_[0,1] high(0): min over t in [0,1] = +1 ; Eventually rising(1): +1 ; and=+1
    assert rho == 1.0


def test_satisfied_boolean():
    mu = _mu([[1.0]])
    pidx = {("high", 0): 0}
    assert satisfied(mu, Predicate("high", 0), pidx) is True
    assert satisfied(mu, Not(Predicate("high", 0)), pidx) is False


def test_validate_rejects_inverted_interval():
    try:
        validate(Eventually(5, 2, Predicate("high", 0)))
    except STLSyntaxError:
        return
    raise AssertionError("expected STLSyntaxError for inverted interval")


def test_validate_rejects_out_of_range_predicate():
    try:
        validate(Predicate("high", 7), n_predicates=4)
    except STLSyntaxError:
        return
    raise AssertionError("expected STLSyntaxError for out-of-range predicate")


def test_predicates_collection():
    phi = And(Predicate("high", 0), Eventually(0, 2, Predicate("rising", 1)))
    assert phi.predicates() == {"high(0)", "rising(1)"}
