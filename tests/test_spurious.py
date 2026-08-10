"""Invariants of the planted-spurious necessity benchmark.

These pin the properties the necessity experiment *depends on*. If any fail, the
headline shift result would be uninterpretable:

* answer_star is exactly executor(phi_star, mu_star)  -> no circularity, the
  label is the executor's verdict on the PLANTED truths.
* programs reference only causal channels             -> the symbolic path
  structurally cannot use the spurious channel.
* classes are balanced                                -> no constant-predictor wins.
* the spurious shortcut works in train/indist and breaks under shift -> the
  whole point of the dataset.
* indist and shift share identical causal channels    -> the indist->shift drop
  is attributable to the spurious channel and nothing else.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import functools
import random

import torch

from benchmark.spurious import (
    SpuriousInstance,
    balance_report,
    generate_spurious_benchmark,
    sample_program,
)
from executor.grammar import Always, And, Eventually, Not, Or, Predicate, Until
from executor.hard_logic import evaluate
from perception.grounding import predicate_index

DEPTHS = [1, 2, 3, 4]


@functools.lru_cache(maxsize=None)
def _small_bm(shift_mode="shift_decorr", seed=0):
    return generate_spurious_benchmark(
        n_train=200, n_test=120, n_causal=3, T=32,
        depths=DEPTHS, shift_mode=shift_mode, seed=seed,
    )


def _op_depth(node) -> int:
    """Operator-nesting depth: Predicate=0, one operator adds 1."""
    if isinstance(node, Predicate):
        return 0
    if isinstance(node, Not):
        return 1 + _op_depth(node.child)
    if isinstance(node, (Eventually, Always)):
        return 1 + _op_depth(node.child)
    if isinstance(node, (And, Or)):
        return 1 + max(_op_depth(node.left), _op_depth(node.right))
    if isinstance(node, Until):
        return 1 + max(_op_depth(node.left), _op_depth(node.right))
    raise TypeError(type(node).__name__)


def _channels(node) -> set[int]:
    """All channel indices referenced by the program's predicate leaves."""
    if isinstance(node, Predicate):
        return {node.channel}
    if isinstance(node, Not):
        return _channels(node.child)
    if isinstance(node, (Eventually, Always)):
        return _channels(node.child)
    if isinstance(node, (And, Or, Until)):
        return _channels(node.left) | _channels(node.right)
    raise TypeError(type(node).__name__)


# --------------------------------------------------------------------------- #
# program sampler
# --------------------------------------------------------------------------- #

def test_sample_program_has_exact_depth():
    rng = random.Random(0)
    for d in DEPTHS:
        for _ in range(20):
            phi = sample_program(d, [0, 1, 2], T=32, rng=rng)
            assert _op_depth(phi) == d, f"requested depth {d}, got {_op_depth(phi)}"


def test_sample_program_only_uses_causal_channels():
    rng = random.Random(1)
    causal = [0, 1, 2]
    for d in DEPTHS:
        for _ in range(20):
            phi = sample_program(d, causal, T=32, rng=rng)
            assert _channels(phi) <= set(causal)


# --------------------------------------------------------------------------- #
# core non-circularity invariant
# --------------------------------------------------------------------------- #

def test_answer_star_equals_executor_on_mu_star():
    bm = _small_bm()
    C = bm["meta"]["n_channels"]
    pidx = predicate_index(C)
    for split in ("train", "test_indist", "test_shift"):
        for inst in bm[split]:
            rho, _ = evaluate(inst.mu_star, inst.phi_star, pidx, anchor=0)
            assert (rho > 0.0) == inst.answer_star
            # rho on crisp planted truths is a +/-1 margin
            assert abs(abs(rho) - 1.0) < 1e-6


def test_programs_never_reference_spurious_channel():
    bm = _small_bm()
    for split in ("train", "test_indist", "test_shift"):
        for inst in bm[split]:
            assert inst.spurious_channel not in _channels(inst.phi_star)
            assert inst.spurious_channel == inst.n_causal


def test_mu_star_is_crisp_and_correctly_shaped():
    bm = _small_bm()
    T = bm["meta"]["T"]
    C = bm["meta"]["n_channels"]
    for inst in bm["train"][:50]:
        assert inst.mu_star.shape == (T, 4 * C)
        assert bool(((inst.mu_star == 0) | (inst.mu_star == 1)).all())
        assert inst.X.shape == (T, C)


# --------------------------------------------------------------------------- #
# balance
# --------------------------------------------------------------------------- #

def test_classes_are_balanced_overall_and_per_depth():
    bm = _small_bm()
    for split in ("train", "test_indist", "test_shift"):
        rep = balance_report(bm[split])
        assert 0.40 <= rep["yes_frac"] <= 0.60, (split, rep["yes_frac"])
        for d, info in rep["per_depth"].items():
            assert 0.35 <= info["yes_frac"] <= 0.65, (split, d, info)


# --------------------------------------------------------------------------- #
# the necessity property
# --------------------------------------------------------------------------- #

def test_spurious_shortcut_separates_train_but_not_shift_decorr():
    bm = _small_bm(shift_mode="shift_decorr")
    assert balance_report(bm["train"])["spurious_shortcut_acc"] == 1.0
    assert balance_report(bm["test_indist"])["spurious_shortcut_acc"] == 1.0
    sh = balance_report(bm["test_shift"])["spurious_shortcut_acc"]
    assert 0.30 <= sh <= 0.70, sh  # ~chance: shortcut carries no signal


def test_spurious_shortcut_inverts_under_shift_flip():
    bm = _small_bm(shift_mode="shift_flip")
    assert balance_report(bm["train"])["spurious_shortcut_acc"] == 1.0
    # flipped: a model that learned the shortcut is now exactly wrong
    assert balance_report(bm["test_shift"])["spurious_shortcut_acc"] == 0.0


def test_indist_and_shift_share_causal_channels_only_spurious_differs():
    bm = _small_bm()
    ti, ts = bm["test_indist"], bm["test_shift"]
    assert len(ti) == len(ts)
    for a, b in zip(ti, ts):
        nc = a.n_causal
        assert torch.allclose(a.X[:, :nc], b.X[:, :nc])       # causal identical
        assert not torch.allclose(a.X[:, nc], b.X[:, nc])     # spurious differs
        assert a.answer_star == b.answer_star                  # same rule/answer


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #

def test_generation_is_deterministic_given_seed():
    a = _small_bm(seed=7)
    b = _small_bm(seed=7)
    assert [i.answer_star for i in a["train"]] == [i.answer_star for i in b["train"]]
    assert torch.allclose(a["train"][0].X, b["train"][0].X)