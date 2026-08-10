"""Grammar validation tests: invalid STL programs must be rejected."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

from executor.grammar import (Always, And, Eventually, Not, Or, Predicate,
                              Until, validate, STLSyntaxError)


def _expect_error(node, **kw):
    try:
        validate(node, **kw)
    except STLSyntaxError:
        return
    raise AssertionError(f"expected STLSyntaxError for {node!r}")


def test_valid_program_passes():
    validate(And(Predicate("high", 0), Eventually(0, 5, Predicate("rising", 1))),
             n_predicates=8)


def test_inverted_interval_rejected():
    _expect_error(Eventually(5, 2, Predicate("high", 0)))


def test_negative_bound_rejected():
    _expect_error(Always(-1, 3, Predicate("low", 0)))


def test_out_of_range_predicate_rejected():
    _expect_error(Predicate("high", 99), n_predicates=8)


def test_negative_channel_rejected():
    _expect_error(Predicate("high", -1))


def test_canonical_roundtrip_distinct():
    a = Eventually(0, 5, Predicate("high", 0))
    b = Always(0, 5, Predicate("high", 0))
    assert a.canonical() != b.canonical()
