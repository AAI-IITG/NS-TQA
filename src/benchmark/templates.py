"""Structured temporal question templates (no natural language required).

A "question" here is a structured object: a template name plus channel/interval
bindings. Each template instantiates directly to (1) an executable STL program
phi and (2) an optional templated English rendering for display. The answer is
obtained by executing phi on the symbolic state -- never authored by hand.

This is the no-text design: questions are generated, programs are their meaning,
answers are computed. A compositional family of templates is what justifies the
symbolic approach (end-to-end models should struggle as composition deepens).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

from executor.grammar import (
    Always,
    And,
    Eventually,
    Node,
    Not,
    Or,
    Predicate,
    Until,
)

LEVEL = ["high", "low"]
TREND = ["rising", "falling"]


@dataclass
class Question:
    template: str            # family name
    program: Node            # ground-truth STL program phi*
    bindings: dict           # the slot values used
    text: Optional[str] = None  # templated rendering (display only)


# Each builder takes (channels, T, rng) and returns a Question.
# Channels are integer indices; T is the window length.

def _rand_interval(T: int, rng: random.Random, max_len: int | None = None) -> tuple[int, int]:
    max_len = max_len or T
    a = rng.randint(0, max(0, T - 2))
    b = min(T - 1, a + rng.randint(1, max_len))
    return a, b


def q_existence(channels, T, rng) -> Question:
    """Does <pred>(c) ever hold in [a,b]?  ->  Eventually_[a,b] pred(c)."""
    c = rng.choice(channels)
    pred = rng.choice(LEVEL + TREND)
    a, b = _rand_interval(T, rng)
    phi = Eventually(a, b, Predicate(pred, c))
    txt = f"does {pred}(ch{c}) ever hold within [{a},{b}]?"
    return Question("existence", phi, {"c": c, "pred": pred, "a": a, "b": b}, txt)


def q_persistence(channels, T, rng) -> Question:
    """Does <pred>(c) hold throughout [a,b]?  ->  Always_[a,b] pred(c)."""
    c = rng.choice(channels)
    pred = rng.choice(LEVEL)
    a, b = _rand_interval(T, rng, max_len=max(2, T // 2))
    phi = Always(a, b, Predicate(pred, c))
    txt = f"does {pred}(ch{c}) hold throughout [{a},{b}]?"
    return Question("persistence", phi, {"c": c, "pred": pred, "a": a, "b": b}, txt)


def q_temporal_order(channels, T, rng) -> Question:
    """Does <p1>(c1) hold until <p2>(c2)?  ->  p1(c1) U_[a,b] p2(c2)  (before-ness)."""
    c1, c2 = rng.sample(channels, 2) if len(channels) >= 2 else (channels[0], channels[0])
    p1 = rng.choice(TREND + LEVEL)
    p2 = rng.choice(TREND + LEVEL)
    a, b = _rand_interval(T, rng)
    phi = Until(a, b, Predicate(p1, c1), Predicate(p2, c2))
    txt = f"does {p1}(ch{c1}) persist until {p2}(ch{c2}) within [{a},{b}]?"
    return Question(
        "temporal_order", phi, {"c1": c1, "c2": c2, "p1": p1, "p2": p2, "a": a, "b": b}, txt
    )


def q_conjunction(channels, T, rng) -> Question:
    """Compositional: <pred persists> AND <other eventually> -- depth-2 pattern."""
    c1, c2 = rng.sample(channels, 2) if len(channels) >= 2 else (channels[0], channels[0])
    p1 = rng.choice(LEVEL)
    p2 = rng.choice(TREND)
    a1, b1 = _rand_interval(T, rng, max_len=max(2, T // 3))
    a2, b2 = _rand_interval(T, rng)
    phi = And(
        Always(a1, b1, Predicate(p1, c1)),
        Eventually(a2, b2, Predicate(p2, c2)),
    )
    txt = (
        f"does {p1}(ch{c1}) hold throughout [{a1},{b1}] "
        f"and {p2}(ch{c2}) occur within [{a2},{b2}]?"
    )
    return Question("conjunction", phi, {"c1": c1, "c2": c2, "p1": p1, "p2": p2}, txt)


TEMPLATES: dict[str, Callable] = {
    "existence": q_existence,
    "persistence": q_persistence,
    "temporal_order": q_temporal_order,
    "conjunction": q_conjunction,
}


def sample_question(
    n_channels: int,
    T: int,
    families: list[str] | None = None,
    seed: int | None = None,
) -> Question:
    """Sample one structured question over the given channels/window length."""
    rng = random.Random(seed)
    families = families or list(TEMPLATES.keys())
    fam = rng.choice(families)
    channels = list(range(n_channels))
    return TEMPLATES[fam](channels, T, rng)


def generate_questions(
    n: int,
    n_channels: int,
    T: int,
    families: list[str] | None = None,
    seed: int = 0,
) -> list[Question]:
    """Generate n structured questions (deterministic given seed)."""
    rng = random.Random(seed)
    families = families or list(TEMPLATES.keys())
    channels = list(range(n_channels))
    out = []
    for _ in range(n):
        fam = rng.choice(families)
        out.append(TEMPLATES[fam](channels, T, rng))
    return out
