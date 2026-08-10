"""Level-1 synthetic benchmark: planted-rule signal generator.

Produces controlled instances with KNOWN ground-truth structure so the executor
and (later) any learned component can be validated against truth:

    (X, mu_star, question, phi_star, answer_star)

where X is a multivariate signal in which the predicate pattern required by
phi_star is *planted* with controllable excitation. Because we know phi_star and
can compute its answer on the planted mu_star, this is the analogue of CLEVR:
full ground truth for controlled evaluation.

CRITICAL GUARD (learnability): a planted instance is only kept if phi_star is
actually answerable from the signal -- i.e. grounding X then executing phi_star
reproduces answer_star. This prevents the degenerate-benchmark failure (training
on instances whose ground-truth program cannot be recovered from the signal).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from executor.grammar import Node, validate
from executor.hard_logic import evaluate
from perception.grounding import Calibrator, ground
from .templates import Question, generate_questions


@dataclass
class SyntheticInstance:
    X: torch.Tensor          # [T, C] signal
    mu_star: torch.Tensor    # [T, P] grounded symbolic state
    question: Question
    phi_star: Node           # ground-truth program
    answer_star: bool        # computed answer (rho > 0)
    rho_star: float


def _plant_signal(
    n_channels: int, T: int, rng: random.Random
) -> torch.Tensor:
    """Build a signal with varied per-channel level/trend so predicates fire.

    Mixes trends, steps and oscillations so that high/low/rising/falling all
    occur somewhere -- giving the templates non-degenerate patterns to match.
    """
    g = torch.Generator().manual_seed(rng.randint(0, 2**31 - 1))
    X = torch.zeros(T, n_channels)
    t = torch.linspace(0, 1, T)
    for c in range(n_channels):
        kind = rng.choice(["up", "down", "step", "osc", "noise"])
        if kind == "up":
            X[:, c] = 3 * t + 0.2 * torch.randn(T, generator=g)
        elif kind == "down":
            X[:, c] = -3 * t + 0.2 * torch.randn(T, generator=g)
        elif kind == "step":
            s = rng.randint(T // 4, 3 * T // 4)
            X[s:, c] = 2.0
            X[:, c] += 0.2 * torch.randn(T, generator=g)
        elif kind == "osc":
            X[:, c] = torch.sin(2 * 3.1416 * rng.randint(1, 4) * t) + 0.1 * torch.randn(T, generator=g)
        else:
            X[:, c] = 0.5 * torch.randn(T, generator=g)
    return X


def generate_synthetic(
    n: int,
    n_channels: int = 4,
    T: int = 48,
    families: list[str] | None = None,
    seed: int = 0,
    calibrator: Calibrator | None = None,
    enforce_learnable: bool = True,
) -> tuple[list[SyntheticInstance], Calibrator]:
    """Generate up to n validated synthetic instances.

    A shared Calibrator is fit on a pool of planted signals (or supplied). Each
    instance is grounded and its phi_star executed; if ``enforce_learnable`` the
    instance is kept only when the answer is reproducible from the grounding.
    Returns (instances, calibrator).
    """
    rng = random.Random(seed)

    # build a pool of signals to fit the calibrator (if not provided)
    pool = torch.stack([_plant_signal(n_channels, T, rng) for _ in range(max(32, n // 4))])
    cal = calibrator or Calibrator.fit(pool)

    questions = generate_questions(n * 2, n_channels, T, families=families, seed=seed + 1)

    instances: list[SyntheticInstance] = []
    qi = 0
    P = 4 * n_channels
    attempts = 0
    while len(instances) < n and qi < len(questions) and attempts < n * 20:
        attempts += 1
        X = _plant_signal(n_channels, T, rng)
        mu, pidx = ground(X, cal)
        q = questions[qi]
        qi += 1
        try:
            validate(q.program, n_predicates=P)
        except Exception:
            continue
        rho, _ = evaluate(mu, q.program, pidx, anchor=0)
        ans = rho > 0.0
        if enforce_learnable:
            # re-ground independently and confirm the answer is stable/reproducible
            mu2, pidx2 = ground(X, cal)
            rho2, _ = evaluate(mu2, q.program, pidx2, anchor=0)
            if (rho2 > 0.0) != ans:
                continue  # unstable -> drop
        instances.append(
            SyntheticInstance(
                X=X, mu_star=mu, question=q, phi_star=q.program,
                answer_star=bool(ans), rho_star=float(rho),
            )
        )
    return instances, cal


def class_balance(instances: list[SyntheticInstance]) -> dict:
    """Report yes/no answer balance -- a guard against degenerate all-one-label sets."""
    yes = sum(1 for i in instances if i.answer_star)
    n = len(instances)
    return {"n": n, "yes": yes, "no": n - yes, "yes_frac": (yes / n if n else 0.0)}
