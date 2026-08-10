"""Semi-synthetic necessity bridge: plant a spurious shortcut channel onto a REAL benchmark.

Takes a benchmark produced by ``realdata.build_real_benchmark`` (real signal windows,
causal program ``phi_star`` over the real channels, non-circular ``answer_star``) and
**appends one spurious channel** whose value encodes the answer:

  * train / test_indist : spurious channel correlated with the answer (shortcut works)
  * test_shift          : correlation broken -- decorrelated (``shift_decorr``) or
                          inverted (``shift_flip``)

The causal program is untouched and never references the spurious channel, so:

  * an end-to-end model is free to take the (depth-0, trivially easy) shortcut and
    therefore collapses on ``test_shift``;
  * the faithful NS-TQA path grounds every channel but the executor evaluates only the
    causal program, so it cannot use the spurious channel -- by construction.

This makes the crisp necessity claim on REAL signal backbones (the synthetic experiment
shows it in a fully controlled setting; this shows it survives real perception).

Non-circularity is preserved: ``answer_star`` is unchanged (still the executor on the
privileged real grounding); the spurious channel is added *after* the answer is fixed.

Layout note: predicate truths are family-major (``column = family_idx*C + channel``), so
adding a channel requires REINDEXING ``mu_star`` from ``4C`` to ``4(C+1)`` columns, not a
plain append. ``_reindex_with_spurious`` does this and writes the spurious channel's
predicate truths into the new trailing slots.
"""
from __future__ import annotations

import random

import torch

from perception.grounding import PRED_FAMILIES

from .adapters import RealInstance

NF = len(PRED_FAMILIES)                         # 4: high, low, rising, falling
FI = {name: i for i, name in enumerate(PRED_FAMILIES)}


def _reindex_with_spurious(mu_old: torch.Tensor, C: int, sval: float) -> torch.Tensor:
    """[T, 4C] (family-major over C) -> [T, 4(C+1)] with spurious channel at index C."""
    T = mu_old.shape[0]
    mu_new = torch.zeros(T, NF * (C + 1), dtype=mu_old.dtype)
    for fi in range(NF):
        mu_new[:, fi * (C + 1): fi * (C + 1) + C] = mu_old[:, fi * C: fi * C + C]
    sp = C  # spurious channel index
    mu_new[:, FI["high"] * (C + 1) + sp] = 1.0 if sval > 0 else 0.0
    mu_new[:, FI["low"] * (C + 1) + sp] = 1.0 if sval < 0 else 0.0
    mu_new[:, FI["rising"] * (C + 1) + sp] = 0.5     # constant channel -> no trend
    mu_new[:, FI["falling"] * (C + 1) + sp] = 0.5
    return mu_new


def attach_spurious_channel(bm: dict, shift_mode: str = "shift_decorr",
                            strength: float = 3.0, noise: float = 0.01,
                            seed: int = 0) -> dict:
    """Return a NEW benchmark dict with a planted spurious channel appended.

    ``shift_mode``: ``shift_decorr`` (spurious independent of answer under shift) or
    ``shift_flip`` (inverted under shift). ``strength`` sets the spurious level (well
    outside the normalized real signal range so it is a trivially easy shortcut).
    """
    if shift_mode not in ("shift_decorr", "shift_flip"):
        raise ValueError(f"shift_mode must be shift_decorr|shift_flip, got {shift_mode!r}")
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    C = bm["meta"]["n_channels"]

    def make(insts, regime, correlated):
        out = []
        for i in insts:
            ans = bool(i.answer_star)
            if correlated:
                sval = strength if ans else -strength
            elif shift_mode == "shift_decorr":
                sval = strength if rng.random() < 0.5 else -strength
            else:  # shift_flip
                sval = -strength if ans else strength
            T = i.X.shape[0]
            scol = torch.full((T, 1), float(sval)) + noise * torch.randn(T, 1, generator=g)
            Xp = torch.cat([i.X, scol], dim=1)               # [T, C+1]
            mu_p = _reindex_with_spurious(i.mu_star, C, sval)  # [T, 4(C+1)]
            out.append(RealInstance(
                X=Xp, question=i.question, phi_star=i.phi_star, answer_star=i.answer_star,
                depth=i.depth, unit_id=i.unit_id, n_channels=C + 1, condition=i.condition,
                regime=regime, mu_star=mu_p, rho_star=i.rho_star, spurious_channel=C,
                provenance={**(i.provenance or {}), "spurious_strength": strength,
                            "shift_mode": shift_mode},
            ))
        return out

    return {
        "train": make(bm["train"], "train", correlated=True),
        "test_indist": make(bm["test_indist"], "indist", correlated=True),
        "test_shift": make(bm["test_shift"], "shift", correlated=False),
        "meta": {**bm["meta"], "n_channels": C + 1, "spurious_channel": C,
                 "shift_mode": shift_mode, "spurious_strength": strength},
        "calibrator": bm.get("calibrator"),
        "normalizer": bm.get("normalizer"),
    }