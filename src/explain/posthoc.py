"""Post-hoc attribution methods for the by-construction-vs-post-hoc study (WO-2B).

Each method takes a black-box baseline (via a ``logit_fn: X[B,T,C] -> logit[B]``
closure, so it is model-agnostic) and a single window ``X[T,C]`` and returns a
saliency map ``[T,C]`` (non-negative magnitude). ``src/explain/extract.py`` then
turns a saliency map into the SAME explanation tuple NS-TQA emits (critical interval
+ supporting channel), so both are scored on identical faithfulness axes.

Methods:
  * ``gradient_input``       -- |∂logit/∂X ⊙ X| (a standard, cheap attribution).
  * ``integrated_gradients`` -- path integral from a zero baseline (Sundararajan
                                et al.); native torch, no dependency.
  * ``shap_gradient``        -- SHAP GradientExplainer (optional; needs ``shap``;
                                returns None if unavailable -- reported as TBD).
  * ``attention_rollout``    -- best-effort attention flow for a Transformer baseline
                                (needs per-layer attention; returns None if the model
                                does not expose it).
"""
from __future__ import annotations

from typing import Callable, Optional

import torch


def gradient_input(logit_fn: Callable, X: torch.Tensor) -> torch.Tensor:
    """|grad(logit, X) ⊙ X| -> [T,C]."""
    Xb = X.unsqueeze(0).clone().requires_grad_(True)
    logit = logit_fn(Xb)
    g, = torch.autograd.grad(logit.sum(), Xb)
    return (g * Xb).abs().detach()[0]


def integrated_gradients(logit_fn: Callable, X: torch.Tensor, steps: int = 32,
                         baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Integrated Gradients from ``baseline`` (default zeros) to ``X`` -> [T,C].

    IG(x) = (x - x') ⊙ ∫_0^1 ∂logit(x' + α(x-x'))/∂x dα, approximated with a
    ``steps``-point Riemann sum. Returns the per-(t,c) attribution magnitude."""
    base = torch.zeros_like(X) if baseline is None else baseline
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=X.device)
    grads = torch.zeros_like(X)
    for a in alphas:
        xi = (base + a * (X - base)).unsqueeze(0).clone().requires_grad_(True)
        logit = logit_fn(xi)
        g, = torch.autograd.grad(logit.sum(), xi)
        grads += g.detach()[0]
    avg = grads / steps
    return ((X - base) * avg).abs()


def shap_gradient(model, query: torch.Tensor, X: torch.Tensor, background: torch.Tensor,
                  device) -> Optional[torch.Tensor]:
    """SHAP GradientExplainer attribution -> [T,C], or None if ``shap`` is absent.

    ``model(X,q)`` is wrapped so SHAP sees a fixed query. ``background`` is a small
    set of train windows [Nb,T,C]. Optional dependency: kept out of the core path."""
    try:
        import shap
    except Exception:
        return None

    class _Wrap(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.m = model; s.q = query.unsqueeze(0)

        def forward(s, x):
            return s.m(x, s.q.expand(x.shape[0], -1)).unsqueeze(-1)

    try:
        w = _Wrap().to(device).eval()
        expl = shap.GradientExplainer(w, background.to(device))
        sv = expl.shap_values(X.unsqueeze(0).to(device))
        sv = sv[0] if isinstance(sv, list) else sv
        return torch.as_tensor(sv).abs().reshape(X.shape).to(X.device)
    except Exception:
        return None


def attention_rollout(model, query: torch.Tensor, X: torch.Tensor, device) -> Optional[torch.Tensor]:
    """Best-effort attention rollout over a Transformer baseline -> [T,C], or None.

    Registers forward hooks on ``nn.MultiheadAttention`` modules to capture per-layer
    attention, rolls them out (Abnar & Zuidema), maps the token-importance back to
    time, and broadcasts across channels. Returns None if the model exposes no
    attention (e.g. a TCN/LSTM), so the caller marks it not-applicable."""
    import torch.nn as nn
    attns: list = []
    handles = []

    def hook(mod, inp, out):
        # force need_weights; out may be (attn_out, attn_weights)
        if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
            attns.append(out[1].detach())

    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            handles.append(m.register_forward_hook(hook))
    if not handles:
        return None
    try:
        with torch.no_grad():
            model(X.unsqueeze(0).to(device), query.unsqueeze(0).to(device))
    except Exception:
        for h in handles:
            h.remove()
        return None
    for h in handles:
        h.remove()
    if not attns:
        return None
    # rollout: product of (A + I) normalised, per layer
    T = X.shape[0]
    roll = None
    for A in attns:
        A = A.mean(1) if A.dim() == 4 else A          # avg heads -> [B,L,L]
        A = A[0]
        A = A + torch.eye(A.shape[-1], device=A.device)
        A = A / A.sum(-1, keepdim=True)
        roll = A if roll is None else A @ roll
    if roll is None:
        return None
    imp = roll.mean(0)                                 # token importance
    imp = imp[:T] if imp.numel() >= T else torch.nn.functional.interpolate(
        imp[None, None], size=T, mode="linear", align_corners=False)[0, 0]
    imp = (imp - imp.min()).clamp(min=0)
    return imp.unsqueeze(1).expand(T, X.shape[1]).to(X.device)
