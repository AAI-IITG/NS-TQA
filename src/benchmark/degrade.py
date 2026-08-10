"""Input-side sensor degradation for the robustness study (WO-1B).

These are **pure, deterministic** transforms applied to a windowed signal
``X [T, C]`` (the model's normalized input view). They model realistic sensing
faults — additive noise, dropped samples, calibration drift, clock jitter — so we
can ask *at what severity, if any, a learned perception head beats a fixed-threshold
STL-only grounding.*

CRITICAL INVARIANT (keeps the test non-circular):
    Degradation is applied to the model INPUT ONLY. The privileged label
    ``answer_star`` stays defined on the CLEAN ``mu_star`` (clean privileged
    grounding). We never re-ground the degraded signal to make a new label. So the
    question is always "can the method still recover the clean answer from a
    corrupted signal", not "does the label move with the corruption". See
    ``degrade_instances`` and ``docs`` / the WO-1B write-up.

Determinism: every function takes a ``seed`` (or a ``torch.Generator``); the same
(window, severity, seed) always yields the same corruption, so a benchmark degraded
at build time is reproducible across methods and model seeds.

Only ``add_noise`` and ``calibration_drift`` are the primary WO-1B families;
``drop_samples`` and ``resample_jitter`` round out the fault taxonomy.
"""
from __future__ import annotations

from typing import Optional, Union

import torch

Number = Union[float, torch.Tensor]


def _gen(seed: Optional[int], device=None) -> torch.Generator:
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(int(seed))
    return g


def add_noise(X: torch.Tensor, snr_db: float, seed: Optional[int] = 0) -> torch.Tensor:
    """Add white Gaussian noise at a per-channel signal-to-noise ratio ``snr_db``.

    Signal power is measured **per channel** as the mean square of that channel's
    window; noise std is set so that ``10*log10(P_signal / P_noise) = snr_db``. A
    flat channel (zero power) receives no noise (0/0 guarded).
    """
    X = X.float()
    T, C = X.shape
    p_sig = (X ** 2).mean(dim=0)                       # [C]
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    std = torch.sqrt(torch.clamp(p_noise, min=0.0))    # [C]
    g = _gen(seed)
    noise = torch.randn(T, C, generator=g) * std       # broadcast per channel
    return X + noise


def drop_samples(
    X: torch.Tensor, p: float, mode: str = "zero", seed: Optional[int] = 0
) -> torch.Tensor:
    """Randomly drop a fraction ``p`` of time steps (per channel, independently).

    ``mode``:
      - ``"zero"``   : dropped samples set to 0 (sensor read fails to a rail).
      - ``"hold"``   : forward-fill the last valid sample (zero-order hold).
      - ``"interp"`` : linearly interpolate across the gap from valid neighbours.
    A leading run of drops (no prior valid sample) is back-filled from the first
    valid sample for ``hold``/``interp`` so no NaNs remain.
    """
    if not 0.0 <= p < 1.0:
        raise ValueError(f"drop fraction p must be in [0,1), got {p}")
    if p == 0.0:
        return X.clone().float()
    X = X.float()
    T, C = X.shape
    g = _gen(seed)
    drop = torch.rand(T, C, generator=g) < p           # True = dropped
    out = X.clone()
    if mode == "zero":
        out[drop] = 0.0
        return out
    if mode not in ("hold", "interp"):
        raise ValueError(f"unknown drop mode {mode!r}")
    for c in range(C):
        valid_idx = torch.nonzero(~drop[:, c], as_tuple=False).flatten()
        if valid_idx.numel() == 0:                     # whole channel dropped -> zeros
            out[:, c] = 0.0
            continue
        col = X[:, c]
        t = torch.arange(T, dtype=torch.float32)
        if mode == "hold":
            # index of the most recent valid sample at or before each t
            last = torch.full((T,), -1, dtype=torch.long)
            cur = -1
            for i in range(T):
                if not drop[i, c]:
                    cur = i
                last[i] = cur
            first_valid = int(valid_idx[0])
            last[last < 0] = first_valid               # back-fill leading gap
            out[:, c] = col[last]
        else:  # interp
            vx = valid_idx.to(torch.float32)
            vy = col[valid_idx]
            out[:, c] = _interp1d(t, vx, vy)
    return out


def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """1-D linear interpolation (like numpy.interp) with flat extrapolation."""
    x = x.clamp(min=float(xp[0]), max=float(xp[-1]))
    idx = torch.searchsorted(xp, x, right=True).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    denom = (x1 - x0).clamp(min=1e-8)
    return y0 + (y1 - y0) * (x - x0) / denom


def calibration_drift(
    X: torch.Tensor, gain: Number = 1.0, offset: Number = 0.0, seed: Optional[int] = 0
) -> torch.Tensor:
    """Per-channel multiplicative gain and additive offset: ``X * gain + offset``.

    ``gain``/``offset`` may be scalars (same for all channels) or length-``C``
    tensors. Models slow calibration drift of a sensor. Deterministic (no
    randomness); ``seed`` is accepted for a uniform interface.
    """
    X = X.float()
    C = X.shape[1]
    g = gain if isinstance(gain, torch.Tensor) else torch.full((C,), float(gain))
    o = offset if isinstance(offset, torch.Tensor) else torch.full((C,), float(offset))
    return X * g.float() + o.float()


def resample_jitter(X: torch.Tensor, factor: float, seed: Optional[int] = 0) -> torch.Tensor:
    """Clock jitter: warp the time axis by ``factor`` then resample back to ``T``.

    ``factor>1`` stretches (slower clock), ``factor<1`` compresses; the window is
    linearly resampled onto ``T`` points so shape is preserved. Endpoints are held.
    Deterministic given ``factor``.
    """
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")
    X = X.float()
    T, C = X.shape
    src = torch.arange(T, dtype=torch.float32)
    # warped sample positions on the original grid
    warped = torch.linspace(0.0, (T - 1) * factor, T).clamp(max=float(T - 1))
    out = torch.empty_like(X)
    for c in range(C):
        out[:, c] = _interp1d(warped, src, X[:, c])
    return out


# --------------------------------------------------------------------------- #
# applying a degradation to a set of QA instances (label-preserving)
# --------------------------------------------------------------------------- #

_FAMILIES = {
    "noise": add_noise,
    "drop": drop_samples,
    "drift": calibration_drift,
    "jitter": resample_jitter,
}


def degrade_instances(instances: list, family: str, seed: int = 0, **params) -> list:
    """Return copies of ``instances`` with ``X`` degraded and the LABEL UNCHANGED.

    ``answer_star`` / ``mu_star`` / ``phi_star`` are carried over from the clean
    instance (the answer is still defined on the clean privileged grounding). Only
    ``inst.X`` is corrupted. Each instance gets a distinct but deterministic seed
    (``seed`` mixed with its index) so the corruption is reproducible yet not
    identical across instances.
    """
    if family not in _FAMILIES:
        raise ValueError(f"unknown degradation family {family!r}; choose {list(_FAMILIES)}")
    fn = _FAMILIES[family]
    import copy
    out = []
    for i, inst in enumerate(instances):
        inst_seed = (int(seed) * 1_000_003 + i) % (2 ** 31)
        Xd = fn(inst.X, seed=inst_seed, **params)
        new = copy.copy(inst)                          # shallow: shares mu_star etc.
        new.X = Xd
        out.append(new)
    return out
