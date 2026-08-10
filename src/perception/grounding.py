"""Temporal perception: signal window -> fuzzy predicate truths (symbolic state).

Grounds four interpretable predicate families per channel:
    high_c, low_c   (level)     rising_c, falling_c  (trend)
as calibrated sigmoids over a smoothed level and a local slope. Thresholds are
fixed from training-set quantiles (``Calibrator``), so the predicates are
trustworthy by construction rather than learned.

Output symbolic state mu has shape [T, P] with P = 4 * n_channels, and a
predicate index map (name, channel) -> column for the executor.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

PRED_FAMILIES = ["high", "low", "rising", "falling"]


def smooth(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Causal moving-average smoothing along time. x: [T, C] -> [T, C]."""
    if k <= 1:
        return x
    T, C = x.shape
    pad = x[:1].repeat(k - 1, 1)  # left-pad with first value (causal)
    xp = torch.cat([pad, x], dim=0)            # [T+k-1, C]
    kernel = torch.ones(k) / k
    out = torch.stack(
        [torch.nn.functional.conv1d(
            xp[:, c].view(1, 1, -1), kernel.view(1, 1, -1)
        ).view(-1) for c in range(C)],
        dim=1,
    )
    return out  # [T, C]


def local_slope(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Local slope via first difference of the smoothed signal. [T, C] -> [T, C]."""
    s = smooth(x, k)
    d = torch.zeros_like(s)
    d[1:] = s[1:] - s[:-1]
    return d


@dataclass
class Calibrator:
    """Per-channel thresholds and sharpness for grounding, fit on training data."""

    hi: torch.Tensor      # [C] high threshold (level)
    lo: torch.Tensor      # [C] low threshold (level)
    slope_scale: torch.Tensor  # [C] scale for trend sharpness
    a_level: float = 4.0  # sigmoid sharpness for level predicates
    smooth_k: int = 5

    @staticmethod
    def fit(
        train_windows: torch.Tensor,
        hi_q: float = 0.80,
        lo_q: float = 0.20,
        smooth_k: int = 5,
    ) -> "Calibrator":
        """Fit thresholds from training windows X:[W, T, C] using level quantiles."""
        W, T, C = train_windows.shape
        flat = train_windows.reshape(-1, C)  # [W*T, C]
        hi = torch.quantile(flat, hi_q, dim=0)
        lo = torch.quantile(flat, lo_q, dim=0)
        # slope scale: std of per-step differences, per channel
        diffs = train_windows[:, 1:, :] - train_windows[:, :-1, :]
        slope_scale = diffs.reshape(-1, C).std(dim=0).clamp_min(1e-3)
        return Calibrator(hi=hi, lo=lo, slope_scale=slope_scale, smooth_k=smooth_k)


def ground(x: torch.Tensor, cal: Calibrator) -> tuple[torch.Tensor, dict]:
    """Ground one window x:[T, C] into fuzzy truths mu:[T, 4C] + predicate index.

    Columns are ordered family-major then channel:
        high(0..C-1), low(0..C-1), rising(0..C-1), falling(0..C-1).
    """
    T, C = x.shape
    z = smooth(x, cal.smooth_k)              # level estimate [T, C]
    dz = local_slope(x, cal.smooth_k)        # slope [T, C]

    mu_high = torch.sigmoid(cal.a_level * (z - cal.hi))
    mu_low = torch.sigmoid(cal.a_level * (cal.lo - z))
    mu_rising = torch.sigmoid(dz / cal.slope_scale)
    mu_falling = torch.sigmoid(-dz / cal.slope_scale)

    mu = torch.cat([mu_high, mu_low, mu_rising, mu_falling], dim=1)  # [T, 4C]

    pidx: dict[tuple[str, int], int] = {}
    for fi, fam in enumerate(PRED_FAMILIES):
        for c in range(C):
            pidx[(fam, c)] = fi * C + c
    return mu, pidx


def predicate_index(n_channels: int) -> dict[tuple[str, int], int]:
    """Predicate index map without grounding (for building/validating programs)."""
    pidx: dict[tuple[str, int], int] = {}
    for fi, fam in enumerate(PRED_FAMILIES):
        for c in range(n_channels):
            pidx[(fam, c)] = fi * n_channels + c
    return pidx
