"""PRONOSTIA / FEMTO bearing dataset: raw vibration -> feature channels.

Unlike C-MAPSS (per-cycle aligned sensors), PRONOSTIA provides high-rate
horizontal/vertical accelerometer signals. Predicates cannot ground on raw
vibration directly, so we extract a windowed feature time series -- RMS,
kurtosis, peak, crest factor, and band energy -- per accelerometer axis. These
features become the "channels" the perception layer grounds predicates on.

File format (per the FEMTO release): CSV rows of
    hour, minute, second, microsecond, horiz_accel, vert_accel
one file per acquisition snapshot; many snapshots per bearing run-to-failure.

This module focuses on the feature transform; the directory walking is left to
the loader script so paths stay configurable.
"""
from __future__ import annotations

import numpy as np
import torch


def _kurtosis(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float(((x - m) ** 4).mean() / (s ** 4))


def _crest_factor(x: np.ndarray) -> float:
    rms = np.sqrt((x ** 2).mean())
    if rms < 1e-12:
        return 0.0
    return float(np.abs(x).max() / rms)


def window_features(signal: np.ndarray, n_bands: int = 0) -> np.ndarray:
    """Extract scalar features from one accelerometer snapshot (1D array).

    Returns [rms, kurtosis, peak, crest_factor] (+ optional band energies).
    """
    rms = float(np.sqrt((signal ** 2).mean()))
    kurt = _kurtosis(signal)
    peak = float(np.abs(signal).max())
    crest = _crest_factor(signal)
    feats = [rms, kurt, peak, crest]

    if n_bands > 0:
        spec = np.abs(np.fft.rfft(signal))
        edges = np.linspace(0, len(spec), n_bands + 1).astype(int)
        for i in range(n_bands):
            seg = spec[edges[i] : edges[i + 1]]
            feats.append(float((seg ** 2).sum()))
    return np.asarray(feats, dtype=np.float64)


FEATURE_NAMES = ["rms", "kurtosis", "peak", "crest"]


def build_trajectory(
    snapshots: list[np.ndarray],
    axis: str = "both",
    n_bands: int = 0,
) -> tuple[torch.Tensor, list[str]]:
    """Turn an ordered list of accelerometer snapshots into a feature trajectory.

    Each snapshot is a 2D array [n_samples, 2] (horizontal, vertical). The output
    is a [T, C] tensor where T = number of snapshots (the degradation timeline)
    and C = features per selected axis. ``axis`` in {horiz, vert, both}.
    """
    rows = []
    for snap in snapshots:
        if snap.ndim == 1:
            snap = snap[:, None]
        cols = []
        if axis in ("horiz", "both"):
            cols.append(window_features(snap[:, 0], n_bands))
        if axis in ("vert", "both") and snap.shape[1] > 1:
            cols.append(window_features(snap[:, 1], n_bands))
        rows.append(np.concatenate(cols))
    traj = torch.tensor(np.stack(rows), dtype=torch.float32)  # [T, C]

    names = []
    band_names = [f"band{i}" for i in range(n_bands)]
    base = FEATURE_NAMES + band_names
    if axis in ("horiz", "both"):
        names += [f"h_{n}" for n in base]
    if axis in ("vert", "both"):
        names += [f"v_{n}" for n in base]
    return traj, names


def standardize(traj: torch.Tensor) -> torch.Tensor:
    """Per-channel standardization of a feature trajectory."""
    mean = traj.mean(0, keepdim=True)
    std = traj.std(0, keepdim=True).clamp_min(1e-8)
    return (traj - mean) / std
