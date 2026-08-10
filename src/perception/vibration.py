"""Vibration feature front-end: raw bearing accelerometer snapshots -> HI trajectory.

Run-to-failure bearing rigs (XJTU-SY, FEMTO/PRONOSTIA, IMS) record a short
high-rate vibration *snapshot* every acquisition (here ~1 min @ 25.6 kHz, two
axes: Horizontal + Vertical). Predicates cannot ground on raw 25k-sample
waveforms, so each snapshot is reduced to a small vector of standard PHM
**health indicators (HI)** -- RMS, kurtosis, peak, crest factor, and a few FFT
band energies, per axis. The ordered sequence of snapshots over a bearing's life
becomes a tabular HI *trajectory* ``[n_snapshots, n_HI]``, identical in shape to
the C-MAPSS per-cycle sensor table, so the rest of the pipeline is unchanged.

Design choices (fixed, so the channel set is consistent across bearings/loads):

* features are computed per axis then concatenated (H-features ++ V-features);
* band energies split the one-sided spectrum into ``n_bands`` equal-frequency
  bands (energy = sum of squared magnitude in the band);
* everything is pure numpy/stdlib and deterministic (no pandas, no randomness).

The companion ``perception/features.py`` is the older PRONOSTIA-specific helper;
this module is the uniform front-end the XJTU adapter (and later FEMTO/IMS) uses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Per-axis scalar features (in fixed order); band energies are appended after.
_SCALAR_NAMES = ["rms", "kurtosis", "peak", "crest"]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def _kurtosis(x: np.ndarray) -> float:
    """Fisher kurtosis (Gaussian -> 3.0). Rises as impulsive defects appear."""
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean((x - m) ** 4) / (s ** 4))


def _band_energies(x: np.ndarray, n_bands: int) -> np.ndarray:
    """Energy in ``n_bands`` equal-frequency bands of the one-sided spectrum."""
    if n_bands <= 0:
        return np.empty(0, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x))
    edges = np.linspace(0, len(spec), n_bands + 1).astype(int)
    out = np.empty(n_bands, dtype=np.float64)
    for i in range(n_bands):
        seg = spec[edges[i]: edges[i + 1]]
        out[i] = float(np.sum(seg ** 2))
    return out


def _axis_features(x: np.ndarray, n_bands: int) -> np.ndarray:
    """[rms, kurtosis, peak, crest, band0..band{n_bands-1}] for one 1-D axis."""
    rms = _rms(x)
    peak = float(np.max(np.abs(x)))
    crest = float(peak / rms) if rms > 1e-12 else 0.0
    kurt = _kurtosis(x)
    return np.concatenate(
        [np.array([rms, kurt, peak, crest], dtype=np.float64),
         _band_energies(x, n_bands)]
    )


def feature_names(n_bands: int = 3) -> list[str]:
    """HI channel names for one snapshot: H-features then V-features."""
    base = _SCALAR_NAMES + [f"fft_b{i}" for i in range(n_bands)]
    return [f"h_{n}" for n in base] + [f"v_{n}" for n in base]


def snapshot_features(
    snapshot: np.ndarray, fs: int = 25600, n_bands: int = 3
) -> tuple[np.ndarray, list[str]]:
    """One snapshot ``[N, 2]`` (Horizontal, Vertical) -> one HI vector + names.

    Concatenates the per-axis PHM features (H then V) into a single health-
    indicator vector. ``fs`` is accepted for interface/forward-compatibility
    (band edges are spectrum-relative, so the result is fs-independent given a
    fixed snapshot length). Deterministic.
    """
    snap = np.asarray(snapshot, dtype=np.float64)
    if snap.ndim == 1:
        snap = snap[:, None]
    if snap.shape[1] < 2:
        raise ValueError(f"snapshot must have 2 columns (H, V), got {snap.shape}")
    h = _axis_features(snap[:, 0], n_bands)
    v = _axis_features(snap[:, 1], n_bands)
    return np.concatenate([h, v]), feature_names(n_bands)


def _read_snapshot_csv(path: Path) -> np.ndarray:
    """Read one XJTU snapshot CSV (header + 2 columns H,V) -> [N, 2] float64."""
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)


def _ordered_snapshot_paths(bearing_dir: Path) -> list[Path]:
    """All ``{k}.csv`` snapshots in numeric filename order (robust to gaps)."""
    paths = list(bearing_dir.glob("*.csv"))

    def key(p: Path):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem)  # non-numeric names sort after, lexicographically

    return sorted(paths, key=key)


def bearing_trajectory(
    bearing_dir: str | Path, fs: int = 25600, n_bands: int = 3
) -> tuple[np.ndarray, list[str]]:
    """Read a bearing's snapshot CSVs in numeric order -> HI trajectory + names.

    Returns ``(values[n_snapshots, n_HI], names)`` where row ``k`` is the HI
    vector of the k-th snapshot in the bearing's life (the degradation timeline).
    """
    bearing_dir = Path(bearing_dir)
    paths = _ordered_snapshot_paths(bearing_dir)
    if not paths:
        raise FileNotFoundError(f"no snapshot CSVs in {bearing_dir}")
    rows = []
    names: list[str] = []
    for p in paths:
        snap = _read_snapshot_csv(p)
        feats, names = snapshot_features(snap, fs=fs, n_bands=n_bands)
        rows.append(feats)
    values = np.stack(rows, axis=0)  # [n_snapshots, n_HI]
    return values, names
