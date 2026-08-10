"""PRONOSTIA / FEMTO bearing dataset loader.

Builds run-to-failure feature trajectories from per-bearing acceleration CSV
snapshots. Raw vibration is never grounded directly: each ``acc_*.csv`` snapshot
is first converted into scalar feature channels by ``perception.features``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Callable

import numpy as np
import torch

from perception.features import FEATURE_NAMES, window_features


_ACC_RE = re.compile(r"acc_(\d+)\.csv$")


@dataclass
class PRONOSTIA:
    trajectories: list[torch.Tensor]      # per bearing: [T_i, C]
    rul: list[torch.Tensor]               # per bearing: [T_i]
    channel_names: list[str]
    bearing_ids: list[str]
    splits: list[str]
    dropped_channels: list[str] = field(default_factory=list)
    rul_cap: int | None = None
    rul_units: str = "snapshots"

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)


def _snapshot_number(path: Path) -> int:
    m = _ACC_RE.match(path.name)
    if not m:
        raise ValueError(f"not an acceleration snapshot: {path}")
    return int(m.group(1))


def acceleration_files(bearing_dir: str | Path) -> list[Path]:
    """Return ordered ``acc_*.csv`` files for one bearing run."""
    bearing_dir = Path(bearing_dir)
    return sorted(bearing_dir.glob("acc_*.csv"), key=_snapshot_number)


def _read_acceleration(path: Path) -> np.ndarray:
    """Read one PRONOSTIA acceleration CSV as [n_samples, 2]."""
    return np.loadtxt(path, delimiter=",", usecols=(4, 5), dtype=np.float64)


def _feature_names(axis: str, n_bands: int) -> list[str]:
    band_names = [f"band{i}" for i in range(n_bands)]
    base = FEATURE_NAMES + band_names
    names: list[str] = []
    if axis in ("horiz", "both"):
        names += [f"h_{n}" for n in base]
    if axis in ("vert", "both"):
        names += [f"v_{n}" for n in base]
    return names


def bearing_feature_trajectory(
    bearing_dir: str | Path,
    axis: str = "both",
    n_bands: int = 0,
) -> tuple[torch.Tensor, list[str]]:
    """Convert one bearing's ordered acceleration snapshots into features."""
    if axis not in {"horiz", "vert", "both"}:
        raise ValueError(f"axis must be horiz, vert, or both; got {axis!r}")

    rows = []
    for path in acceleration_files(bearing_dir):
        snap = _read_acceleration(path)
        cols = []
        if axis in ("horiz", "both"):
            cols.append(window_features(snap[:, 0], n_bands))
        if axis in ("vert", "both"):
            cols.append(window_features(snap[:, 1], n_bands))
        rows.append(np.concatenate(cols))

    if not rows:
        raise ValueError(f"no acc_*.csv files found in {bearing_dir}")

    traj = torch.tensor(np.stack(rows), dtype=torch.float32)
    return traj, _feature_names(axis, n_bands)


def bearing_dirs(root: str | Path, splits: list[str]) -> list[tuple[str, Path]]:
    """Return ``(split, bearing_dir)`` pairs for requested dataset splits."""
    root = Path(root)
    out: list[tuple[str, Path]] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"PRONOSTIA split not found: {split_dir}")
        out.extend((split, p) for p in sorted(split_dir.iterdir()) if p.is_dir())
    return out


def _rul_for_length(length: int, cap: int | None) -> torch.Tensor:
    rul = torch.arange(length - 1, -1, -1, dtype=torch.float32)
    if cap is not None:
        rul = torch.clamp(rul, max=float(cap))
    return rul


def load_pronostia(
    root: str | Path,
    splits: list[str] | None = None,
    axis: str = "both",
    n_bands: int = 0,
    rul_cap: int | None = None,
    flat_std_thresh: float = 1e-8,
    progress: Callable[[str], None] | None = None,
) -> PRONOSTIA:
    """Load complete PRONOSTIA bearing runs as standardized feature trajectories.

    ``root`` should point at the dataset directory containing ``Learning_set``,
    ``Full_Test_Set``, and ``Test_set``. Use complete run-to-failure splits
    (``Learning_set`` and optionally ``Full_Test_Set``) when RUL labels are
    needed; the truncated challenge ``Test_set`` is intentionally not included
    by default.
    """
    splits = splits or ["Learning_set"]

    raw_trajs: list[torch.Tensor] = []
    bearing_ids: list[str] = []
    names: list[str] | None = None
    for split, path in bearing_dirs(root, splits):
        if progress is not None:
            progress(f"extracting {split}/{path.name} ...")
        traj, traj_names = bearing_feature_trajectory(path, axis=axis, n_bands=n_bands)
        if progress is not None:
            progress(f"  {split}/{path.name}: snapshots={traj.shape[0]}")
        if names is None:
            names = traj_names
        elif names != traj_names:
            raise ValueError("feature channel mismatch across bearings")
        raw_trajs.append(traj)
        bearing_ids.append(f"{split}/{path.name}")

    if names is None or not raw_trajs:
        raise ValueError(f"no PRONOSTIA bearings loaded from {root}")

    all_rows = torch.cat(raw_trajs, dim=0)
    stds = all_rows.std(dim=0)
    keep_mask = stds > flat_std_thresh
    kept_names = [n for n, keep in zip(names, keep_mask.tolist()) if keep]
    dropped = [n for n, keep in zip(names, keep_mask.tolist()) if not keep]

    kept = all_rows[:, keep_mask]
    mean = kept.mean(dim=0, keepdim=True)
    std = kept.std(dim=0, keepdim=True).clamp_min(1e-8)

    trajectories = [((traj[:, keep_mask] - mean) / std).float() for traj in raw_trajs]
    ruls = [_rul_for_length(traj.shape[0], rul_cap) for traj in trajectories]

    return PRONOSTIA(
        trajectories=trajectories,
        rul=ruls,
        channel_names=kept_names,
        bearing_ids=bearing_ids,
        splits=splits,
        dropped_channels=dropped,
        rul_cap=rul_cap,
    )
