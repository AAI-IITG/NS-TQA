"""C-MAPSS turbofan dataset loader.

Parses the standard C-MAPSS text files into per-engine run-to-failure
trajectories, computes (piecewise-linear, capped) RUL, and identifies
non-informative (flat) sensor channels so that predicate grounding does not
produce degenerate predicates.

File format (whitespace-separated, no header), per the NASA release:
    unit  cycle  op1 op2 op3  s1 s2 ... s21

Usage:
    ds = load_cmapss("data/raw/CMAPSS", subset="FD001")
    ds.trajectories  -> list of [T_i, C] float tensors (one per engine)
    ds.rul           -> list of [T_i] RUL targets
    ds.channel_names -> informative channel names kept
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
ALL_COLS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS


@dataclass
class CMAPSS:
    trajectories: list[torch.Tensor]      # per engine: [T_i, C]
    rul: list[torch.Tensor]               # per engine: [T_i]
    channel_names: list[str]              # kept (informative) channels
    engine_ids: list[int] = field(default_factory=list)
    dropped_channels: list[str] = field(default_factory=list)
    rul_cap: int = 125
    subset: str = "FD001"

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)


def _read_raw(path: Path) -> np.ndarray:
    """Read a C-MAPSS train_*.txt into a 2D float array, trimming trailing blanks."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            rows.append([float(x) for x in parts])
    arr = np.asarray(rows, dtype=np.float64)
    # some releases have 2 trailing empty columns; keep only the first 26
    if arr.shape[1] > 26:
        arr = arr[:, :26]
    return arr


def _compute_rul(cycles: np.ndarray, cap: int) -> np.ndarray:
    """Piecewise-linear capped RUL: RUL(t) = min(max_cycle - cycle, cap)."""
    max_cycle = cycles.max()
    rul = max_cycle - cycles
    return np.minimum(rul, cap)


def _kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    """Tiny Lloyd's k-means (no sklearn — lean deps). Returns per-row cluster labels.

    Used only to identify C-MAPSS operating regimes from the 3 op-setting columns,
    which form ~k well-separated point clusters, so convergence is fast and stable.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centroids = X[rng.choice(n, size=min(k, n), replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)  # [n,k]
        new = d.argmin(axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for c in range(len(centroids)):
            m = labels == c
            if m.any():
                centroids[c] = X[m].mean(axis=0)
    return labels


def load_cmapss(
    root: str | Path,
    subset: str = "FD001",
    rul_cap: int = 125,
    flat_std_thresh: float = 1e-6,
    op_normalize: bool = False,
    n_regimes: int = 6,
) -> CMAPSS:
    """Load a C-MAPSS subset into per-engine trajectories with capped RUL.

    Flat sensors (std below ``flat_std_thresh`` across the whole training set)
    are dropped: grounding ``high``/``rising`` on a constant channel yields a
    degenerate always-true/always-false predicate, which we must avoid.

    ``op_normalize`` (operating-condition normalization): FD002/FD004 interleave
    ``n_regimes`` operating conditions cycle-to-cycle, which dominates the raw
    sensor signal and makes an FD001-trained model see them as out-of-distribution.
    When True, the 3 op-setting columns are clustered (k-means) into regimes and
    each SENSOR is z-scored WITHIN its regime, removing the regime-induced level
    jumps so the underlying degradation is comparable across conditions/subsets.
    This is a LABEL-FREE input transform (uses only op-settings + sensor values,
    never RUL/answers), computed on this subset's own inputs — non-circular, and
    the standard preprocessing for the multi-regime C-MAPSS subsets.
    """
    root = Path(root)
    train_path = root / f"train_{subset}.txt"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Expected C-MAPSS files like train_FD001.txt "
            f"under {root}."
        )

    arr = _read_raw(train_path)  # [N, 26]
    units = arr[:, 0].astype(int)
    cycles = arr[:, 1]
    feats = arr[:, 2:]  # [N, 24] = 3 op + 21 sensors
    feat_names = OP_COLS + SENSOR_COLS

    if op_normalize:
        n_op = len(OP_COLS)
        op = feats[:, :n_op]
        sensors = feats[:, n_op:].copy()
        # regime label per row from op-settings (standardize op cols first for k-means scale)
        op_std = (op - op.mean(0)) / np.where(op.std(0) < 1e-8, 1.0, op.std(0))
        labels = _kmeans(op_std, k=n_regimes)
        # z-score each sensor WITHIN its regime (train-only stats)
        for r in np.unique(labels):
            m = labels == r
            mu = sensors[m].mean(0)
            sd = sensors[m].std(0)
            sd[sd < 1e-8] = 1.0
            sensors[m] = (sensors[m] - mu) / sd
        feats = sensors                     # op cols dropped (regime removed)
        feat_names = list(SENSOR_COLS)

    # detect flat channels on the whole training set
    stds = feats.std(axis=0)
    keep_mask = stds > flat_std_thresh
    kept_names = [n for n, k in zip(feat_names, keep_mask) if k]
    dropped = [n for n, k in zip(feat_names, keep_mask) if not k]

    # per-channel standardization stats on kept channels (train-only)
    kept = feats[:, keep_mask]
    mean = kept.mean(axis=0)
    std = kept.std(axis=0)
    std[std < 1e-8] = 1.0

    trajectories, ruls, engine_ids = [], [], []
    for u in np.unique(units):
        m = units == u
        x = (kept[m] - mean) / std            # standardized [T_i, C]
        r = _compute_rul(cycles[m], rul_cap)  # [T_i]
        # ensure chronological order by cycle
        order = np.argsort(cycles[m])
        trajectories.append(torch.tensor(x[order], dtype=torch.float32))
        ruls.append(torch.tensor(r[order], dtype=torch.float32))
        engine_ids.append(int(u))

    return CMAPSS(
        trajectories=trajectories,
        rul=ruls,
        channel_names=kept_names,
        engine_ids=engine_ids,
        dropped_channels=dropped,
        rul_cap=rul_cap,
        subset=subset,
    )


def windows(
    ds: CMAPSS, seq_len: int = 48, stride: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice trajectories into fixed windows. Returns (X:[W,seq_len,C], rul:[W]).

    The RUL target for a window is the RUL at its last timestep.
    """
    xs, rs = [], []
    for traj, rul in zip(ds.trajectories, ds.rul):
        T = traj.shape[0]
        if T < seq_len:
            continue
        for s in range(0, T - seq_len + 1, stride):
            xs.append(traj[s : s + seq_len])
            rs.append(rul[s + seq_len - 1])
    if not xs:
        raise ValueError("no windows produced; seq_len longer than all trajectories")
    return torch.stack(xs), torch.stack(rs)


def window_engine_ids(ds: CMAPSS, seq_len: int = 48, stride: int = 1) -> list[int]:
    """Return the source engine id for each window produced by ``windows``."""
    ids = []
    for engine_id, traj in zip(ds.engine_ids, ds.trajectories):
        T = traj.shape[0]
        if T < seq_len:
            continue
        ids.extend([engine_id] * len(range(0, T - seq_len + 1, stride)))
    return ids
