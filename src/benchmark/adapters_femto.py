"""FEMTO / PRONOSTIA run-to-failure bearing adapter -> per-bearing HI Series.

Layout (under ``data/raw/IEEE_PRONOSTIA_PHM/dataset/``)::

    Learning_set/   Bearing1_1, Bearing1_2, Bearing2_1, ... / {acc_00001.csv, ...}
    Full_Test_Set/  Bearing1_3 .. Bearing1_7, Bearing2_3 .. , Bearing3_3 / {acc_*.csv}

Each ``acc_XXXXX.csv`` is one ~0.1 s vibration snapshot @ 25.6 kHz; the row format
is ``hour, minute, second, microsecond, horizontal_accel, vertical_accel`` (6
columns, comma-separated), so columns 4 and 5 are the two accelerometer axes. The
ordered sequence of snapshots over a bearing's life is its degradation trajectory.

This adapter is the cross-rig counterpart of ``XJTUAdapter``: it applies the SAME
unified vibration front-end (``perception.vibration.snapshot_features``) so FEMTO
bearings produce the identical 14-channel health-indicator (HI) vocabulary as
XJTU. That shared vocabulary is what makes cross-rig transfer possible -- a
perception net trained on one rig has the right input/output shape on another, and
the executor/programs are unchanged (see ``MultiRigAdapter``).

* ``condition`` = the operating condition ``"1" | "2" | "3"`` (the digit in
  ``Bearing{c}_{n}``); ``split_axis = "condition"``.
* ``unit_id`` = ``"femto:{c}:{bearing}"`` (namespaced across rigs).
* ``meta["rul"]`` = per-snapshot remaining life in snapshots (``n - k``).
* HI trajectories are **cached** to ``data/processed/femto_hi.pkl`` (FFT over many
  files is slow); pass ``force_recompute=True`` to rebuild.

The older ``benchmark/pronostia.py`` is a bespoke loader with a different feature
naming; this adapter supersedes it for the modelling pipeline.
"""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from perception.vibration import feature_names, snapshot_features

from .adapters import DatasetAdapter, Series

# PRONOSTIA operating conditions are encoded in the bearing name (Bearing{c}_{n}).
FEMTO_CONDITIONS = ("1", "2", "3")
_SPLITS = ("Learning_set", "Full_Test_Set")  # full lives; Test_set is a truncated dup

_DEFAULT_ROOT = "data/raw/IEEE_PRONOSTIA_PHM/dataset"
_DEFAULT_CACHE = "data/processed/femto_hi.pkl"

_ACC_RE = re.compile(r"acc_(\d+)\.csv$")
_BEARING_RE = re.compile(r"Bearing(\d+)_(\d+)$")


def _acc_index(path: Path) -> int:
    m = _ACC_RE.search(path.name)
    return int(m.group(1)) if m else -1


def _ordered_acc_files(bearing_dir: Path) -> list[Path]:
    """All ``acc_*.csv`` snapshots in numeric filename order."""
    return sorted(bearing_dir.glob("acc_*.csv"), key=_acc_index)


def _detect_delimiter(path: Path) -> Optional[str]:
    """PRONOSTIA mixes ',' and ';' (and occasionally whitespace) across bearings."""
    with open(path) as fh:
        first = fh.readline()
    for d in (",", ";", "\t"):
        if first.count(d) >= 5:          # need 6 columns -> >=5 separators
            return d
    return None                          # fall back to any-whitespace splitting


def _read_snapshot(path: Path) -> np.ndarray:
    """One PRONOSTIA acc CSV -> [N, 2] (horizontal, vertical) float64.

    Robust to the dataset's mixed comma/semicolon/whitespace delimiters.
    """
    return np.loadtxt(path, delimiter=_detect_delimiter(path),
                      usecols=(4, 5), dtype=np.float64)


def _femto_trajectory(bearing_dir: Path, fs: int, n_bands: int) -> tuple[np.ndarray, list[str]]:
    """Read a bearing's acc snapshots in order -> HI trajectory [n, C] + names."""
    paths = _ordered_acc_files(bearing_dir)
    if not paths:
        raise FileNotFoundError(f"no acc_*.csv in {bearing_dir}")
    rows, names = [], feature_names(n_bands)
    for p in paths:
        snap = _read_snapshot(p)
        feats, names = snapshot_features(snap, fs=fs, n_bands=n_bands)
        rows.append(feats)
    return np.stack(rows, axis=0), names


def _bearing_signature(bearing_dir: Path) -> tuple[int, int]:
    """Cheap fingerprint: (#acc files, max numeric index) -> cache invalidation."""
    n, mx = 0, -1
    for p in bearing_dir.glob("acc_*.csv"):
        n += 1
        mx = max(mx, _acc_index(p))
    return n, mx


class FEMTOAdapter(DatasetAdapter):
    name = "femto"
    split_axis = "condition"   # operating condition 1/2/3; "unit" also valid (held-out bearing)

    def __init__(
        self,
        root: str | Path = _DEFAULT_ROOT,
        conditions: Optional[tuple[str, ...]] = None,
        fs: int = 25600,
        n_bands: int = 3,
        min_snapshots: int = 1,
        cache_path: str | Path = _DEFAULT_CACHE,
        force_recompute: bool = False,
    ):
        self.root = Path(root)
        # stored as ``_conditions`` so we don't shadow the base ``conditions(series)`` helper
        self._conditions = tuple(conditions) if conditions is not None else FEMTO_CONDITIONS
        self.fs = fs
        self.n_bands = n_bands
        self.min_snapshots = min_snapshots
        self.cache_path = Path(cache_path)
        self.force_recompute = force_recompute

    # -- cache helpers -------------------------------------------------------- #

    def _cache_key(self) -> dict:
        return {"fs": self.fs, "n_bands": self.n_bands}

    def _load_cache(self) -> dict:
        if self.force_recompute or not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, "rb") as fh:
                blob = pickle.load(fh)
        except Exception:
            return {}
        if blob.get("key") != self._cache_key():
            return {}
        return blob.get("bearings", {})

    def _save_cache(self, bearings: dict) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        blob = {"key": self._cache_key(), "bearings": bearings}
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.cache_path)  # atomic

    # -- discovery ------------------------------------------------------------ #

    def _discover(self) -> list[tuple[str, str, Path]]:
        """(condition, bearing, dir) for every bearing folder with acc snapshots.

        De-duplicates by bearing name (a bearing appears once across splits).
        """
        seen: dict[str, tuple[str, str, Path]] = {}
        for split in _SPLITS:
            sdir = self.root / split
            if not sdir.is_dir():
                continue
            for bdir in sorted(sdir.iterdir()):
                if not bdir.is_dir():
                    continue
                m = _BEARING_RE.search(bdir.name)
                if not m:
                    continue
                cond = m.group(1)
                if cond not in self._conditions:
                    continue
                if bdir.name in seen:
                    continue
                if sum(1 for _ in bdir.glob("acc_*.csv")) >= self.min_snapshots:
                    seen[bdir.name] = (cond, bdir.name, bdir)
        return sorted(seen.values(), key=lambda t: (t[0], t[1]))

    # -- cache-only reconstruction (portable: no raw tree needed) ------------- #

    def _series_from_cache(self, cache: dict, names: list) -> list[Series]:
        """Rebuild Series purely from the HI cache when raw data is absent."""
        out: list[Series] = []
        for uid, entry in sorted(cache.items()):
            _, cond, bearing = uid.split(":", 2)   # "femto:{cond}:{bearing}"
            vals = torch.as_tensor(entry["values"], dtype=torch.float32)
            n = vals.shape[0]
            out.append(Series(
                values=vals, unit_id=uid, dataset="femto", channel_names=list(names),
                condition=cond, fs=float(self.fs),
                meta={"rul": torch.arange(n, 0, -1, dtype=torch.float32),
                      "bearing": bearing, "condition": cond, "n_snapshots": n,
                      "rig": "femto"}))
        return out

    # -- main ----------------------------------------------------------------- #

    def load(self) -> list[Series]:
        names = feature_names(self.n_bands)
        cache = self._load_cache()
        discovered = self._discover()
        if not discovered:
            if cache:
                print(f"[FEMTOAdapter] raw data absent under {self.root}; "
                      f"loading {len(cache)} bearings from cache {self.cache_path}", flush=True)
                return self._series_from_cache(cache, names)
            raise SystemExit(f"no FEMTO bearings with acc snapshots under {self.root}")

        series: list[Series] = []
        new_cache = dict(cache)
        dirty = False
        for cond, bearing, bdir in discovered:
            uid = f"femto:{cond}:{bearing}"
            sig = _bearing_signature(bdir)
            entry = cache.get(uid)
            if entry is not None and entry.get("sig") == list(sig):
                values = entry["values"]
            else:
                values, names = _femto_trajectory(bdir, self.fs, self.n_bands)
                new_cache[uid] = {"sig": list(sig), "values": values}
                dirty = True

            vals = torch.as_tensor(values, dtype=torch.float32)  # [n, C]
            n = vals.shape[0]
            rul = torch.arange(n, 0, -1, dtype=torch.float32)
            series.append(Series(
                values=vals,
                unit_id=uid,
                dataset="femto",
                channel_names=list(names),
                condition=cond,
                fs=float(self.fs),
                meta={
                    "rul": rul,
                    "bearing": bearing,
                    "condition": cond,
                    "n_snapshots": n,
                    "rig": "femto",
                },
            ))

        if dirty:
            self._save_cache(new_cache)
        return series
