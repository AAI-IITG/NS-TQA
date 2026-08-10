"""XJTU-SY run-to-failure bearing adapter -> per-bearing HI Series.

Layout (under ``.../Data/XJTU-SY_Bearing_Datasets/``)::

    35Hz12kN/   Bearing1_1 .. Bearing1_5 /  {1.csv, 2.csv, ...}
    37.5Hz11kN/ Bearing2_1 .. Bearing2_5 /  {1.csv, ...}
    40Hz10kN/   Bearing3_1 .. Bearing3_5 /  {1.csv, ...}

Each CSV is one ~1-min vibration snapshot @ 25.6 kHz with two columns
(Horizontal, Vertical). The adapter applies the fixed vibration front-end
(``perception.vibration``) to turn each bearing's ordered snapshot sequence into
a tabular health-indicator (HI) trajectory, so the cross-condition (load) split
and the held-out-bearing split run off the shared ``build_real_benchmark``
pipeline exactly like C-MAPSS.

* ``condition`` = the load folder (e.g. ``"40Hz10kN"``); ``split_axis="condition"``.
* ``unit_id`` = ``"{condition}:{bearing}"`` (namespaced across loads).
* ``meta["rul"]`` = per-snapshot remaining life in snapshots (``n - k``).
* Computing FFT features over thousands of files is slow, so trajectories are
  **cached** to ``data/processed/xjtu_hi.pkl`` (keyed by config + per-bearing
  signature); pass ``force_recompute=True`` to rebuild.

Robustness: bearing folders may have gaps or missing indices, and (in partial
downloads) some loads/bearings may be empty -- the adapter loads exactly the
bearings that contain snapshot CSVs and skips the rest, keeping a consistent
14-channel HI set across whatever is present.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import torch

from perception.vibration import bearing_trajectory, feature_names

from .adapters import DatasetAdapter, Series

# Known XJTU operating conditions (load folders), in the dataset's order.
XJTU_CONDITIONS = ("35Hz12kN", "37.5Hz11kN", "40Hz10kN")

_DEFAULT_ROOT = ("data/raw/XJTU-SY_Bearing_Datasets/Data/"
                 "XJTU-SY_Bearing_Datasets")
_DEFAULT_CACHE = "data/processed/xjtu_hi.pkl"


def _bearing_signature(bearing_dir: Path) -> tuple[int, int]:
    """Cheap fingerprint of a bearing folder: (#csv files, max numeric index).

    Lets the cache detect added/removed snapshots without re-reading every file.
    """
    n, mx = 0, -1
    for p in bearing_dir.glob("*.csv"):
        n += 1
        try:
            mx = max(mx, int(p.stem))
        except ValueError:
            pass
    return n, mx


class XJTUAdapter(DatasetAdapter):
    name = "xjtu"
    split_axis = "condition"   # primary: cross-load; "unit" also valid (held-out bearing)

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
        # NB: stored as ``_conditions`` so we don't shadow the base class's
        # ``conditions(series)`` helper (distinct conditions present in a load).
        self._conditions = tuple(conditions) if conditions is not None else XJTU_CONDITIONS
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
            return {}  # front-end config changed -> recompute
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
        """(condition, bearing, dir) for every NON-EMPTY bearing folder present."""
        found: list[tuple[str, str, Path]] = []
        for cond in self._conditions:
            cdir = self.root / cond
            if not cdir.is_dir():
                continue
            for bdir in sorted(cdir.iterdir()):
                if not bdir.is_dir():
                    continue
                n_csv = sum(1 for _ in bdir.glob("*.csv"))
                if n_csv >= self.min_snapshots:
                    found.append((cond, bdir.name, bdir))
        return found

    # -- cache-only reconstruction (portable: no raw tree needed) ------------- #

    def _series_from_cache(self, cache: dict, names: list) -> list[Series]:
        """Rebuild Series purely from the HI cache when raw data is absent
        (e.g. on a cluster where only data/processed/*.pkl was synced)."""
        out: list[Series] = []
        for uid, entry in sorted(cache.items()):
            cond, bearing = uid.split(":", 1)
            vals = torch.as_tensor(entry["values"], dtype=torch.float32)
            n = vals.shape[0]
            out.append(Series(
                values=vals, unit_id=uid, dataset="xjtu", channel_names=list(names),
                condition=cond, fs=float(self.fs),
                meta={"rul": torch.arange(n, 0, -1, dtype=torch.float32),
                      "bearing": bearing, "condition": cond, "n_snapshots": n}))
        return out

    # -- main ----------------------------------------------------------------- #

    def load(self) -> list[Series]:
        names = feature_names(self.n_bands)
        cache = self._load_cache()
        discovered = self._discover()
        if not discovered:
            if cache:
                print(f"[XJTUAdapter] raw data absent under {self.root}; "
                      f"loading {len(cache)} bearings from cache {self.cache_path}", flush=True)
                return self._series_from_cache(cache, names)
            raise SystemExit(
                f"no XJTU bearings with CSV snapshots under {self.root}")

        series: list[Series] = []
        new_cache = dict(cache)
        dirty = False
        for cond, bearing, bdir in discovered:
            uid = f"{cond}:{bearing}"
            sig = _bearing_signature(bdir)
            entry = cache.get(uid)
            if entry is not None and entry.get("sig") == list(sig):
                values = entry["values"]                      # cached numpy [n, C]
            else:
                values, names = bearing_trajectory(
                    bdir, fs=self.fs, n_bands=self.n_bands)
                new_cache[uid] = {"sig": list(sig), "values": values}
                dirty = True

            vals = torch.as_tensor(values, dtype=torch.float32)  # [n, C]
            n = vals.shape[0]
            rul = torch.arange(n, 0, -1, dtype=torch.float32)     # n, n-1, ..., 1
            series.append(Series(
                values=vals,
                unit_id=uid,
                dataset="xjtu",
                channel_names=list(names),
                condition=cond,
                fs=float(self.fs),
                meta={
                    "rul": rul,                  # [n] per-snapshot remaining life
                    "bearing": bearing,
                    "condition": cond,
                    "n_snapshots": n,
                },
            ))

        if dirty:
            self._save_cache(new_cache)
        return series
