"""NASA IMS run-to-failure bearing adapter -> per-bearing HI Series.

Layout (under ``data/raw/NASA_IMS_Bearings/``)::

    1st_test/            <timestamp> files   (8 columns = 4 bearings x 2 accels)
    2nd_test/            <timestamp> files   (4 columns = 4 bearings x 1 accel)
    3rd_test/4th_test/txt/  <timestamp> files (4 columns = 4 bearings x 1 accel)

Each file is one ~1 s vibration snapshot (20,480 samples @ 20 kHz), tab-separated,
no header. File names are timestamps (``YYYY.MM.DD.HH.MM.SS``) that sort
chronologically, so the ordered files are the degradation trajectory. Each of the
3 test runs is a *different rig setup* (we treat each as one operating
``condition`` ``ims1|ims2|ims3``); within a run, 4 bearings share the shaft.

To match the unified 14-channel HI vocabulary used by XJTU and FEMTO, every
bearing is reduced to a 2-axis ``[N,2]`` snapshot before
``perception.vibration.snapshot_features``:

* test 1 has 2 accelerometers per bearing -> use both axes directly;
* tests 2 & 3 have 1 accelerometer per bearing -> **duplicate the single axis**
  (H = V) so the channel count stays 14. The V-features are then redundant copies
  (harmless: per-channel perception and the executor use whatever channels a
  program references). This is a deliberate modelling choice for cross-rig shape
  consistency, recorded in ``meta``.

RUL note: only some IMS bearings actually fail (run 1: B3, B4; run 2: B1; run 3:
B3). We store ``meta["rul"] = n - k`` as an **end-of-run proxy** for every bearing
and flag true-failure bearings in ``meta["fails"]``; the proxy is fine for the
predicate-transfer question (which does not use RUL) but should not be read as
true remaining life for non-failing bearings.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from perception.vibration import feature_names, snapshot_features

from .adapters import DatasetAdapter, Series

# condition name -> (relative snapshot dir, set of bearings that truly fail)
IMS_TESTS: dict[str, tuple[str, set[int]]] = {
    "ims1": ("1st_test", {3, 4}),
    "ims2": ("2nd_test", {1}),
    "ims3": ("3rd_test/4th_test/txt", {3}),
}
N_BEARINGS = 4

_DEFAULT_ROOT = "data/raw/NASA_IMS_Bearings"
_DEFAULT_CACHE = "data/processed/ims_hi.pkl"


def _snapshot_files(snap_dir: Path) -> list[Path]:
    """Chronologically-ordered snapshot files (timestamp names, no extension)."""
    files = [p for p in snap_dir.iterdir()
             if p.is_file() and p.suffix.lower() != ".pdf"]
    return sorted(files, key=lambda p: p.name)


def _bearing_axes(arr: np.ndarray, bearing: int, n_cols: int) -> np.ndarray:
    """Slice one bearing's [N,2] from a snapshot [N, n_cols].

    n_cols==8 -> 2 axes/bearing (cols 2b, 2b+1); n_cols==4 -> 1 axis/bearing
    (col b), duplicated to 2.
    """
    per = n_cols // N_BEARINGS
    if per >= 2:
        c = (bearing - 1) * per
        return arr[:, [c, c + 1]]
    col = arr[:, bearing - 1]
    return np.stack([col, col], axis=1)        # duplicate single axis


def _test_trajectories(snap_dir: Path, fs: int, n_bands: int
                        ) -> tuple[dict[int, np.ndarray], list[str]]:
    """Build all 4 bearings' HI trajectories for one test, reading each file once."""
    files = _snapshot_files(snap_dir)
    if not files:
        raise FileNotFoundError(f"no snapshot files in {snap_dir}")
    n_cols = np.loadtxt(files[0]).shape[1]
    rows: dict[int, list] = {b: [] for b in range(1, N_BEARINGS + 1)}
    names = feature_names(n_bands)
    for f in files:
        arr = np.loadtxt(f)                     # [N, n_cols] (tab/whitespace)
        for b in range(1, N_BEARINGS + 1):
            feats, names = snapshot_features(_bearing_axes(arr, b, n_cols),
                                             fs=fs, n_bands=n_bands)
            rows[b].append(feats)
    trajs = {b: np.stack(rows[b], axis=0) for b in rows}
    return trajs, names


class IMSAdapter(DatasetAdapter):
    name = "ims"
    split_axis = "condition"   # condition = test rig ims1|ims2|ims3

    def __init__(
        self,
        root: str | Path = _DEFAULT_ROOT,
        tests: Optional[tuple[str, ...]] = None,
        fs: int = 20480,
        n_bands: int = 3,
        cache_path: str | Path = _DEFAULT_CACHE,
        force_recompute: bool = False,
    ):
        self.root = Path(root)
        self._tests = tuple(tests) if tests is not None else tuple(IMS_TESTS)
        self.fs = fs
        self.n_bands = n_bands
        self.cache_path = Path(cache_path)
        self.force_recompute = force_recompute

    # -- cache helpers (mirror XJTU/FEMTO) ------------------------------------ #

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
        os.replace(tmp, self.cache_path)

    # -- main ----------------------------------------------------------------- #

    def load(self) -> list[Series]:
        names = feature_names(self.n_bands)
        cache = self._load_cache()
        series: list[Series] = []
        new_cache = dict(cache)
        dirty = False

        any_test = False
        for test in self._tests:
            rel, fails = IMS_TESTS[test]
            snap_dir = self.root / rel
            if not snap_dir.is_dir():
                continue
            any_test = True
            sig = len(_snapshot_files(snap_dir))            # #snapshots -> cache key

            # need a recompute if any bearing of this test is missing/stale
            need = any(cache.get(f"ims:{test}:Bearing{b}", {}).get("sig") != sig
                       for b in range(1, N_BEARINGS + 1))
            if need:
                trajs, names = _test_trajectories(snap_dir, self.fs, self.n_bands)
                for b in range(1, N_BEARINGS + 1):
                    new_cache[f"ims:{test}:Bearing{b}"] = {"sig": sig, "values": trajs[b]}
                dirty = True

            for b in range(1, N_BEARINGS + 1):
                uid = f"ims:{test}:Bearing{b}"
                values = new_cache[uid]["values"]
                vals = torch.as_tensor(values, dtype=torch.float32)
                n = vals.shape[0]
                rul = torch.arange(n, 0, -1, dtype=torch.float32)
                series.append(Series(
                    values=vals, unit_id=uid, dataset="ims",
                    channel_names=list(names), condition=test, fs=float(self.fs),
                    meta={
                        "rul": rul,                 # end-of-run PROXY (see module docstring)
                        "bearing": f"Bearing{b}",
                        "condition": test,
                        "n_snapshots": n,
                        "rig": "ims",
                        "fails": b in fails,
                        "rul_is_proxy": True,
                    },
                ))

        if not any_test:
            if cache:
                print(f"[IMSAdapter] raw data absent under {self.root}; "
                      f"loading from cache {self.cache_path}", flush=True)
                return self._series_from_cache(cache, names)
            raise SystemExit(f"no IMS test directories found under {self.root}")
        if dirty:
            self._save_cache(new_cache)
        return series

    # -- cache-only reconstruction (portable: no raw tree needed) ------------- #

    def _series_from_cache(self, cache: dict, names: list) -> list[Series]:
        """Rebuild Series purely from the HI cache when raw data is absent."""
        out: list[Series] = []
        for uid, entry in sorted(cache.items()):
            _, test, bearing = uid.split(":", 2)     # "ims:{test}:Bearing{b}"
            if test not in self._tests:
                continue
            b = int(bearing.replace("Bearing", ""))
            fails = IMS_TESTS.get(test, (None, ()))[1]
            vals = torch.as_tensor(entry["values"], dtype=torch.float32)
            n = vals.shape[0]
            out.append(Series(
                values=vals, unit_id=uid, dataset="ims", channel_names=list(names),
                condition=test, fs=float(self.fs),
                meta={"rul": torch.arange(n, 0, -1, dtype=torch.float32),
                      "bearing": bearing, "condition": test, "n_snapshots": n,
                      "rig": "ims", "fails": b in fails, "rul_is_proxy": True}))
        return out
