"""C-MAPSS adapter: wrap the tested loader into the frozen DatasetAdapter protocol.

Produces one ``Series`` per engine (``unit_id`` namespaced by FD subset), with
``condition`` set to the subset so the cross-condition split (train FD001 ->
test FD002/FD004) and the held-out-engine split both work off the shared
pipeline in ``benchmark.adapters``.

Two things this adapter adds on top of ``benchmark.cmapss.load_cmapss``:

* **Degenerate-channel dropping.** ``load_cmapss`` drops *flat* sensors by std,
  but a sensor like s6 is not globally flat -- it has rare excursions that lift
  its std above the threshold while the bulk is pinned to a single value, so its
  calibration band collapses (hi == lo) and ``high``/``low`` become degenerate.
  We additionally drop channels whose train quantile span ``q_hi - q_lo`` is
  below ``min_qspan`` (computed on the standardized trajectories).

* **Consistent channel set across subsets.** Channel selection is computed once
  on the FIRST subset loaded (intended to be the train subset, e.g. FD001) and
  reused for every later subset, so the train and shifted-test signals share an
  identical channel layout. Pass ``keep_channels`` explicitly to pin it.

Note on normalization: ``load_cmapss`` already standardizes per channel; the
non-circular build re-fits a ``ChannelNormalizer`` on TRAIN windows on top of
this, which yields a correct train-only normalization. The only all-data step
left is channel *selection* (flat + degenerate), which is a fixed sensor
property of the dataset, not a label-dependent leak.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .adapters import DatasetAdapter, Series
from .cmapss import load_cmapss


class CMAPSSAdapter(DatasetAdapter):
    name = "cmapss"
    split_axis = "condition"   # primary: FD001 -> FD002/4; "unit" also valid (held-out engine)

    def __init__(
        self,
        root: str | Path,
        subsets: tuple[str, ...] = ("FD001",),
        rul_cap: int = 125,
        flat_std_thresh: float = 1e-6,
        min_qspan: float = 0.05,         # drop channels whose train q_hi-q_lo span is tiny (kills s6)
        hi_q: float = 0.85,
        lo_q: float = 0.15,
        keep_channels: Optional[list[str]] = None,
        namespace_units: bool = True,
        op_normalize: bool = False,
        n_regimes: int = 6,
    ):
        self.root = Path(root)
        self.subsets = tuple(subsets)
        self.rul_cap = rul_cap
        self.flat_std_thresh = flat_std_thresh
        self.min_qspan = min_qspan
        self.hi_q = hi_q
        self.lo_q = lo_q
        self.keep_channels = list(keep_channels) if keep_channels is not None else None
        self.namespace_units = namespace_units
        self.op_normalize = op_normalize
        self.n_regimes = n_regimes

    def _nondegenerate_channels(self, ds) -> list[str]:
        """Channels whose standardized train quantile span is non-trivial."""
        allvals = torch.cat(ds.trajectories, dim=0)            # [sum T_i, C]
        qhi = torch.quantile(allvals, self.hi_q, dim=0)
        qlo = torch.quantile(allvals, self.lo_q, dim=0)
        span = (qhi - qlo).abs()
        return [n for n, s in zip(ds.channel_names, span.tolist()) if s >= self.min_qspan]

    def load(self) -> list[Series]:
        series: list[Series] = []
        keep = self.keep_channels                               # pinned across subsets once set
        dropped_degenerate: list[str] = []

        for subset in self.subsets:
            ds = load_cmapss(self.root, subset=subset, rul_cap=self.rul_cap,
                             flat_std_thresh=self.flat_std_thresh,
                             op_normalize=self.op_normalize, n_regimes=self.n_regimes)
            if keep is None:
                nondeg = self._nondegenerate_channels(ds)
                dropped_degenerate = [c for c in ds.channel_names if c not in nondeg]
                keep = nondeg
            keep_idx = [ds.channel_names.index(n) for n in keep if n in ds.channel_names]
            kept_names = [ds.channel_names[i] for i in keep_idx]

            for eid, traj, rul in zip(ds.engine_ids, ds.trajectories, ds.rul):
                uid = f"{subset}:{eid}" if self.namespace_units else str(eid)
                series.append(Series(
                    values=traj[:, keep_idx].contiguous(),
                    unit_id=uid,
                    dataset="cmapss",
                    channel_names=kept_names,
                    condition=subset,
                    fs=None,
                    meta={
                        "rul": rul,                 # [T_i] per-cycle RUL (for RUL-grounded QA later)
                        "engine_id": eid,
                        "subset": subset,
                        "dropped_flat": ds.dropped_channels,
                        "dropped_degenerate": dropped_degenerate,
                    },
                ))
        return series