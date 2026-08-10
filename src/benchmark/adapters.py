"""Stage-1 core: turn any dataset into modeling-ready windows + QA instances.

This module is dataset-agnostic. Each concrete dataset implements ``DatasetAdapter``
and returns a list of ``Series`` (one per unit -- engine, bearing, client, ...),
already past any per-dataset feature front-end (e.g. raw vibration -> RMS/kurtosis
health indicators). Everything downstream is shared:

    adapter.load() -> [Series]            (per-unit, modeling-ready channels)
        -> window_series(...)             -> [Window]   (fixed-T windows + provenance)
        -> a split strategy               -> assign each unit/window to train/val/test
        -> ChannelNormalizer.fit(train)   -> z-score using TRAIN statistics ONLY
        -> (QA construction, Stage 2)     -> [RealInstance]

``RealInstance`` deliberately exposes the SAME attribute names the experiment
engine already reads (``X, phi_star, answer_star, depth, mu_star, question``),
so real-data instances drop straight into ``benchmark.necessity.run_single``,
``models.nstqa_learned.LearnedNSTQA``, and ``perception.learned.train_perception``
without changing the engine.

The three split strategies correspond to the shift axes frozen in the Stage-0
table: held-out unit (C-MAPSS engines, electricity clients, bearings),
cross-condition (XJTU A->B, C-MAPSS FD subsets), and temporal (ETT early->late).
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch

from .templates import Question  # RealInstance carries phi_star as Question.program


# --------------------------------------------------------------------------- #
# data containers
# --------------------------------------------------------------------------- #

@dataclass
class Series:
    """One unit's full-resolution, modeling-ready trajectory."""
    values: torch.Tensor              # [L, C]
    unit_id: str
    dataset: str
    channel_names: list[str]
    condition: Optional[str] = None   # operating condition / subset / regime
    fs: Optional[float] = None        # sampling rate (Hz), if meaningful
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.values.ndim != 2:
            raise ValueError(f"Series.values must be [L,C], got {tuple(self.values.shape)}")
        if len(self.channel_names) != self.values.shape[1]:
            raise ValueError(
                f"channel_names ({len(self.channel_names)}) != C ({self.values.shape[1]})"
            )

    @property
    def L(self) -> int:
        return self.values.shape[0]

    @property
    def C(self) -> int:
        return self.values.shape[1]


@dataclass
class Window:
    """A fixed-length window cut from one unit, with provenance for splitting."""
    X: torch.Tensor                   # [T, C]
    unit_id: str
    dataset: str
    channel_names: list[str]
    start: int                        # start index within the unit
    condition: Optional[str] = None
    meta: dict = field(default_factory=dict)


@dataclass
class RealInstance:
    """A QA instance over real (or semi-synthetic) signal.

    Attribute names match what ``benchmark.necessity`` / ``LearnedNSTQA`` /
    ``train_perception`` read, so these are drop-in for the existing engine.
    ``mu_star`` (privileged predicate truths) is filled by the QA builder in
    Stage 2; until then it may be None for instances used only by end-to-end
    baselines.
    """
    X: torch.Tensor                                   # [T, C] model's (normalized) view
    question: Question                                # carries phi_star as .program
    phi_star: object                                  # executor Node
    answer_star: bool
    depth: int
    unit_id: str
    n_channels: int
    condition: Optional[str] = None
    regime: str = "test"                              # train | val | test | shift
    mu_star: Optional[torch.Tensor] = None            # [T, 4C] privileged truths
    rho_star: Optional[float] = None
    spurious_channel: Optional[int] = None            # set if semi-synthetic
    provenance: dict = field(default_factory=dict)    # dataset, start, channel_names


# --------------------------------------------------------------------------- #
# adapter protocol
# --------------------------------------------------------------------------- #

class DatasetAdapter(ABC):
    """Parse one dataset into per-unit, modeling-ready Series.

    Concrete adapters apply any dataset-specific feature front-end inside
    ``load`` (tabular datasets pass channels through; vibration datasets convert
    raw accelerometer streams into health-indicator channels). The rest of the
    pipeline never sees raw formats.
    """

    name: str = "abstract"

    @abstractmethod
    def load(self) -> list[Series]:
        """Return one Series per unit, channels already in modeling-ready form."""
        raise NotImplementedError

    def conditions(self, series: list[Series]) -> list[str]:
        """Distinct operating conditions present (for cross-condition splits)."""
        return sorted({s.condition for s in series if s.condition is not None})


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #

def window_series(
    series: Series,
    T: int,
    stride: int,
    max_windows_per_unit: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> list[Window]:
    """Sliding windows of length T over one unit. Optionally subsample windows."""
    L = series.L
    if L < T:
        return []
    starts = list(range(0, L - T + 1, stride))
    if max_windows_per_unit is not None and len(starts) > max_windows_per_unit:
        rng = rng or random.Random(0)
        starts = sorted(rng.sample(starts, max_windows_per_unit))
    out = []
    for s in starts:
        out.append(Window(
            X=series.values[s : s + T].clone(),
            unit_id=series.unit_id, dataset=series.dataset,
            channel_names=series.channel_names, start=s, condition=series.condition,
        ))
    return out


def window_all(
    series_list: list[Series], T: int, stride: int,
    max_windows_per_unit: Optional[int] = None, seed: int = 0,
) -> list[Window]:
    rng = random.Random(seed)
    out: list[Window] = []
    for s in series_list:
        out.extend(window_series(s, T, stride, max_windows_per_unit, rng))
    return out


def subsample_channels(series: Series, k: int, rng: random.Random) -> Series:
    """Keep k channels (for very high-dimensional datasets, e.g. Traffic)."""
    C = series.C
    if k >= C:
        return series
    idx = sorted(rng.sample(range(C), k))
    return Series(
        values=series.values[:, idx], unit_id=series.unit_id, dataset=series.dataset,
        channel_names=[series.channel_names[i] for i in idx],
        condition=series.condition, fs=series.fs, meta={**series.meta, "channel_idx": idx},
    )


# --------------------------------------------------------------------------- #
# train-only normalization
# --------------------------------------------------------------------------- #

class ChannelNormalizer:
    """Per-channel z-score. Fit on TRAIN windows only; apply to all."""

    def __init__(self, eps: float = 1e-6):
        self.mean: Optional[torch.Tensor] = None  # [C]
        self.std: Optional[torch.Tensor] = None   # [C]
        self.eps = eps

    def fit(self, train_windows: list[Window]) -> "ChannelNormalizer":
        if not train_windows:
            raise ValueError("cannot fit normalizer on zero train windows")
        stacked = torch.cat([w.X for w in train_windows], dim=0)  # [N*T, C]
        self.mean = stacked.mean(dim=0)
        self.std = stacked.std(dim=0).clamp_min(self.eps)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            raise RuntimeError("ChannelNormalizer not fit")
        return (X - self.mean) / self.std

    def transform_window(self, w: Window) -> Window:
        return Window(
            X=self.transform(w.X), unit_id=w.unit_id, dataset=w.dataset,
            channel_names=w.channel_names, start=w.start, condition=w.condition, meta=w.meta,
        )


# --------------------------------------------------------------------------- #
# split strategies (one per Stage-0 shift axis)
# --------------------------------------------------------------------------- #

def split_by_unit(
    unit_ids: list[str], fracs=(0.7, 0.15, 0.15), seed: int = 0
) -> dict[str, str]:
    """Held-out-unit split: each unique unit_id -> one of train/val/test."""
    units = sorted(set(unit_ids))
    rng = random.Random(seed)
    rng.shuffle(units)
    n = len(units)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    assign = {}
    for i, u in enumerate(units):
        assign[u] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return assign


def split_by_condition(
    train_conditions: list[str], test_conditions: list[str]
) -> dict[str, str]:
    """Cross-condition split: map condition label -> split. val drawn from train side."""
    assign = {c: "train" for c in train_conditions}
    for c in test_conditions:
        assign[c] = "test"
    return assign


def split_by_time(L: int, T: int, fracs=(0.7, 0.15, 0.15)) -> dict[str, tuple[int, int]]:
    """Temporal split (ETT early->late): time-index boundaries for window STARTS.

    Returns {split: (start_lo, start_hi)} on the window-start axis so that a
    window belongs to a split iff its start falls in that half-open range. Test
    windows therefore come strictly after train windows in time.
    """
    last_start = max(0, L - T)
    b1 = int(round(fracs[0] * last_start))
    b2 = int(round((fracs[0] + fracs[1]) * last_start))
    return {"train": (0, b1), "val": (b1, b2), "test": (b2, last_start + 1)}


# --------------------------------------------------------------------------- #
# applying splits to windows
# --------------------------------------------------------------------------- #

def partition_by_unit(windows: list[Window], assign: dict[str, str]) -> dict[str, list[Window]]:
    out: dict[str, list[Window]] = {"train": [], "val": [], "test": []}
    for w in windows:
        s = assign.get(w.unit_id)
        if s is not None:
            out[s].append(w)
    return out


def partition_by_condition(windows: list[Window], assign: dict[str, str]) -> dict[str, list[Window]]:
    out: dict[str, list[Window]] = {"train": [], "val": [], "test": []}
    for w in windows:
        s = assign.get(w.condition)
        if s is not None:
            out[s].append(w)
    return out


def partition_by_time(windows: list[Window], bounds: dict[str, tuple[int, int]]) -> dict[str, list[Window]]:
    out: dict[str, list[Window]] = {"train": [], "val": [], "test": []}
    for w in windows:
        for s, (lo, hi) in bounds.items():
            if lo <= w.start < hi:
                out[s].append(w)
                break
    return out