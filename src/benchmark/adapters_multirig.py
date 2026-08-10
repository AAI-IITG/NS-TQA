"""Combine several bearing-rig adapters into one, with RIG as the condition.

For the cross-rig transfer experiment we want to train on one machine (rig) and
test on a physically different one. Every bearing-rig adapter (XJTU, FEMTO, IMS)
already emits the *same* 14-channel HI vocabulary via the unified vibration
front-end, so the only thing needed is to relabel each unit's ``condition`` with
the rig it came from. Then the standard cross-condition pipeline
``build_real_benchmark(shift="condition", train_conditions=[rigA],
test_conditions=[rigB, ...])`` becomes a cross-rig builder with no new engine.

``MultiRigAdapter`` concatenates the children's ``load()`` outputs, re-tags
``condition = rig name`` (the child's ``dataset``: ``xjtu`` / ``femto`` / ``ims``),
preserves the already-namespaced ``unit_id``, and **verifies the channel
vocabularies match** across rigs (otherwise transfer is meaningless).
"""
from __future__ import annotations

from .adapters import DatasetAdapter, Series


class MultiRigAdapter(DatasetAdapter):
    name = "multirig"
    split_axis = "condition"   # condition == rig name (xjtu|femto|ims)

    def __init__(self, adapters: list[DatasetAdapter]):
        if not adapters:
            raise ValueError("MultiRigAdapter needs at least one child adapter")
        self.adapters = list(adapters)

    def load(self) -> list[Series]:
        out: list[Series] = []
        ref_names: list[str] | None = None
        for adapter in self.adapters:
            for s in adapter.load():
                rig = s.dataset                      # "xjtu" | "femto" | "ims"
                if ref_names is None:
                    ref_names = list(s.channel_names)
                elif list(s.channel_names) != ref_names:
                    raise ValueError(
                        f"channel vocabulary mismatch: rig {rig!r} has "
                        f"{s.channel_names} but expected {ref_names}; all rigs must "
                        "share the same HI front-end (same n_bands) for transfer.")
                out.append(Series(
                    values=s.values, unit_id=s.unit_id, dataset=s.dataset,
                    channel_names=s.channel_names, condition=rig, fs=s.fs,
                    meta={**s.meta, "rig": rig, "orig_condition": s.condition},
                ))
        return out
