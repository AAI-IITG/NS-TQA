"""Learned-STL competitor (Bombara-style decision tree) — WO-2C.

A symbolic-LEARNING baseline, the counterpart to NS-TQA's symbolic-EXECUTION: instead
of executing a given program over perceived predicates, it LEARNS a depth-limited
decision tree over a grid of STL primitives, trained on answer labels (Bombara et al.,
2016). Each tree split is a threshold on a candidate primitive
(channel x window x statistic), so a path from root to leaf is a learned conjunction of
STL-style atoms -- an interpretable formula.

Fairness (avoids reproducing the labeler): the tree is given RAW per-channel/per-window
statistics (mean, max, min, std, slope) and learns its own thresholds via the splits;
it never sees the privileged grounding ``mu*`` that defines the labels. Because the
answer is question-dependent (the same window has different answers for different
programs), the program is encoded symbolically (``encode_program``: predicate mask +
operator counts + window bounds) and concatenated to the signal features -- the same
program information the end-to-end baselines receive.

Depends on scikit-learn (see ``requirements-ext.txt``); CPU-only. Reported as the
symbolic-learning baseline in the C-MAPSS / XJTU tables.
"""
from __future__ import annotations

import numpy as np
import torch

from benchmark.necessity import encode_program

_WINDOWS = (("full", 0.0, 1.0), ("early", 0.0, 0.5), ("late", 0.5, 1.0))


def _channel_window_stats(X: np.ndarray) -> np.ndarray:
    """X[T,C] -> raw STL-primitive ingredients per (channel, window): mean, max, min,
    std, slope. The tree learns thresholds on these (= learned high/low/rising/falling
    with data-chosen cutoffs). Returns a flat feature vector."""
    T, C = X.shape
    feats = []
    t = np.arange(T, dtype=np.float64)
    for _, lo, hi in _WINDOWS:
        a, b = int(lo * T), max(int(hi * T), int(lo * T) + 2)
        seg = X[a:b]                                   # [w, C]
        ts = t[a:b]
        tc = ts - ts.mean()
        denom = (tc ** 2).sum() or 1.0
        slope = (tc[:, None] * (seg - seg.mean(0))).sum(0) / denom   # [C] least-squares slope
        feats.append(seg.mean(0))
        feats.append(seg.max(0))
        feats.append(seg.min(0))
        feats.append(seg.std(0))
        feats.append(slope)
    return np.concatenate(feats).astype(np.float32)    # [5*|windows|*C]


def instance_features(inst, C: int, T: int) -> np.ndarray:
    """[raw signal primitives] ++ [symbolic program encoding]."""
    sig = _channel_window_stats(inst.X.numpy())
    q = encode_program(inst.phi_star, C, T).numpy().astype(np.float32)
    return np.concatenate([sig, q])


def build_xy(instances, C: int, T: int):
    X = np.stack([instance_features(i, C, T) for i in instances])
    y = np.array([int(bool(i.answer_star)) for i in instances])
    return X, y


def train_stl_tree(instances, C: int, T: int, max_depth: int = 6,
                   min_samples_leaf: int = 5, seed: int = 0):
    """Fit a depth-limited decision tree (learned STL formula) on answer labels."""
    from sklearn.tree import DecisionTreeClassifier
    X, y = build_xy(instances, C, T)
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  criterion="gini", random_state=seed)
    tree.fit(X, y)
    return tree


def eval_stl_tree(tree, instances, C: int, T: int) -> dict:
    if not instances:
        return {"n": 0, "answer_accuracy": None, "by_depth": {}}
    from utils.oracle_metrics import group_accuracy
    X, y = build_xy(instances, C, T)
    pred = tree.predict(X).astype(bool)
    gold = y.astype(bool)
    acc = float((pred == gold).mean())
    depth = [i.depth for i in instances]
    by_depth = {int(k): v["accuracy"]
                for k, v in group_accuracy(depth, pred.tolist(), gold.tolist()).items()}
    return {"n": len(instances), "answer_accuracy": acc, "by_depth": by_depth}
