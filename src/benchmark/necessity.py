"""Shared core for the necessity experiment (single source of truth).

Holds the program encoding and the per-(model, seed) train/eval logic so both
``scripts/10_run_necessity.py`` and ``scripts/11_run_necessity_multiseed.py``
use the same code path. ``run_single`` executes one seed end-to-end (every
baseline architecture plus the faithful NS-TQA path) and returns a compact,
JSON-friendly results dict that the multi-seed wrapper aggregates.
"""
from __future__ import annotations

import torch

from benchmark.spurious import balance_report
from executor.grammar import Always, And, Eventually, Node, Not, Or, Predicate, Until
from executor.hard_logic import evaluate
from models.baselines import make_baseline
from models.nstqa_learned import LearnedNSTQA
from perception.grounding import PRED_FAMILIES, predicate_index
from perception.learned import predicate_metrics, train_perception
from utils.oracle_metrics import binary_metrics, group_accuracy

OPS = ["Eventually", "Always", "Until", "And", "Or", "Not"]


# --------------------------------------------------------------------------- #
# program encoding (fair query for end-to-end baselines: the program, not the
# executor); handles arbitrary depth-d trees.
# --------------------------------------------------------------------------- #

def op_depth(node: Node) -> int:
    if isinstance(node, Predicate):
        return 0
    if isinstance(node, (Not, Eventually, Always)):
        return 1 + op_depth(node.child)
    if isinstance(node, (And, Or, Until)):
        return 1 + max(op_depth(node.left), op_depth(node.right))
    raise TypeError(type(node).__name__)


def encode_program(phi: Node, n_channels: int, T: int) -> torch.Tensor:
    op_counts = {o: 0 for o in OPS}
    pred_mask = [0.0] * (len(PRED_FAMILIES) * n_channels)
    a_list, b_list = [], []

    def walk(n: Node) -> None:
        if isinstance(n, Predicate):
            fi = PRED_FAMILIES.index(n.name)
            pred_mask[fi * n_channels + n.channel] = 1.0
            return
        op_counts[type(n).__name__] += 1
        if isinstance(n, (Eventually, Always, Until)):
            a_list.append(n.a)
            b_list.append(n.b)
        for c in n._children():
            walk(c)

    walk(phi)
    denom = max(1, T - 1)
    mean_a = (sum(a_list) / len(a_list) / denom) if a_list else 0.0
    mean_b = (sum(b_list) / len(b_list) / denom) if b_list else 0.0
    feats = (
        [float(op_counts[o]) for o in OPS]
        + pred_mask
        + [float(op_depth(phi)), float(sum(pred_mask)), mean_a, mean_b]
    )
    return torch.tensor(feats, dtype=torch.float32)


def program_query_dim(n_channels: int) -> int:
    return len(OPS) + len(PRED_FAMILIES) * n_channels + 4


# --------------------------------------------------------------------------- #
# end-to-end baseline train / eval (generic over architecture name)
# --------------------------------------------------------------------------- #

def stack_baseline(instances: list, n_channels: int, T: int):
    X = torch.stack([i.X for i in instances])
    q = torch.stack([encode_program(i.phi_star, n_channels, T) for i in instances])
    y = torch.tensor([float(i.answer_star) for i in instances])
    depth = [i.depth for i in instances]
    return X, q, y, depth


def train_baseline(name, train_instances, n_channels, T, cfg, device, seed, verbose=True):
    torch.manual_seed(seed)
    model = make_baseline(
        name, n_channels=n_channels, query_dim=program_query_dim(n_channels),
        hidden=cfg["hidden"], dropout=cfg["dropout"],
        nhead=cfg.get("nhead", 4), n_layers=cfg.get("n_layers", 2),
        kernel=cfg.get("kernel", 3),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    X, q, y, _ = stack_baseline(train_instances, n_channels, T)
    X, q, y = X.to(device), q.to(device), y.to(device)
    n = X.shape[0]
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=device)
        total, correct = 0.0, 0
        for s in range(0, n, cfg["batch_size"]):
            idx = perm[s : s + cfg["batch_size"]]
            opt.zero_grad(set_to_none=True)
            logits = model(X[idx], q[idx])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[idx])
            loss.backward()
            if cfg.get("grad_clip_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            opt.step()
            total += float(loss.detach()) * len(idx)
            correct += int(((logits.sigmoid() >= 0.5) == y[idx].bool()).sum())
        if verbose and (epoch % cfg.get("log_every", 10) == 0 or epoch == cfg["epochs"] - 1):
            print(f"  [{name}] epoch {epoch:3d} bce={total/n:.4f} train_acc={correct/n:.3f}")
    return model


@torch.no_grad()
def eval_baseline(model, instances, n_channels, T, device) -> dict:
    model.eval()
    X, q, y, depth = stack_baseline(instances, n_channels, T)
    logits = model(X.to(device), q.to(device))
    pred = [bool(p) for p in (logits.sigmoid() >= 0.5).cpu().tolist()]
    gold = [bool(g) for g in y.bool().tolist()]
    b = binary_metrics(pred, gold)
    return {
        "n": len(gold),
        "answer_accuracy": sum(p == g for p, g in zip(pred, gold)) / max(1, len(gold)),
        "balanced_accuracy": b["balanced_accuracy"],
        "f1": b["f1"],
        "by_depth": _by_depth(depth, pred, gold),
    }


# --------------------------------------------------------------------------- #
# NS-TQA (faithful) eval + references
# --------------------------------------------------------------------------- #

def eval_nstqa(nst: LearnedNSTQA, instances) -> dict:
    out = nst.evaluate_instances(instances)
    recs = out["records"]
    pred = [r["pred_answer"] for r in recs]
    gold = [r["gold_answer"] for r in recs]
    depth = [r["depth"] for r in recs]
    b = binary_metrics(pred, gold)
    return {
        "n": len(gold),
        "answer_accuracy": out["answer_accuracy"],
        "balanced_accuracy": b["balanced_accuracy"],
        "f1": b["f1"],
        "by_depth": _by_depth(depth, pred, gold),
    }


def oracle_accuracy(instances, pidx) -> dict:
    pred = [bool(evaluate(i.mu_star, i.phi_star, pidx, 0)[0] > 0) for i in instances]
    gold = [bool(i.answer_star) for i in instances]
    return {
        "n": len(gold),
        "answer_accuracy": sum(p == g for p, g in zip(pred, gold)) / max(1, len(gold)),
        "by_depth": _by_depth([i.depth for i in instances], pred, gold),
    }


def _by_depth(depth, pred, gold) -> dict:
    """{int depth -> accuracy} from group_accuracy's {str -> {accuracy,n}}."""
    ga = group_accuracy(depth, pred, gold)
    return {int(k): v["accuracy"] for k, v in ga.items()}


# --------------------------------------------------------------------------- #
# one full seed
# --------------------------------------------------------------------------- #

def run_single(bm: dict, cfg: dict, seed: int, baseline_names: list[str],
               verbose: bool = False) -> dict:
    """Train every baseline + NS-TQA for ONE seed; return a results dict.

    Structure:
      { "<baseline>": {"indist": {...}, "shift": {...}}, ...,
        "NS-TQA":     {"indist": {...}, "shift": {...}},
        "oracle":     {"indist": acc-dict, "shift": acc-dict},
        "spurious_shortcut": {"indist": acc, "shift": acc},
        "perception_f1":     {"indist": f1, "shift": f1} }
    """
    from benchmark.baseline import select_device

    meta = bm["meta"]
    C, T = meta["n_channels"], meta["T"]
    device = select_device(cfg.get("device", "cpu"))
    pidx = predicate_index(C)
    res: dict = {}

    # A single-operating-condition dataset (e.g. only one XJTU load on disk) has no
    # shifted regime; the held-out-UNIT generalization (test_indist) is still valid.
    # When test_shift is empty, report shift metrics as None rather than evaluating
    # on an empty set. Downstream mean_std() filters None -> NaN.
    has_shift = bool(bm["test_shift"])
    empty_eval = {"n": 0, "answer_accuracy": None, "balanced_accuracy": None,
                  "f1": None, "by_depth": {}}

    # end-to-end baselines
    for name in baseline_names:
        model = train_baseline(name, bm["train"], C, T, cfg["lstm"], device, seed, verbose)
        res[name] = {
            "indist": eval_baseline(model, bm["test_indist"], C, T, device),
            "shift": eval_baseline(model, bm["test_shift"], C, T, device) if has_shift
                     else dict(empty_eval),
        }

    # faithful NS-TQA (one perception net per seed)
    pres = train_perception(
        bm["train"], n_channels=C, hidden=cfg["perception"]["hidden"],
        kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
        per_channel=cfg["perception"]["per_channel"], epochs=cfg["perception"]["epochs"],
        batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
        weight_decay=cfg["perception"]["weight_decay"],
        device_pref=cfg.get("device", "cpu"), seed=seed,
        log_every=cfg["perception"].get("log_every", 20), verbose=verbose,
    )
    nst = LearnedNSTQA(pres.model, n_channels=C)
    res["NS-TQA"] = {
        "indist": eval_nstqa(nst, bm["test_indist"]),
        "shift": eval_nstqa(nst, bm["test_shift"]) if has_shift else dict(empty_eval),
    }

    # references
    res["oracle"] = {
        "indist": oracle_accuracy(bm["test_indist"], pidx),
        "shift": oracle_accuracy(bm["test_shift"], pidx) if has_shift
                 else {"n": 0, "answer_accuracy": None, "by_depth": {}},
    }
    
    # spurious-shortcut reference is only defined for planted-spurious benchmarks;
    # real-data benchmarks have no spurious channel, so report None there.
    def _has_spurious(insts):
        return bool(insts) and getattr(insts[0], "spurious_channel", None) is not None
    if _has_spurious(bm["test_indist"]):
        res["spurious_shortcut"] = {
            "indist": balance_report(bm["test_indist"])["spurious_shortcut_acc"],
            "shift": balance_report(bm["test_shift"])["spurious_shortcut_acc"],
        }
    else:
        res["spurious_shortcut"] = {"indist": None, "shift": None}

    res["perception_f1"] = {
        "indist": predicate_metrics(pres.model, bm["test_indist"])["macro_f1"],
        "shift": predicate_metrics(pres.model, bm["test_shift"])["macro_f1"]
                 if has_shift else None,
    }
    return res