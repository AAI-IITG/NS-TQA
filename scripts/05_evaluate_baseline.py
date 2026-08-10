"""05 - Evaluate the direct-answer LSTM baseline.

Run:  python scripts/05_evaluate_baseline.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
from collections import defaultdict
from functools import partial

import torch
from torch.utils.data import DataLoader, Subset

from benchmark.baseline import baseline_collate, load_qa_pickle, query_dim, select_device
from benchmark.dataloaders import QADataset
from models.baselines import LSTMBaseline


def _accuracy(pred: list[bool], gold: list[bool]) -> float:
    return sum(p == g for p, g in zip(pred, gold)) / max(1, len(gold))


def _binary_metrics(pred: list[bool], gold: list[bool]) -> dict:
    tp = sum(p and g for p, g in zip(pred, gold))
    tn = sum((not p) and (not g) for p, g in zip(pred, gold))
    fp = sum(p and (not g) for p, g in zip(pred, gold))
    fn = sum((not p) and g for p, g in zip(pred, gold))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "majority_no_accuracy": sum(not g for g in gold) / max(1, len(gold)),
    }


def _group_accuracy(keys: list[str], pred: list[bool], gold: list[bool]) -> dict:
    groups = defaultdict(lambda: {"correct": 0, "n": 0})
    for key, p, g in zip(keys, pred, gold):
        if key is None:
            continue
        groups[str(key)]["correct"] += int(p == g)
        groups[str(key)]["n"] += 1
    return {
        key: {"accuracy": val["correct"] / max(1, val["n"]), "n": val["n"]}
        for key, val in sorted(groups.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "runs" / "baseline" / "cmapss_FD001" / "checkpoint.pt"),
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    dataset_path = ROOT / ckpt["dataset_path"]
    artifact = load_qa_pickle(dataset_path)
    instances = artifact["instances"]
    if ckpt["config"].get("max_instances") is not None:
        instances = instances[: int(ckpt["config"]["max_instances"])]

    if args.split == "all":
        indices = list(range(len(instances)))
    else:
        indices = ckpt["splits"][args.split]

    device = select_device(ckpt["config"].get("device", "cpu"))
    model = LSTMBaseline(
        n_channels=ckpt["n_channels"],
        query_dim=query_dim(),
        hidden=ckpt["hidden"],
        dropout=ckpt.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = QADataset(instances)
    collate = partial(baseline_collate, n_channels=ckpt["n_channels"], T=ckpt["T"])
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=ckpt["config"]["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )

    total_loss = 0.0
    pred_all: list[bool] = []
    gold_all: list[bool] = []
    template_all: list[str] = []
    engine_all: list[str | None] = []
    bearing_all: list[str | None] = []

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            q = batch["q"].to(device)
            y = batch["answer_star"].to(device)
            logits = model(X, q)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            pred = (logits.sigmoid() >= 0.5).cpu().tolist()
            gold = y.bool().cpu().tolist()
            total_loss += float(loss.item()) * len(gold)
            pred_all.extend(bool(x) for x in pred)
            gold_all.extend(bool(x) for x in gold)
            template_all.extend(batch["template"])
            engine_all.extend(batch["engine_id"])
            bearing_all.extend(batch["bearing_id"])

    metrics = {
        "dataset_path": ckpt["dataset_path"],
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "split_meta": ckpt.get("split_meta", {"split_mode": "random"}),
        "n": len(gold_all),
        "loss": total_loss / max(1, len(gold_all)),
        "answer_accuracy": _accuracy(pred_all, gold_all),
        "gold_yes_frac": sum(gold_all) / max(1, len(gold_all)),
        "pred_yes_frac": sum(pred_all) / max(1, len(pred_all)),
        "binary": _binary_metrics(pred_all, gold_all),
        "by_template": _group_accuracy(template_all, pred_all, gold_all),
        "by_engine_id": _group_accuracy(engine_all, pred_all, gold_all),
        "by_bearing_id": _group_accuracy(bearing_all, pred_all, gold_all),
    }

    out_path = checkpoint_path.parent / f"metrics_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"saved metrics to {out_path}")


if __name__ == "__main__":
    main()
