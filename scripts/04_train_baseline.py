"""04 - Train the direct-answer LSTM baseline.

This is an ablation path only: it maps (signal, encoded structured query) to
answer_star directly and must not be used as the faithful NS-TQA answer path.

Run:  python scripts/04_train_baseline.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
import random
import time
from functools import partial

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from benchmark.baseline import (
    baseline_collate,
    group_split_indices,
    load_qa_pickle,
    query_dim,
    select_device,
    split_indices,
)
from benchmark.dataloaders import QADataset
from models.baselines import LSTMBaseline


def _dataset_tag(path: Path) -> str:
    return path.stem.removesuffix("_qa")


def _make_splits(instances: list, cfg: dict):
    split_mode = cfg.get("split_mode", "random")
    if split_mode == "random":
        train_idx, val_idx, test_idx = split_indices(
            len(instances), cfg["train_frac"], cfg["val_frac"], cfg["seed"]
        )
        return train_idx, val_idx, test_idx, {"split_mode": "random"}
    if split_mode == "group":
        group_key = cfg.get("group_key")
        if not group_key:
            raise SystemExit("group split requires baseline.yaml group_key")
        train_idx, val_idx, test_idx, meta = group_split_indices(
            instances, group_key, cfg["train_frac"], cfg["val_frac"], cfg["seed"]
        )
        meta["split_mode"] = "group"
        return train_idx, val_idx, test_idx, meta
    raise SystemExit(f"unknown split_mode={split_mode!r}")


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _loss_and_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float]:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    pred = logits.sigmoid() >= 0.5
    acc = (pred == labels.bool()).float().mean().item()
    return loss, acc


def _run_epoch(
    model, loader, device, optimizer=None, desc: str = "epoch", grad_clip_norm=None
) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total = 0
    progress = tqdm(loader, desc=desc, leave=False, disable=not sys.stderr.isatty())
    for batch in progress:
        X = batch["X"].to(device)
        q = batch["q"].to(device)
        y = batch["answer_star"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(X, q)
        loss, _ = _loss_and_accuracy(logits, y)
        if training:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        with torch.no_grad():
            pred = logits.sigmoid() >= 0.5
            total_correct += int((pred == y.bool()).sum().item())
            total_loss += float(loss.item()) * y.numel()
            total += y.numel()
            progress.set_postfix(
                loss=f"{total_loss / max(1, total):.4f}",
                acc=f"{total_correct / max(1, total):.3f}",
            )
    return {"loss": total_loss / max(1, total), "accuracy": total_correct / max(1, total)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--split-mode", choices=["random", "group"], default=None)
    parser.add_argument("--group-key", default=None)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    if args.dataset_path is not None:
        cfg["dataset_path"] = args.dataset_path
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.max_instances is not None:
        cfg["max_instances"] = args.max_instances
    if args.split_mode is not None:
        cfg["split_mode"] = args.split_mode
    if args.group_key is not None:
        cfg["group_key"] = args.group_key

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    dataset_path = ROOT / cfg["dataset_path"]
    artifact = load_qa_pickle(dataset_path)
    instances = artifact["instances"]
    if cfg.get("max_instances") is not None:
        instances = instances[: int(cfg["max_instances"])]
    if not instances:
        raise SystemExit(f"no instances loaded from {dataset_path}")

    first_x = instances[0]["X"] if isinstance(instances[0], dict) else instances[0].X
    T, C = first_x.shape
    train_idx, val_idx, test_idx, split_meta = _make_splits(instances, cfg)

    ds = QADataset(instances)
    collate = partial(baseline_collate, n_channels=C, T=T)
    train_loader = DataLoader(
        Subset(ds, train_idx), batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        Subset(ds, val_idx), batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate
    )

    device = select_device(cfg.get("device", "cpu"))
    model = LSTMBaseline(
        n_channels=C,
        query_dim=query_dim(),
        hidden=cfg["hidden"],
        dropout=cfg.get("dropout", 0.0),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0)
    )

    tag = _dataset_tag(dataset_path)
    if cfg.get("split_mode", "random") == "group":
        tag = f"{tag}_group_{cfg['group_key']}"
    if cfg.get("max_instances") is not None:
        tag = f"{tag}_n{int(cfg['max_instances'])}"
    run_dir = ROOT / cfg["run_root"] / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"training LSTM baseline on {dataset_path}")
    tqdm.write(
        f"  instances={len(instances)} train={len(train_idx)} "
        f"val={len(val_idx)} test={len(test_idx)}"
    )
    tqdm.write(f"  split_mode={split_meta['split_mode']} group_key={split_meta.get('group_key')}")
    if split_meta["split_mode"] == "group":
        tqdm.write(
            f"  groups train={split_meta['train_groups']} "
            f"val={split_meta['val_groups']} test={split_meta['test_groups']}"
        )
    tqdm.write(f"  T={T} C={C} query_dim={query_dim()} device={device}")

    history = []
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    min_delta = cfg.get("early_stop_min_delta", 0.0)
    patience = cfg.get("early_stop_patience")
    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            desc=f"epoch {epoch:03d} train",
            grad_clip_norm=cfg.get("grad_clip_norm"),
        )
        val_metrics = _run_epoch(
            model, val_loader, device, desc=f"epoch {epoch:03d} val"
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        tqdm.write(
            f"  epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f}"
        )
        if val_metrics["loss"] < best_val - min_delta:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "n_channels": C,
                    "T": T,
                    "query_dim": query_dim(),
                    "hidden": cfg["hidden"],
                    "dropout": cfg.get("dropout", 0.0),
                    "config": cfg,
                    "dataset_path": str(dataset_path.relative_to(ROOT)),
                    "splits": {"train": train_idx, "val": val_idx, "test": test_idx},
                    "split_meta": split_meta,
                    "best_epoch": best_epoch,
                },
                run_dir / "checkpoint.pt",
            )
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                tqdm.write(
                    f"  early stopping at epoch={epoch:03d}; "
                    f"best_epoch={best_epoch:03d} best_val_loss={best_val:.4f}"
                )
                break

    summary = {
        "dataset_path": str(dataset_path.relative_to(ROOT)),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "n_instances": len(instances),
        "n_channels": C,
        "T": T,
        "query_dim": query_dim(),
        "split_meta": split_meta,
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"] if history else 0,
        "best_val_loss": best_val,
        "elapsed_sec": time.time() - start,
        "history": history,
    }
    with open(run_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    tqdm.write(f"  saved checkpoint and summary to {run_dir}")


if __name__ == "__main__":
    main()
