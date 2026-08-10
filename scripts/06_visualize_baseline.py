"""06 - Visualize LSTM baseline training and evaluation metrics.

Run:
    python scripts/06_visualize_baseline.py --run-dir runs/baseline/cmapss_FD001
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json

import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _bar(ax, labels, values, title, ylabel, ylim=None):
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=30)


def _plot_history(run_dir: Path, out_dir: Path) -> None:
    summary_path = run_dir / "train_summary.json"
    if not summary_path.exists():
        return
    summary = _load_json(summary_path)
    history = summary.get("history", [])
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    train_acc = [h["train"]["accuracy"] for h in history]
    val_acc = [h["val"]["accuracy"] for h in history]
    best_epoch = summary.get("best_epoch")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, val_loss, marker="o", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    if best_epoch:
        axes[0].axvline(best_epoch, color="#E45756", linestyle="--", label="best val")
    axes[0].legend()
    axes[1].plot(epochs, train_acc, marker="o", label="train")
    axes[1].plot(epochs, val_acc, marker="o", label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    if best_epoch:
        axes[1].axvline(best_epoch, color="#E45756", linestyle="--", label="best val")
    axes[1].legend()
    fig.suptitle(summary.get("dataset_path", run_dir.name))
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=300)
    plt.close(fig)


def _plot_metrics(metrics_path: Path, out_dir: Path) -> None:
    metrics = _load_json(metrics_path)
    binary = metrics.get("binary", {})

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"{Path(metrics['dataset_path']).stem} | split={metrics['split']}")

    cm = [
        [binary.get("tn", 0), binary.get("fp", 0)],
        [binary.get("fn", 0), binary.get("tp", 0)],
    ]
    im = axes[0, 0].imshow(cm, cmap="Blues")
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_xticks([0, 1], ["pred no", "pred yes"])
    axes[0, 0].set_yticks([0, 1], ["gold no", "gold yes"])
    for r, row in enumerate(cm):
        for c, value in enumerate(row):
            axes[0, 0].text(c, r, str(value), ha="center", va="center")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    score_labels = ["accuracy", "balanced", "precision", "recall", "f1", "majority no"]
    score_values = [
        metrics.get("answer_accuracy", 0.0),
        binary.get("balanced_accuracy", 0.0),
        binary.get("precision", 0.0),
        binary.get("recall", 0.0),
        binary.get("f1", 0.0),
        binary.get("majority_no_accuracy", 0.0),
    ]
    _bar(axes[0, 1], score_labels, score_values, "Core Scores", "score", (0, 1))

    template = metrics.get("by_template", {})
    _bar(
        axes[1, 0],
        list(template.keys()),
        [v["accuracy"] for v in template.values()],
        "Accuracy By Template",
        "accuracy",
        (0, 1),
    )

    _bar(
        axes[1, 1],
        ["gold yes", "pred yes"],
        [metrics.get("gold_yes_frac", 0.0), metrics.get("pred_yes_frac", 0.0)],
        "Yes Rate",
        "fraction",
        (0, 1),
    )

    fig.tight_layout()
    fig.savefig(out_dir / f"{metrics['split']}_overview.png", dpi=300)
    plt.close(fig)

    for group_name, group_metrics in [
        ("engine", metrics.get("by_engine_id", {})),
        ("bearing", metrics.get("by_bearing_id", {})),
    ]:
        if not group_metrics:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        _bar(
            ax,
            list(group_metrics.keys()),
            [v["accuracy"] for v in group_metrics.values()],
            f"Accuracy By {group_name.title()}",
            "accuracy",
            (0, 1),
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{metrics['split']}_by_{group_name}.png", dpi=300)
        plt.close(fig)


def _plot_comparison(run_dirs: list[Path], split: str, out_path: Path) -> None:
    rows = []
    for run_dir in run_dirs:
        metrics_path = run_dir / f"metrics_{split}.json"
        if not metrics_path.exists():
            continue
        metrics = _load_json(metrics_path)
        binary = metrics.get("binary", {})
        split_meta = metrics.get("split_meta", {})
        label = f"{Path(metrics['dataset_path']).stem}\\n{split_meta.get('split_mode', 'random')}"
        if split_meta.get("group_key"):
            label += f": {split_meta['group_key']}"
        rows.append(
            {
                "label": label,
                "accuracy": metrics.get("answer_accuracy", 0.0),
                "balanced_accuracy": binary.get("balanced_accuracy", 0.0),
                "f1": binary.get("f1", 0.0),
                "recall": binary.get("recall", 0.0),
                "majority_no": binary.get("majority_no_accuracy", 0.0),
            }
        )
    if not rows:
        raise SystemExit("no metrics found for comparison")

    labels = [r["label"] for r in rows]
    score_names = ["accuracy", "balanced_accuracy", "f1", "recall", "majority_no"]
    x = range(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(max(9, 2.5 * len(labels)), 5))
    for offset, score in enumerate(score_names):
        values = [r[score] for r in rows]
        xs = [i + (offset - 2) * width for i in x]
        ax.bar(xs, values, width=width, label=score)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title(f"Baseline Comparison | split={split}")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--compare-out", default=None)
    args = parser.parse_args()

    run_dirs = [ROOT / d for d in args.run_dir]
    for run_dir in run_dirs:
        metrics_path = run_dir / f"metrics_{args.split}.json"
        if not metrics_path.exists():
            raise SystemExit(f"metrics file not found: {metrics_path}")
        out_dir = run_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        _plot_history(run_dir, out_dir)
        _plot_metrics(metrics_path, out_dir)
        print(f"saved plots to {out_dir}")

    if args.compare_out is not None:
        out_path = ROOT / args.compare_out
        _plot_comparison(run_dirs, args.split, out_path)
        print(f"saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
