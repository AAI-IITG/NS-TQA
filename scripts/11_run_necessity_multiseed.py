"""11 - Multi-seed necessity experiment (publication-ready table + figures).

Runs the necessity experiment (every baseline architecture + faithful NS-TQA)
across several seeds and aggregates mean +/- std, so the headline table carries
error bars. Reads model hyperparameters from configs/necessity.yaml; ``seeds``
and ``baselines`` may be set there or via CLI.

Run:
  python scripts/11_run_necessity_multiseed.py
  python scripts/11_run_necessity_multiseed.py --seeds 0 1 2 3 4 --baselines lstm transformer tcn
  python scripts/11_run_necessity_multiseed.py --benchmark data/synthetic/spurious_shift_flip.pkl
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from benchmark.necessity import run_single

METRICS = ["answer_accuracy", "balanced_accuracy", "f1"]
COLORS = {"lstm": "#B0413E", "transformer": "#C97B27", "tcn": "#7A5C99",
          "NS-TQA": "#2C7A7B"}


def _mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def aggregate(seed_results: list[dict], methods: list[str], depths: list[int]) -> dict:
    agg: dict = {}
    for m in methods:
        agg[m] = {}
        for split in ("indist", "shift"):
            entry = {}
            for metric in METRICS:
                entry[metric] = _mean_std([r[m][split][metric] for r in seed_results])
            entry["by_depth"] = {
                d: _mean_std([r[m][split]["by_depth"].get(d, float("nan"))
                              for r in seed_results])
                for d in depths
            }
            agg[m][split] = entry
    # references
    agg["oracle"] = {
        s: _mean_std([r["oracle"][s]["answer_accuracy"] for r in seed_results])
        for s in ("indist", "shift")
    }
    agg["spurious_shortcut"] = {
        s: _mean_std([r["spurious_shortcut"][s] for r in seed_results])
        for s in ("indist", "shift")
    }
    agg["perception_f1"] = {
        s: _mean_std([r["perception_f1"][s] for r in seed_results])
        for s in ("indist", "shift")
    }
    return agg


def _fmt(ms: tuple[float, float]) -> str:
    return f"{ms[0]:.3f}+-{ms[1]:.3f}"


def write_results(run_dir: Path, meta: dict, seed_results: list[dict],
                  agg: dict, methods: list[str], depths: list[int]):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "results.json", "w") as f:
        json.dump({"meta": meta, "aggregate": agg, "per_seed": seed_results}, f, indent=2)

    head = ["method", "split", "accuracy", "balanced_acc", "f1"] + [f"acc@d{d}" for d in depths]
    lines = ["# Necessity experiment (multi-seed)", "",
             f"benchmark: `{meta['benchmark']}` (shift_mode={meta['shift_mode']}), "
             f"seeds={meta['seeds']}, n_causal={meta['n_causal']}, T={meta['T']}, "
             f"depths={depths}", "",
             "Values are mean+-std over seeds.", "",
             "| " + " | ".join(head) + " |",
             "| " + " | ".join(["---"] * len(head)) + " |"]
    for m in methods:
        for split in ("indist", "shift"):
            e = agg[m][split]
            cells = [m, split, _fmt(e["answer_accuracy"]), _fmt(e["balanced_accuracy"]),
                     _fmt(e["f1"])] + [_fmt(e["by_depth"][d]) for d in depths]
            lines.append("| " + " | ".join(cells) + " |")
    for split in ("indist", "shift"):
        lines.append("| " + " | ".join(
            ["oracle (mu_star)", split, _fmt(agg["oracle"][split]), "-", "-"]
            + ["-"] * len(depths)) + " |")
        lines.append("| " + " | ".join(
            ["spurious shortcut", split, _fmt(agg["spurious_shortcut"][split]), "-", "-"]
            + ["-"] * len(depths)) + " |")
    lines += ["", f"Perception macro-F1: indist={_fmt(agg['perception_f1']['indist'])}, "
              f"shift={_fmt(agg['perception_f1']['shift'])} "
              "(equal by design: identical causal channels across splits)."]
    (run_dir / "results.md").write_text("\n".join(lines) + "\n")


def plot_bar(run_dir: Path, agg: dict, methods: list[str], shift_mode: str):
    splits = ["indist", "shift"]
    x = np.arange(len(splits))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, m in enumerate(methods):
        means = [agg[m][s]["answer_accuracy"][0] for s in splits]
        stds = [agg[m][s]["answer_accuracy"][1] for s in splits]
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, means, w, yerr=stds, capsize=3,
               label=("NS-TQA (ours)" if m == "NS-TQA" else m),
               color=COLORS.get(m, None))
    short = [agg["spurious_shortcut"][s][0] for s in splits]
    ax.plot(x, short, "o--", color="#888", label="spurious shortcut")
    ax.axhline(0.5, ls=":", color="black", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(["in-distribution", f"shift ({shift_mode})"])
    ax.set_ylabel("answer accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Necessity: end-to-end models collapse under shift; NS-TQA holds")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout(); fig.savefig(run_dir / "fig_necessity_bar.png", dpi=300); plt.close(fig)


def plot_depth(run_dir: Path, agg: dict, methods: list[str], depths: list[int]):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, split in zip(axes, ("indist", "shift")):
        for m in methods:
            means = np.array([agg[m][split]["by_depth"][d][0] for d in depths])
            stds = np.array([agg[m][split]["by_depth"][d][1] for d in depths])
            ax.plot(depths, means, "o-", color=COLORS.get(m, None),
                    label=("NS-TQA (ours)" if m == "NS-TQA" else m))
            ax.fill_between(depths, means - stds, means + stds, color=COLORS.get(m, None), alpha=0.15)
        ax.axhline(0.5, ls=":", color="black", lw=1)
        ax.set_xlabel("program depth"); ax.set_xticks(depths); ax.set_ylim(0, 1.05)
        ax.set_title(f"{split}")
    axes[0].set_ylabel("answer accuracy")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Accuracy vs compositional depth (mean +- std over seeds)")
    fig.tight_layout(); fig.savefig(run_dir / "fig_depth_curve.png", dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "necessity.yaml"))
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--baselines", nargs="+", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.benchmark is not None:
        cfg["benchmark"] = args.benchmark
    seeds = args.seeds or cfg.get("seeds") or [0, 1, 2, 3, 4]
    baselines = args.baselines or cfg.get("baselines") or ["lstm", "transformer", "tcn"]
    methods = baselines + ["NS-TQA"]

    bench_path = ROOT / cfg["benchmark"]
    with open(bench_path, "rb") as f:
        bm = pickle.load(f)
    meta = bm["meta"]
    depths = meta["depths"]
    print(
        f"loaded {bench_path.name} | seeds={seeds} baselines={baselines} "
        f"| C={meta['n_channels']} T={meta['T']} depths={depths}",
        flush=True,
    )

    t0 = time.time()
    seed_results = []
    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        seed_results.append(run_single(bm, cfg, seed, baselines, verbose=False))
        last = seed_results[-1]
        for m in methods:
            print(f"  {m:11s} indist={last[m]['indist']['answer_accuracy']:.3f} "
                  f"shift={last[m]['shift']['answer_accuracy']:.3f}", flush=True)

    agg = aggregate(seed_results, methods, depths)
    run_dir = ROOT / cfg["run_root"] / meta["shift_mode"] / "multiseed"
    meta_out = {"benchmark": str(bench_path.relative_to(ROOT)), "seeds": seeds,
                "baselines": baselines, **meta, "elapsed_sec": time.time() - t0}
    write_results(run_dir, meta_out, seed_results, agg, methods, depths)
    plot_bar(run_dir, agg, methods, meta["shift_mode"])
    plot_depth(run_dir, agg, methods, depths)

    print("\n=== aggregate (mean+-std over seeds) ===", flush=True)
    for m in methods:
        print(f"{m:11s} indist={_fmt(agg[m]['indist']['answer_accuracy'])} "
              f"shift={_fmt(agg[m]['shift']['answer_accuracy'])}", flush=True)
    print(
        f"oracle      indist={_fmt(agg['oracle']['indist'])} "
        f"shift={_fmt(agg['oracle']['shift'])}",
        flush=True,
    )
    print(f"saved table + figures to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
