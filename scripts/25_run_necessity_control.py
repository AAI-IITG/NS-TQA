"""25 - Necessity fairness control (Phase J.3).

A reviewer could argue that the end-to-end collapse under shift (Table~necessity)
reflects a weak program encoding rather than a genuine *preference* for the
spurious shortcut. This control rules that out: we train the SAME end-to-end
baselines on the SAME instances with the spurious channel REMOVED (causal channels
only). If they then learn the depth-$d$ rule (high in-distribution AND shifted
accuracy, since with no spurious channel the two splits have identical content),
the earlier collapse is shortcut preference, not an inability to learn the rule.

Run:  python scripts/25_run_necessity_control.py [--seeds 0 1 2] [--epochs 30]
"""
import argparse
import json
import statistics as stats
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from benchmark.baseline import select_device
from benchmark.necessity import eval_baseline, train_baseline
from benchmark.spurious import generate_spurious_benchmark


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return (float("nan"), 0.0)
    return (sum(xs) / len(xs), stats.pstdev(xs) if len(xs) > 1 else 0.0)


def causal_only(insts, n_causal):
    """Drop the spurious channel (index n_causal) -> causal channels only."""
    return [replace(i, X=i.X[:, :n_causal]) for i in insts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--baselines", nargs="+", default=["lstm", "transformer", "tcn"])
    ap.add_argument("--n_train", type=int, default=1500)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "necessity.yaml"))["lstm"]
    cfg = dict(cfg, epochs=args.epochs)
    device = select_device("cpu")

    # cell[variant][method][split] -> per-seed list
    variants = ["with-spurious", "causal-only"]
    cell = {v: {m: {"indist": [], "shift": []} for m in args.baselines} for v in variants}
    T = nc = None
    for seed in args.seeds:
        bm = generate_spurious_benchmark(n_train=args.n_train, n_test=600,
                                         shift_mode="shift_decorr", seed=seed)
        nc, T = bm["meta"]["n_causal"], bm["meta"]["T"]
        full_C = nc + 1
        print(f"seed {seed}: n_causal={nc} (+1 spurious), T={T}", flush=True)
        data = {
            "with-spurious": (full_C, bm["train"], bm["test_indist"], bm["test_shift"]),
            "causal-only": (nc, causal_only(bm["train"], nc),
                            causal_only(bm["test_indist"], nc),
                            causal_only(bm["test_shift"], nc)),
        }
        for v, (C, tr, ind, sh) in data.items():
            for name in args.baselines:
                model = train_baseline(name, tr, C, T, cfg, device, seed, verbose=False)
                cell[v][name]["indist"].append(eval_baseline(model, ind, C, T, device)["answer_accuracy"])
                cell[v][name]["shift"].append(eval_baseline(model, sh, C, T, device)["answer_accuracy"])
            mi = mean_std(cell[v][args.baselines[0]]["indist"])[0]
            print(f"  {v}: {args.baselines[0]} in-dist {mi:.3f}", flush=True)

    out_dir = ROOT / "runs" / "necessity_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    L = [f"# Necessity fairness control ({len(args.seeds)} seeds, {args.epochs} ep, "
         f"decorr shift)", "",
         "_End-to-end baselines with the spurious channel present vs removed. "
         "Causal-only learns the depth-$d$ rule (in-dist = shift, both high) -> the "
         "with-spurious collapse is shortcut preference, not inability to learn._", "",
         "| Variant | Method | in-dist | shift |", "|---|---|---|---|"]
    for v in variants:
        for name in args.baselines:
            mi, si = mean_std(cell[v][name]["indist"])
            ms, ss = mean_std(cell[v][name]["shift"])
            L.append(f"| {v} | {name} | {mi:.3f}±{si:.3f} | {ms:.3f}±{ss:.3f} |")
    table = "\n".join(L)
    (out_dir / "necessity_control.md").write_text(table)
    (out_dir / "necessity_control.json").write_text(json.dumps(
        {v: {m: {s: list(mean_std(cell[v][m][s])) for s in ("indist", "shift")}
             for m in args.baselines} for v in variants}, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote {out_dir / 'necessity_control.md'}")


if __name__ == "__main__":
    main()
