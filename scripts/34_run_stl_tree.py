"""34 - Learned-STL competitor (Bombara-style decision tree), WO-2C.

The symbolic-LEARNING baseline: a depth-limited decision tree learns an STL formula
over a candidate-primitive grid from answer labels, in contrast to NS-TQA which
EXECUTES a given program. Reports in-dist and shifted answer accuracy on XJTU and
C-MAPSS, to be placed between the end-to-end baselines and NS-TQA in the tables.

Run:  python scripts/34_run_stl_tree.py [--config configs/stl_tree.yaml] [--datasets xjtu,cmapss] [--quick]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from benchmark.realdata import build_real_benchmark
from models.stl_tree import eval_stl_tree, train_stl_tree
from utils.oracle_metrics import group_accuracy  # noqa: F401 (used indirectly)


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    return (sum(xs) / len(xs), st.pstdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)


def build_adapter(a):
    if "rul_cap" in a:
        from benchmark.adapters_cmapss import CMAPSSAdapter
        return CMAPSSAdapter(root=ROOT / a["root"], subsets=sorted(set(["FD001", "FD002", "FD004"])),
                             rul_cap=a["rul_cap"], flat_std_thresh=a.get("flat_std_thresh", 1e-6),
                             min_qspan=a.get("min_qspan", 0.05), op_normalize=a.get("op_normalize", False))
    from benchmark.adapters_xjtu import XJTUAdapter
    return XJTUAdapter(root=ROOT / a["root"], fs=a["fs"], n_bands=a["n_bands"],
                       min_snapshots=a.get("min_snapshots", 1), cache_path=ROOT / a["cache_path"])


def build_bm(dcfg):
    a, b = dcfg["adapter"], dcfg["build"]
    return build_real_benchmark(
        build_adapter(a), T=b["T"], stride=b["stride"], depths=tuple(b["depths"]),
        shift=b["shift"], train_conditions=tuple(b["train_conditions"]),
        test_conditions=tuple(b["test_conditions"]), indist_holdout_frac=b["indist_holdout_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=a.get("hi_q", 0.85), lo_q=a.get("lo_q", 0.15), smooth_k=b["smooth_k"],
        a_level=b["a_level"], allow_until=b["allow_until"],
        max_windows_per_unit=b.get("max_windows_per_unit"), over_factor=b["over_factor"],
        seed=b["build_seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "stl_tree.yaml"))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seeds = cfg["experiment"]["seeds"][:1 if args.quick else None]
    depths = cfg.get("max_depths", [cfg.get("max_depth", 6)])
    dsets = list(cfg["datasets"].keys())
    if args.datasets:
        dsets = [d for d in dsets if d in set(args.datasets.split(","))]

    results = {}
    for dname in dsets:
        bm = build_bm(cfg["datasets"][dname])
        C, T = bm["meta"]["n_channels"], cfg["datasets"][dname]["build"]["T"]
        print(f"\n### {dname}: C={C} T={T} train={len(bm['train'])} "
              f"indist={len(bm['test_indist'])} shift={len(bm['test_shift'])}", flush=True)
        by_depth = {}
        for md in depths:
            ind, shf, tr = [], [], []
            for seed in seeds:
                tree = train_stl_tree(bm["train"], C, T, max_depth=md,
                                      min_samples_leaf=cfg.get("min_samples_leaf", 5), seed=seed)
                tr.append(eval_stl_tree(tree, bm["train"], C, T)["answer_accuracy"])
                ind.append(eval_stl_tree(tree, bm["test_indist"], C, T)["answer_accuracy"])
                if bm["test_shift"]:
                    shf.append(eval_stl_tree(tree, bm["test_shift"], C, T)["answer_accuracy"])
            by_depth[md] = {"train": mean_std(tr), "indist": mean_std(ind), "shift": mean_std(shf)}
            print(f"  max_depth={md}: train={mean_std(tr)[0]:.3f} indist={mean_std(ind)[0]:.3f} "
                  f"shift={mean_std(shf)[0]:.3f}", flush=True)
        results[dname] = {"C": C, "by_depth": by_depth}

    _write(cfg, results, seeds, ROOT / cfg["run_root"])


def _write(cfg, results, seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    L = [f"# Learned-STL competitor (Bombara-style decision tree, WO-2C) — {len(seeds)} tree seeds", "",
         "A depth-limited decision tree learns an STL formula over raw per-channel/window "
         "primitives (mean/max/min/std/slope) + the symbolic program encoding, trained on "
         "answer labels. It never sees the privileged grounding. This is the symbolic-LEARNING "
         "baseline vs NS-TQA's symbolic-EXECUTION.", ""]
    for dname, D in results.items():
        L.append(f"## {dname}  (C={D['C']})")
        L += ["", "| max depth | train | in-dist | shift |", "|---|---|---|---|"]
        for md, r in D["by_depth"].items():
            def cell(k):
                m, s = r[k]
                return "—" if m != m else f"{m:.3f}±{s:.3f}"
            L.append(f"| {md} | {cell('train')} | {cell('indist')} | {cell('shift')} |")
        L.append("")
    table = "\n".join(L)
    (out_dir / "stl_tree.md").write_text(table)
    (out_dir / "stl_tree.json").write_text(json.dumps(
        {"seeds": seeds, "results": results}, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote -> {out_dir}/stl_tree.{{md,json}}")


if __name__ == "__main__":
    main()
