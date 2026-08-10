"""41 - Headline generalization at 10 seeds, with per-seed arrays (statistical upgrade).

Re-runs the main real-data generalization table (NS-TQA and the end-to-end baselines on
C-MAPSS FD001->FD002/4 and XJTU cross-load, in-dist + shift) at TEN model seeds instead of
five, and -- unlike the conference script -- SAVES the per-seed accuracy arrays so the
significance post-processor (``scripts/38``) can Holm-correct the headline comparisons. The
benchmark is built once per dataset; only the model seed varies. Adds the deterministic
STL-only grounding as a reference row.

Run:  python scripts/41_run_generalization_10seed.py [--config configs/generalization10.yaml] [--datasets cmapss,xjtu] [--quick]
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
from benchmark.necessity import run_single
from models.stl_only import stl_only_evaluate
from utils.stats import holm_bonferroni, wilcoxon


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
        shift="condition", train_conditions=tuple(b["train_conditions"]),
        test_conditions=tuple(b["test_conditions"]), indist_holdout_frac=b["indist_holdout_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=a.get("hi_q", 0.85), lo_q=a.get("lo_q", 0.15), smooth_k=b["smooth_k"],
        a_level=b["a_level"], allow_until=b["allow_until"],
        max_windows_per_unit=b.get("max_windows_per_unit"), over_factor=b["over_factor"],
        seed=b["build_seed"])


def run_dataset(cfg, dcfg, seeds, quick):
    bm = build_bm(dcfg)
    C, T = bm["meta"]["n_channels"], dcfg["build"]["T"]
    baselines = cfg["experiment"]["baselines"][:1] if quick else cfg["experiment"]["baselines"]
    methods = baselines + ["NS-TQA"]
    # per-seed arrays: per_seed[method][regime] = [acc per seed]
    per_seed = {m: {"indist": [], "shift": []} for m in methods}
    orc = {"indist": [], "shift": []}
    pf1 = {"indist": [], "shift": []}
    for seed in seeds:
        print(f"  seed {seed} ...", flush=True)
        r = run_single(bm, cfg, seed, baselines, verbose=False)
        for m in methods:
            for reg in ("indist", "shift"):
                per_seed[m][reg].append(r[m][reg]["answer_accuracy"])
        for reg in ("indist", "shift"):
            orc[reg].append(r["oracle"][reg]["answer_accuracy"])
            pf1[reg].append(r["perception_f1"][reg])
    # STL-only (deterministic reference)
    stl = {}
    for reg, pool in (("indist", bm["test_indist"]), ("shift", bm["test_shift"])):
        stl[reg] = stl_only_evaluate(pool, C, a_level=dcfg["build"]["a_level"],
                                     smooth_k=dcfg["build"]["smooth_k"])["answer_accuracy"] if pool else None
    return {"C": C, "methods": methods, "per_seed": per_seed, "oracle": orc,
            "perception_f1": pf1, "stl_only": stl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "generalization10.yaml"))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seeds = cfg["experiment"]["seeds"][:2 if args.quick else None]
    dsets = list(cfg["datasets"].keys())
    if args.datasets:
        dsets = [d for d in dsets if d in set(args.datasets.split(","))]
    res = {d: run_dataset(cfg, cfg["datasets"][d], seeds, args.quick) for d in dsets}
    _write(res, seeds, ROOT / cfg["run_root"])


def _write(res, seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    L = [f"# Headline generalization at {len(seeds)} seeds (with Holm-corrected significance)", "",
         "NS-TQA and end-to-end baselines, shifted + in-dist answer accuracy. `p(Holm)` is the "
         "Holm-corrected one-sided Wilcoxon of NS-TQA vs. each baseline within each dataset's "
         "shifted column.", ""]
    for d, D in res.items():
        L.append(f"## {d}  (C={D['C']})")
        L += ["", "| Method | in-dist | shift | p(shift, Holm) |", "|---|---|---|---|"]
        ns_shift = D["per_seed"]["NS-TQA"]["shift"]
        # Holm family: NS-TQA vs each baseline on the shift column
        base = [m for m in D["methods"] if m != "NS-TQA"]
        raw = [wilcoxon(ns_shift, D["per_seed"][m]["shift"], alternative="greater")["p"] for m in base]
        holm = holm_bonferroni(raw)["p_adjusted"]
        padj = dict(zip(base, holm))
        for m in D["methods"]:
            mi, si = mean_std(D["per_seed"][m]["indist"])
            ms, ss = mean_std(D["per_seed"][m]["shift"])
            p = f"{padj[m]:.4f}" if m in padj and padj[m] == padj[m] else "—"
            star = " (NS-TQA ref)" if m == "NS-TQA" else ""
            L.append(f"| {m}{star} | {mi:.3f}±{si:.3f} | {ms:.3f}±{ss:.3f} | {p} |")
        oi, os_ = mean_std(D["oracle"]["indist"])[0], mean_std(D["oracle"]["shift"])[0]
        L.append(f"| oracle | {oi:.3f} | {os_:.3f} | — |")
        if D["stl_only"]["shift"] is not None:
            L.append(f"| STL-only (ref) | {D['stl_only']['indist']:.3f} | {D['stl_only']['shift']:.3f} | — |")
        pf = mean_std(D["perception_f1"]["shift"])[0]
        L.append(f"\n_perception macro-F1 (shift): {pf:.3f}_\n")
    table = "\n".join(L)
    (out_dir / "generalization10.md").write_text(table)
    (out_dir / "generalization10.json").write_text(json.dumps(
        {"seeds": seeds, "results": res}, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote -> {out_dir}/generalization10.{{md,json}}")


if __name__ == "__main__":
    main()
