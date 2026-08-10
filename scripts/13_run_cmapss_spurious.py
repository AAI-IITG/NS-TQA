"""13 - C-MAPSS necessity on REAL signals with a planted spurious shortcut.

Builds the real C-MAPSS benchmark (same as script 12), then plants a spurious
shortcut channel (correlated with the answer in train/in-dist, broken under
shift) and runs the multi-seed necessity comparison for BOTH shift modes
(decorr + flip). This is the crisp necessity result on real signal backbones:
end-to-end baselines take the shortcut and collapse under shift; the faithful
NS-TQA path cannot use the spurious channel and stays invariant.

Reuses configs/cmapss_necessity.yaml; optional ``spurious:`` block:
    spurious: { strength: 3.0, shift_modes: [shift_decorr, shift_flip] }

Run:  python scripts/13_run_cmapss_spurious.py [--config configs/cmapss_necessity.yaml]
"""
import argparse
import json
import statistics as stats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from benchmark.adapters_cmapss import CMAPSSAdapter
from benchmark.necessity import run_single
from benchmark.realdata import build_real_benchmark
from benchmark.spurious_real import attach_spurious_channel


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float("nan"), 0.0)
    m = sum(xs) / len(xs)
    return (m, stats.pstdev(xs) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "cmapss_necessity.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    a, b, exp = cfg["adapter"], cfg["build"], cfg["experiment"]
    sp = cfg.get("spurious", {})
    strength = sp.get("strength", 3.0)
    shift_modes = sp.get("shift_modes", ["shift_decorr", "shift_flip"])
    baselines, seeds = exp["baselines"], exp["seeds"]
    methods = baselines + ["NS-TQA"]

    subsets = tuple(b["train_conditions"]) + tuple(b["test_conditions"])
    adapter = CMAPSSAdapter(
        root=ROOT / a["root"], subsets=subsets, rul_cap=a["rul_cap"],
        flat_std_thresh=float(a["flat_std_thresh"]), min_qspan=a["min_qspan"],
        hi_q=a["hi_q"], lo_q=a["lo_q"],
    )
    print("building real C-MAPSS benchmark (shared backbone) ...")
    bm = build_real_benchmark(
        adapter, T=b["T"], stride=b["stride"], depths=tuple(b["depths"]),
        shift=b["shift"], train_conditions=tuple(b["train_conditions"]),
        test_conditions=tuple(b["test_conditions"]),
        indist_holdout_frac=b["indist_holdout_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=a["hi_q"], lo_q=a["lo_q"], smooth_k=b["smooth_k"], a_level=b["a_level"],
        allow_until=b["allow_until"], max_windows_per_unit=b.get("max_windows_per_unit"),
        over_factor=b["over_factor"], seed=b["build_seed"],
    )
    print(f"  causal channels={bm['meta']['n_channels']} "
          f"train={bm['meta']['n_train']} indist={bm['meta']['n_indist']} shift={bm['meta']['n_shift']}")

    results = {}
    for mode in shift_modes:
        print(f"\n=== shift_mode = {mode} (spurious strength {strength}) ===")
        sbm = attach_spurious_channel(bm, shift_mode=mode, strength=strength, seed=0)
        runs = []
        for seed in seeds:
            print(f"  seed {seed} ...", flush=True)
            runs.append(run_single(sbm, cfg, seed, baselines, verbose=False))
        agg = {mth: {reg: mean_std([r[mth][reg]["answer_accuracy"] for r in runs])
                     for reg in ("indist", "shift")} for mth in methods}
        shortcut = {reg: mean_std([r["spurious_shortcut"][reg] for r in runs])
                    for reg in ("indist", "shift")}
        oracle = {reg: mean_std([r["oracle"][reg]["answer_accuracy"] for r in runs])
                  for reg in ("indist", "shift")}
        pf1 = {reg: mean_std([r["perception_f1"][reg] for r in runs])
               for reg in ("indist", "shift")}
        results[mode] = {"agg": agg, "shortcut": shortcut, "oracle": oracle, "pf1": pf1}

    # write combined table
    out_dir = ROOT / cfg["run_root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    L = [f"# C-MAPSS necessity on REAL signals + planted spurious shortcut ({len(seeds)} seeds)",
         f"causal backbone: train FD001 -> in-dist held-out engines + shift {b['test_conditions']}; "
         f"spurious strength {strength}", ""]
    for mode in shift_modes:
        R = results[mode]
        L += [f"## {mode}", "", "| Method | in-dist acc | shifted acc |", "|---|---|---|"]
        for mth in methods:
            mi, si = R["agg"][mth]["indist"]
            msh, ssh = R["agg"][mth]["shift"]
            L.append(f"| {mth} | {mi:.3f} ± {si:.3f} | {msh:.3f} ± {ssh:.3f} |")
        L.append(f"| spurious shortcut (ref) | {R['shortcut']['indist'][0]:.3f} | {R['shortcut']['shift'][0]:.3f} |")
        L.append(f"| oracle (upper bound) | {R['oracle']['indist'][0]:.3f} | {R['oracle']['shift'][0]:.3f} |")
        L += ["", f"perception macro-F1: indist {R['pf1']['indist'][0]:.3f} | shift {R['pf1']['shift'][0]:.3f}", ""]
    table = "\n".join(L)
    (out_dir / "cmapss_spurious_table.md").write_text(table)
    (out_dir / "cmapss_spurious.json").write_text(json.dumps({
        mode: {
            "agg": {mth: {reg: list(R["agg"][mth][reg]) for reg in R["agg"][mth]} for mth in R["agg"]},
            "shortcut": {reg: list(R["shortcut"][reg]) for reg in R["shortcut"]},
            "oracle": {reg: list(R["oracle"][reg]) for reg in R["oracle"]},
            "pf1": {reg: list(R["pf1"][reg]) for reg in R["pf1"]},
        } for mode, R in results.items()
    }, indent=2, default=str))
    print("\n" + table)
    print(f"wrote table + json to {out_dir}")


if __name__ == "__main__":
    main()