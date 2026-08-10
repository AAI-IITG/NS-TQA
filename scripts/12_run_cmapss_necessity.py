"""12 - C-MAPSS necessity experiment (real signals, non-circular).

Builds a non-circular QA benchmark from real C-MAPSS via ``CMAPSSAdapter`` +
``build_real_benchmark`` (train FD001 -> in-dist held-out engines + shifted
FD002/FD004), then runs the multi-seed necessity comparison (end-to-end
baselines vs the faithful NS-TQA path) and writes an aggregate table + figure.

The benchmark is built ONCE (fixed split + balanced QA) and every seed varies
only model initialisation/training, so the reported spread is model variance on
a fixed, releasable benchmark. Probe split variance by changing build.build_seed.

Run:  python scripts/12_run_cmapss_necessity.py [--config configs/cmapss_necessity.yaml]
      (requires C-MAPSS train_FD001.txt etc. under adapter.root)
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


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float("nan"), 0.0)
    m = sum(xs) / len(xs)
    s = stats.pstdev(xs) if len(xs) > 1 else 0.0
    return (m, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "cmapss_necessity.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    a, b, exp = cfg["adapter"], cfg["build"], cfg["experiment"]
    subsets = tuple(b["train_conditions"]) + tuple(b["test_conditions"])
    adapter = CMAPSSAdapter(
        root=ROOT / a["root"], subsets=subsets, rul_cap=a["rul_cap"],
        flat_std_thresh=float(a["flat_std_thresh"]), min_qspan=a["min_qspan"],
        hi_q=a["hi_q"], lo_q=a["lo_q"],
    )

    print(f"building C-MAPSS benchmark: train={b['train_conditions']} "
          f"shift={b['test_conditions']} ...")
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
    m = bm["meta"]
    print(f"  channels={m['n_channels']} train={m['n_train']} "
          f"indist={m['n_indist']} shift={m['n_shift']}")
    for split in ["train", "test_indist", "test_shift"]:
        r = m["balance"][split]
        print(f"  {split:12s} n={r['n']:4d} yes_frac={r['yes_frac']:.3f}")

    baselines = exp["baselines"]
    seeds = exp["seeds"]
    methods = baselines + ["NS-TQA"]
    runs = []
    for seed in seeds:
        print(f"seed {seed} ...", flush=True)
        runs.append(run_single(bm, cfg, seed, baselines, verbose=False))

    # aggregate answer_accuracy mean+/-std per method x regime
    agg = {mth: {reg: mean_std([r[mth][reg]["answer_accuracy"] for r in runs])
                 for reg in ("indist", "shift")} for mth in methods}
    oracle = {reg: mean_std([r["oracle"][reg]["answer_accuracy"] for r in runs])
              for reg in ("indist", "shift")}
    pf1 = {reg: mean_std([r["perception_f1"][reg] for r in runs])
           for reg in ("indist", "shift")}

    # write table + json
    out_dir = ROOT / cfg["run_root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    L = [f"# C-MAPSS necessity ({len(seeds)} seeds): FD001 -> in-dist held-out engines "
         f"+ shift {b['test_conditions']}", "",
         "| Method | in-dist acc | shifted acc |", "|---|---|---|"]
    for mth in methods:
        mi, si = agg[mth]["indist"]
        msh, ssh = agg[mth]["shift"]
        L.append(f"| {mth} | {mi:.3f} ± {si:.3f} | {msh:.3f} ± {ssh:.3f} |")
    L.append(f"| oracle (upper bound) | {oracle['indist'][0]:.3f} | {oracle['shift'][0]:.3f} |")
    L += ["", f"perception macro-F1: indist {pf1['indist'][0]:.3f} ± {pf1['indist'][1]:.3f} "
          f"| shift {pf1['shift'][0]:.3f} ± {pf1['shift'][1]:.3f}"]
    table = "\n".join(L)
    (out_dir / "cmapss_necessity_table.md").write_text(table)
    (out_dir / "cmapss_necessity.json").write_text(json.dumps({
        "agg": {mth: {reg: list(agg[mth][reg]) for reg in agg[mth]} for mth in agg},
        "oracle": {reg: list(oracle[reg]) for reg in oracle},
        "perception_f1": {reg: list(pf1[reg]) for reg in pf1},
        "meta": {k: m[k] for k in ["dataset", "n_channels", "T", "depths", "shift",
                                   "train_conditions", "test_conditions",
                                   "n_train", "n_indist", "n_shift"]},
        "seeds": seeds,
    }, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote table + json to {out_dir}")

    # optional bar figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.arange(len(methods)); w = 0.38
        ind = [agg[mth]["indist"][0] for mth in methods]
        inde = [agg[mth]["indist"][1] for mth in methods]
        sh = [agg[mth]["shift"][0] for mth in methods]
        she = [agg[mth]["shift"][1] for mth in methods]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - w / 2, ind, w, yerr=inde, capsize=3, label="in-dist")
        ax.bar(x + w / 2, sh, w, yerr=she, capsize=3, label="shifted")
        ax.axhline(oracle["shift"][0], ls="--", c="gray", lw=1, label="oracle")
        ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylim(0, 1.05)
        ax.set_ylabel("answer accuracy")
        ax.set_title(f"C-MAPSS necessity: FD001 -> {','.join(b['test_conditions'])}")
        ax.legend(); fig.tight_layout()
        fig.savefig(out_dir / "fig_cmapss_necessity.png", dpi=300)
        print(f"wrote figure to {out_dir / 'fig_cmapss_necessity.png'}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()