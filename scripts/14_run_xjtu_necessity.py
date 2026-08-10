"""14 - XJTU-SY bearing necessity / generalization (real vibration, non-circular).

Builds a non-circular QA benchmark from XJTU-SY via ``XJTUAdapter`` (raw 2-axis
vibration -> RMS/kurtosis/peak/crest/band-energy HI channels) +
``build_real_benchmark``, then runs the multi-seed comparison (end-to-end
baselines vs the faithful NS-TQA path).

Two regimes, picked by ``build.shift`` in the config:

* ``shift=unit``  -> held-out-BEARING generalization. With only the 40Hz10kN
  load on disk this is the result that runs today (train Bearing3_4 -> test
  Bearing3_5). There is no shifted operating regime, so the shifted column and
  the spurious necessity bridge are reported as N/A.
* ``shift=condition`` -> cross-LOAD necessity (the headline). Needs >1 load
  present; then this script also plants the spurious shortcut (cf. script 13)
  and reports baselines collapsing under load shift while NS-TQA stays invariant.

Run:  python scripts/14_run_xjtu_necessity.py [--config configs/xjtu_necessity.yaml] [--quick]
      --quick: 1 seed, lstm only, few epochs (fast plumbing/sanity pass).
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

from benchmark.adapters_xjtu import XJTUAdapter
from benchmark.necessity import run_single
from benchmark.realdata import build_real_benchmark
from benchmark.spurious_real import attach_spurious_channel


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float("nan"), 0.0)
    m = sum(xs) / len(xs)
    return (m, stats.pstdev(xs) if len(xs) > 1 else 0.0)


def agg_by_depth(runs, methods, regimes, depths):
    """{method: {regime: {depth: (mean, std)}}} of answer accuracy, over seeds.

    ``run_single`` already records per-depth accuracy in each eval dict's
    ``by_depth`` ({int depth -> acc}); here we aggregate it across seeds so the
    composition claim (C5: NS-TQA flat across depth) can be read off directly.
    """
    out = {}
    for mth in methods:
        out[mth] = {}
        for reg in regimes:
            out[mth][reg] = {
                d: mean_std([r[mth][reg].get("by_depth", {}).get(d) for r in runs])
                for d in depths
            }
    return out


def depth_table(title, dagg, methods, regime, depths):
    head = " | ".join(f"d{d}" for d in depths)
    sep = "|---" * (len(depths) + 1) + "|"
    L = [f"## {title}", "", f"| Method | {head} |", sep]
    for mth in methods:
        cells = " | ".join(f"{dagg[mth][regime][d][0]:.3f}" for d in depths)
        L.append(f"| {mth} | {cells} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "xjtu_necessity.yaml"))
    ap.add_argument("--quick", action="store_true",
                    help="1 seed, lstm only, few epochs (sanity pass)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    a, b, exp = cfg["adapter"], cfg["build"], cfg["experiment"]
    if args.quick:
        exp["seeds"] = [0]
        exp["baselines"] = ["lstm"]
        cfg["lstm"]["epochs"] = 5
        cfg["perception"]["epochs"] = 10

    adapter = XJTUAdapter(
        root=ROOT / a["root"], fs=a["fs"], n_bands=a["n_bands"],
        min_snapshots=a.get("min_snapshots", 1),
        cache_path=ROOT / a["cache_path"], force_recompute=a.get("force_recompute", False),
    )

    print(f"building XJTU benchmark: shift={b['shift']} ...")
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
    has_shift = m["n_shift"] > 0
    train_units = sorted({i.unit_id for i in bm["train"]})
    indist_units = sorted({i.unit_id for i in bm["test_indist"]})
    print(f"  channels={m['n_channels']} train={m['n_train']} "
          f"indist={m['n_indist']} shift={m['n_shift']}")
    print(f"  train bearings={train_units}")
    print(f"  held-out bearings={indist_units}")
    for split in ["train", "test_indist", "test_shift"]:
        r = m["balance"][split]
        print(f"  {split:12s} n={r['n']:4d} yes_frac={r['yes_frac']:.3f}")
    if not has_shift:
        print("  NOTE: no shifted regime (single operating condition on disk) -> "
              "held-out-bearing generalization only; shift column = N/A.")

    baselines, seeds = exp["baselines"], exp["seeds"]
    methods = baselines + ["NS-TQA"]
    regimes = ("indist", "shift") if has_shift else ("indist",)

    out_dir = ROOT / cfg["run_root"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- main generalization table (always) ----
    runs = []
    for seed in seeds:
        print(f"seed {seed} ...", flush=True)
        runs.append(run_single(bm, cfg, seed, baselines, verbose=False))
    agg = {mth: {reg: mean_std([r[mth][reg]["answer_accuracy"] for r in runs])
                 for reg in regimes} for mth in methods}
    oracle = {reg: mean_std([r["oracle"][reg]["answer_accuracy"] for r in runs])
              for reg in regimes}
    pf1 = {reg: mean_std([r["perception_f1"][reg] for r in runs]) for reg in regimes}

    cols = "in-dist acc" if not has_shift else "in-dist acc | shifted acc"
    sep = "|---|---|" if not has_shift else "|---|---|---|"
    title = ("held-out bearing" if not has_shift
             else f"cross-load -> {b['test_conditions']}")
    L = [f"# XJTU necessity/generalization ({len(seeds)} seeds): {title}", "",
         f"| Method | {cols} |", sep]
    for mth in methods:
        if has_shift:
            mi, si = agg[mth]["indist"]; msh, ssh = agg[mth]["shift"]
            L.append(f"| {mth} | {mi:.3f} ± {si:.3f} | {msh:.3f} ± {ssh:.3f} |")
        else:
            mi, si = agg[mth]["indist"]
            L.append(f"| {mth} | {mi:.3f} ± {si:.3f} |")
    if has_shift:
        L.append(f"| oracle (upper bound) | {oracle['indist'][0]:.3f} | {oracle['shift'][0]:.3f} |")
        L += ["", f"perception macro-F1: indist {pf1['indist'][0]:.3f} ± {pf1['indist'][1]:.3f} "
              f"| shift {pf1['shift'][0]:.3f} ± {pf1['shift'][1]:.3f}"]
    else:
        L.append(f"| oracle (upper bound) | {oracle['indist'][0]:.3f} |")
        L += ["", f"perception macro-F1: indist {pf1['indist'][0]:.3f} ± {pf1['indist'][1]:.3f}"]
    table = "\n".join(L)

    # ---- composition: accuracy by program depth (C5) ----
    depths = list(b["depths"])
    dagg = agg_by_depth(runs, methods, regimes, depths)
    comp_blocks = [depth_table(f"Composition — {reg} accuracy by program depth",
                               dagg, methods, reg, depths) for reg in regimes]
    composition = "\n\n".join(comp_blocks)
    full_table = table + "\n\n" + composition
    (out_dir / "xjtu_generalization_table.md").write_text(full_table)
    (out_dir / "xjtu_generalization.json").write_text(json.dumps({
        "agg": {mth: {reg: list(agg[mth][reg]) for reg in agg[mth]} for mth in agg},
        "by_depth": {mth: {reg: {d: list(dagg[mth][reg][d]) for d in depths}
                           for reg in regimes} for mth in methods},
        "oracle": {reg: list(oracle[reg]) for reg in oracle},
        "perception_f1": {reg: list(pf1[reg]) for reg in pf1},
        "meta": {k: m[k] for k in ["dataset", "n_channels", "T", "depths", "shift",
                                   "train_conditions", "test_conditions",
                                   "n_train", "n_indist", "n_shift"]},
        "seeds": seeds, "has_shift": has_shift,
    }, indent=2, default=str))
    print("\n" + full_table)
    print(f"\nwrote table + json to {out_dir}")

    # ---- spurious necessity bridge (only when a shift pool exists) ----
    sp = cfg.get("spurious")
    if has_shift and sp:
        strength = sp.get("strength", 3.0)
        for mode in sp.get("shift_modes", ["shift_decorr", "shift_flip"]):
            print(f"\n=== spurious necessity: {mode} (strength {strength}) ===")
            sbm = attach_spurious_channel(bm, shift_mode=mode, strength=strength, seed=0)
            sruns = [run_single(sbm, cfg, s, baselines, verbose=False) for s in seeds]
            sagg = {mth: {reg: mean_std([r[mth][reg]["answer_accuracy"] for r in sruns])
                          for reg in ("indist", "shift")} for mth in methods}
            SL = [f"## XJTU spurious necessity ({mode}, {len(seeds)} seeds)", "",
                  "| Method | in-dist acc | shifted acc |", "|---|---|---|"]
            for mth in methods:
                mi, si = sagg[mth]["indist"]; msh, ssh = sagg[mth]["shift"]
                SL.append(f"| {mth} | {mi:.3f} ± {si:.3f} | {msh:.3f} ± {ssh:.3f} |")
            stable = "\n".join(SL)
            (out_dir / f"xjtu_spurious_{mode}.md").write_text(stable)
            print(stable)
    elif sp:
        print("\n(spurious necessity bridge skipped: needs a shifted regime; "
              "only one operating load is present on disk.)")


if __name__ == "__main__":
    main()
