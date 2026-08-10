"""20 - STL-only baseline (hand-written rules + executor, no learned perception).

Builds the same non-circular benchmark as the NS-TQA experiments (XJTU or C-MAPSS,
via the config) and evaluates the STL-only baseline -- fixed hand-chosen grounding
thresholds + the parameter-free executor -- on the in-distribution and shifted
regimes, with a per-depth breakdown and a small threshold sweep. No training.

The NS-TQA and oracle numbers for the same benchmark are already reported by
scripts 12/14/15; this script supplies the missing "STL-only (hand rules)" row.

Run:  python scripts/20_run_stl_only.py [--config configs/xjtu_necessity.yaml]
"""
import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml

from benchmark.necessity import oracle_accuracy
from benchmark.realdata import build_real_benchmark
from models.stl_only import stl_only_evaluate
from perception.grounding import predicate_index

build_adapter = import_module("15_run_faithfulness").build_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "xjtu_necessity.yaml"))
    ap.add_argument("--hi", type=float, default=1.0)
    ap.add_argument("--lo", type=float, default=-1.0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    a, b = cfg["adapter"], cfg["build"]

    adapter = build_adapter(a, b)
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
    C = bm["meta"]["n_channels"]
    pidx = predicate_index(C)
    regimes = [("test_indist", "indist")]
    if bm["meta"]["n_shift"] > 0:
        regimes.append(("test_shift", "shift"))
    print(f"{adapter.name}: channels={C} "
          + " ".join(f"{r}={len(bm[k])}" for k, r in regimes))

    sk, al = b["smooth_k"], b["a_level"]
    main_res = {}
    for key, reg in regimes:
        stl = stl_only_evaluate(bm[key], C, hi=args.hi, lo=args.lo,
                                smooth_k=sk, a_level=al)
        orc = oracle_accuracy(bm[key], pidx)["answer_accuracy"]
        main_res[reg] = {"stl_only": stl["answer_accuracy"], "oracle": orc,
                         "by_depth": stl["by_depth"]}
        bd = " ".join(f"d{d}:{v:.2f}" for d, v in sorted(stl["by_depth"].items()))
        print(f"  {reg:7s} STL-only={stl['answer_accuracy']:.3f}  oracle={orc:.3f}  [{bd}]")

    # small threshold sweep on the shifted (or in-dist) regime for sensitivity
    sweep_key, sweep_reg = regimes[-1]
    print(f"\nthreshold sweep on {sweep_reg} (hi=-lo):")
    sweep = {}
    for thr in (0.5, 0.75, 1.0, 1.25, 1.5):
        acc = stl_only_evaluate(bm[sweep_key], C, hi=thr, lo=-thr,
                                smooth_k=sk, a_level=al)["answer_accuracy"]
        sweep[thr] = acc
        print(f"  ±{thr:.2f} std -> {acc:.3f}")

    out_dir = ROOT / cfg["run_root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# STL-only baseline (hand rules + executor): {adapter.name} shift={b['shift']}",
             "", "| regime | STL-only acc | oracle |", "|---|---|---|"]
    for reg, v in main_res.items():
        lines.append(f"| {reg} | {v['stl_only']:.3f} | {v['oracle']:.3f} |")
    lines += ["", f"thresholds hi=+{args.hi}/lo={args.lo} std; sweep on {sweep_reg}: "
              + ", ".join(f"±{t}={a:.3f}" for t, a in sweep.items())]
    table = "\n".join(lines)
    (out_dir / "stl_only.md").write_text(table)
    (out_dir / "stl_only.json").write_text(json.dumps(
        {"dataset": adapter.name, "n_channels": C, "results": main_res,
         "sweep": sweep, "hi": args.hi, "lo": args.lo}, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote to {out_dir}")


if __name__ == "__main__":
    main()
