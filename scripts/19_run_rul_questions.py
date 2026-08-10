"""19 - RUL-grounded PHM question family (Phase E.1).

Builds a non-circular benchmark whose programs' temporal windows are anchored to
each unit's NEAR-FAILURE region (per-step RUL <= rul_k, from Series.meta['rul']),
turning generic temporal questions into PHM-prognostic ones ("as the unit
approaches failure, does the degradation indicator rise/stay high?"). RUL only
sets the interval; the answer is still executor(phi, mu*) over sensor predicates,
so balance / non-circularity / (depth>=2) non-leakage are preserved.

Reports balance, the per-depth leakage probe (must be ~chance for depth>=2), one
worked RUL question, and the necessity/generalization comparison (baselines vs
NS-TQA) on this family.

Run:  python scripts/19_run_rul_questions.py [--config configs/xjtu_necessity.yaml]
                                            [--rul-k K] [--quick]
"""
import argparse
import statistics as stats
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml

from benchmark.necessity import run_single
from benchmark.realdata import build_real_benchmark, real_balance_report
from utils.faithfulness import leakage_probe

build_adapter = import_module("15_run_faithfulness").build_adapter
_cs = import_module("16_case_study")          # reuse humanize() + channel_human()


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return (float("nan"), 0.0)
    return (sum(xs) / len(xs), stats.pstdev(xs) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "xjtu_necessity.yaml"))
    ap.add_argument("--rul-k", type=float, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    a, b, exp = cfg["adapter"], cfg["build"], cfg["experiment"]
    rul_k = args.rul_k if args.rul_k is not None else b.get("rul_k", 50)
    baselines, seeds = exp["baselines"], exp["seeds"]
    if args.quick:
        seeds, baselines = seeds[:1], baselines[:1]
        for k in ("epochs",):
            cfg["lstm"][k] = 5
        cfg["perception"]["epochs"] = 10

    adapter = build_adapter(a, b)
    print(f"building RUL-grounded benchmark ({adapter.name}, rul_k={rul_k}, shift={b['shift']}) ...")
    bm = build_real_benchmark(
        adapter, T=b["T"], stride=b["stride"], depths=tuple(b["depths"]),
        shift=b["shift"], train_conditions=tuple(b["train_conditions"]),
        test_conditions=tuple(b["test_conditions"]),
        indist_holdout_frac=b["indist_holdout_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=a["hi_q"], lo_q=a["lo_q"], smooth_k=b["smooth_k"], a_level=b["a_level"],
        allow_until=b["allow_until"],
        max_windows_per_unit=None,            # keep rare late-life windows (don't subsample them away)
        over_factor=max(b["over_factor"], 400),
        seed=b["build_seed"], question_family="rul", rul_k=rul_k,
    )
    m = bm["meta"]
    names = m["channel_names"]
    regimes = [("test_indist", "indist")] + ([("test_shift", "shift")] if m["n_shift"] > 0 else [])
    print(f"  channels={m['n_channels']} train={m['n_train']} indist={m['n_indist']} shift={m['n_shift']}")
    for split in ["train"] + [k for k, _ in regimes]:
        r = real_balance_report(bm[split])
        print(f"  {split:12s} n={r['n']:4d} yes_frac={r['yes_frac']:.3f}")
    if m["n_train"] == 0:
        raise SystemExit("no RUL-grounded instances built; lower rul_k or check meta['rul']")

    # leakage probe (must be ~chance for depth>=2)
    print("\nleakage probe (best single privileged predicate acc; want ~0.5 for depth>=2):")
    for key, reg in regimes:
        pd = leakage_probe(bm[key])
        print(f"  {reg}: " + "  ".join(f"d{d}={v:.3f}" for d, v in sorted(pd.items())))

    # one worked RUL question
    ex = next((i for i in bm[regimes[-1][0]] if i.depth >= 2), bm["train"][0])
    print("\nexample RUL-grounded question:")
    print(f"  unit={ex.unit_id} cond={ex.condition} depth={ex.depth}")
    print(f"  program: {ex.phi_star.canonical()}")
    print(f"  plain:   near failure, is it true that {_cs.humanize(ex.phi_star, names)}?")
    print(f"  answer:  {'YES' if ex.answer_star else 'NO'}")

    # necessity / generalization on the RUL family
    out_dir = ROOT / cfg["run_root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = baselines + ["NS-TQA"]
    runs = [run_single(bm, cfg, s, baselines, verbose=False) for s in seeds]
    agg = {mth: {reg: mean_std([r[mth][reg]["answer_accuracy"] for r in runs])
                 for _, reg in regimes} for mth in methods}
    orc = {reg: mean_std([r["oracle"][reg]["answer_accuracy"] for r in runs]) for _, reg in regimes}
    cols = " | ".join(f"{reg} acc" for _, reg in regimes)
    L = [f"# RUL-grounded questions ({len(seeds)} seeds): {m['dataset']} rul_k={rul_k}", "",
         f"| Method | {cols} |", "|---" * (1 + len(regimes)) + "|"]
    for mth in methods:
        L.append(f"| {mth} | " + " | ".join(
            f"{agg[mth][reg][0]:.3f} ± {agg[mth][reg][1]:.3f}" for _, reg in regimes) + " |")
    L.append(f"| oracle (upper bound) | " + " | ".join(f"{orc[reg][0]:.3f}" for _, reg in regimes) + " |")
    table = "\n".join(L)
    (out_dir / "rul_questions_table.md").write_text(table)
    print("\n" + table)
    print(f"\nwrote table to {out_dir}")


if __name__ == "__main__":
    main()
