"""15 - Explanation-faithfulness evaluation (the title's "Explainable").

Builds a non-circular benchmark (XJTU cross-load by default, via the same config
as script 14), trains the faithful NS-TQA perception path, and measures whether
the explanations it ships are faithful:

  * counterfactual validity   (does the cited evidence actually cause the answer)
  * supporting-predicate acc   (model's governing predicate vs the oracle's)
  * critical-interval IoU       (evidence localisation vs the oracle)
  * leakage probe per depth     (no single predicate encodes the answer -> the
                                 executor is load-bearing, not decorative)

Reports mean +/- std over seeds, on in-dist and (if present) shifted regimes,
with a per-depth breakdown. Works for any ``build_real_benchmark`` adapter; point
``--config`` at a different YAML to run it on C-MAPSS instead.

Run:  python scripts/15_run_faithfulness.py [--config configs/xjtu_necessity.yaml]
                                            [--seeds 3] [--quick]
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

from benchmark.realdata import build_real_benchmark
from models.nstqa_learned import LearnedNSTQA
from perception.learned import train_perception
from utils.faithfulness import faithfulness_report, leakage_probe


def build_adapter(a: dict, b: dict):
    """Construct the dataset adapter from the config (XJTU vibration or C-MAPSS).

    Dispatches on adapter keys so this faithfulness runner works on any
    ``build_real_benchmark`` dataset: ``rul_cap`` -> C-MAPSS, ``n_bands`` -> XJTU.
    """
    if "rul_cap" in a:                                   # C-MAPSS (tabular)
        from benchmark.adapters_cmapss import CMAPSSAdapter
        subsets = tuple(b["train_conditions"]) + tuple(b["test_conditions"])
        return CMAPSSAdapter(
            root=ROOT / a["root"], subsets=subsets, rul_cap=a["rul_cap"],
            flat_std_thresh=float(a["flat_std_thresh"]), min_qspan=a["min_qspan"],
            hi_q=a["hi_q"], lo_q=a["lo_q"],
        )
    from benchmark.adapters_xjtu import XJTUAdapter      # XJTU (vibration)
    return XJTUAdapter(
        root=ROOT / a["root"], fs=a["fs"], n_bands=a["n_bands"],
        min_snapshots=a.get("min_snapshots", 1),
        cache_path=ROOT / a["cache_path"], force_recompute=a.get("force_recompute", False),
    )


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]   # drop None/NaN
    if not xs:
        return (float("nan"), 0.0)
    return (sum(xs) / len(xs), stats.pstdev(xs) if len(xs) > 1 else 0.0)


METRICS = [
    ("counterfactual_validity", "counterfactual validity"),
    ("supporting_predicate_accuracy", "supporting-pred acc"),
    ("support_acc_on_correct", "support acc | correct"),
    ("critical_interval_iou", "critical-interval IoU"),
    ("critical_step_agreement", "critical-step agree"),
    ("answer_accuracy", "answer accuracy"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "xjtu_necessity.yaml"))
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds (default: all in config)")
    ap.add_argument("--quick", action="store_true", help="1 seed, few epochs (plumbing)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    a, b, exp = cfg["adapter"], cfg["build"], cfg["experiment"]
    seeds = exp["seeds"]
    if args.seeds is not None:
        seeds = seeds[:args.seeds]
    if args.quick:
        seeds = seeds[:1]
        cfg["perception"]["epochs"] = 10

    adapter = build_adapter(a, b)
    print(f"building benchmark ({adapter.name}): shift={b['shift']} ...")
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
    regimes = [("test_indist", "indist")]
    if bm["meta"]["n_shift"] > 0:
        regimes.append(("test_shift", "shift"))
    print(f"  channels={C} train={bm['meta']['n_train']} "
          f"indist={bm['meta']['n_indist']} shift={bm['meta']['n_shift']}")

    # ---- leakage probe (benchmark soundness; model-independent) ----
    probe = {reg: leakage_probe(bm[key]) for key, reg in regimes}
    print("\nleakage probe (best single privileged predicate accuracy; want ~0.5 for depth>=2):")
    for reg, pd in probe.items():
        print(f"  {reg}: " + "  ".join(f"d{d}={v:.3f}" for d, v in sorted(pd.items())))

    # ---- faithfulness over seeds ----
    per_seed = {reg: [] for _, reg in regimes}
    for seed in seeds:
        print(f"seed {seed}: training perception ...", flush=True)
        pres = train_perception(
            bm["train"], n_channels=C, hidden=cfg["perception"]["hidden"],
            kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
            per_channel=cfg["perception"]["per_channel"], epochs=cfg["perception"]["epochs"],
            batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
            weight_decay=cfg["perception"]["weight_decay"],
            device_pref=cfg.get("device", "cpu"), seed=seed,
            log_every=cfg["perception"].get("log_every", 20), verbose=False,
        )
        nst = LearnedNSTQA(pres.model, n_channels=C)
        for key, reg in regimes:
            per_seed[reg].append(faithfulness_report(nst, bm[key]))

    # ---- aggregate ----
    def agg(reg, metric, depth=None):
        vals = []
        for rep in per_seed[reg]:
            src = rep["by_depth"].get(depth) if depth is not None else rep
            if src is not None:
                vals.append(src.get(metric))
        return mean_std(vals)

    depths = sorted(b["depths"])
    out_dir = ROOT / cfg["run_root"]
    out_dir.mkdir(parents=True, exist_ok=True)

    L = [f"# Explanation faithfulness ({len(seeds)} seeds): {bm['meta']['dataset']} "
         f"shift={b['shift']}", "",
         "| Metric | " + " | ".join(reg for _, reg in regimes) + " |",
         "|---" * (len(regimes) + 1) + "|"]
    for key, label in METRICS:
        cells = " | ".join(f"{agg(reg, key)[0]:.3f} ± {agg(reg, key)[1]:.3f}"
                           for _, reg in regimes)
        L.append(f"| {label} | {cells} |")
    L += ["", "### Supporting-predicate accuracy by depth", "",
          "| regime | " + " | ".join(f"d{d}" for d in depths) + " |",
          "|---" * (len(depths) + 1) + "|"]
    for _, reg in regimes:
        cells = " | ".join(f"{agg(reg, 'supporting_predicate_accuracy', d)[0]:.3f}" for d in depths)
        L.append(f"| {reg} | {cells} |")
    L += ["", "### Counterfactual validity by depth", "",
          "| regime | " + " | ".join(f"d{d}" for d in depths) + " |",
          "|---" * (len(depths) + 1) + "|"]
    for _, reg in regimes:
        cells = " | ".join(f"{agg(reg, 'counterfactual_validity', d)[0]:.3f}" for d in depths)
        L.append(f"| {reg} | {cells} |")
    L += ["", "### Leakage probe (best single predicate acc; want ~0.5 for depth>=2)", "",
          "| regime | " + " | ".join(f"d{d}" for d in depths) + " |",
          "|---" * (len(depths) + 1) + "|"]
    for _, reg in regimes:
        cells = " | ".join(f"{probe[reg].get(d, float('nan')):.3f}" for d in depths)
        L.append(f"| {reg} | {cells} |")
    table = "\n".join(L)
    (out_dir / "faithfulness_table.md").write_text(table)
    (out_dir / "faithfulness.json").write_text(json.dumps({
        "metrics": {reg: {k: list(agg(reg, k)) for k, _ in METRICS} for _, reg in regimes},
        "by_depth": {reg: {k: {d: list(agg(reg, k, d)) for d in depths}
                           for k in ("supporting_predicate_accuracy", "counterfactual_validity")}
                     for _, reg in regimes},
        "leakage_probe": {reg: probe[reg] for _, reg in regimes},
        "seeds": seeds, "shift": b["shift"], "dataset": bm["meta"]["dataset"],
    }, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote table + json to {out_dir}")


if __name__ == "__main__":
    main()
