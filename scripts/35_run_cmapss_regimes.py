"""35 - C-MAPSS shift decomposition + per-regime normalisation (WO-3A).

C-MAPSS lets us FACTORISE the distribution shift, because its four subsets vary two
things independently:

                     | 1 fault mode | 2 fault modes
    -----------------|--------------|---------------
    1 op-regime      | FD001 (train)| FD003
    6 op-regimes     | FD002        | FD004

so FD001 -> FD003 is a pure FAULT-MODE shift, FD001 -> FD002 is a pure OPERATING-REGIME
shift, and FD001 -> FD004 is both. We evaluate every method on each target separately,
under two input normalisations:

  * global     : one z-score per channel over the training subset (the conference setup);
  * per-regime : cluster the 3 operating-setting columns into k regimes (k-means) and
                 z-score each sensor WITHIN its regime (``op_normalize``), which removes
                 the regime-induced level jumps that dominate FD002/FD004.

Prediction under test: per-regime normalisation should recover the REGIME axis
(FD002/FD004) and do little for the FAULT-MODE axis (FD003).

Run:  python scripts/35_run_cmapss_regimes.py [--config configs/cmapss_regimes.yaml] [--quick]
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

from benchmark.adapters_cmapss import CMAPSSAdapter
from benchmark.baseline import select_device
from benchmark.necessity import (eval_baseline, eval_nstqa, oracle_accuracy,
                                 train_baseline)
from benchmark.realdata import build_real_benchmark, real_balance_report
from models.nstqa_learned import LearnedNSTQA
from models.stl_only import stl_only_evaluate
from perception.grounding import predicate_index
from perception.learned import predicate_metrics, train_perception
from utils.faithfulness import leakage_probe

TARGETS = ["FD002", "FD003", "FD004"]


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    return (sum(xs) / len(xs), st.pstdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)


def build_bm(cfg, op_normalize):
    a, b = cfg["adapter"], cfg["build"]
    adapter = CMAPSSAdapter(
        root=ROOT / a["root"], subsets=("FD001",) + tuple(TARGETS), rul_cap=a["rul_cap"],
        flat_std_thresh=a.get("flat_std_thresh", 1e-6), min_qspan=a.get("min_qspan", 0.05),
        op_normalize=op_normalize, n_regimes=a.get("n_regimes", 6))
    return build_real_benchmark(
        adapter, T=b["T"], stride=b["stride"], depths=tuple(b["depths"]),
        shift="condition", train_conditions=("FD001",), test_conditions=tuple(TARGETS),
        indist_holdout_frac=b["indist_holdout_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=a["hi_q"], lo_q=a["lo_q"], smooth_k=b["smooth_k"], a_level=b["a_level"],
        allow_until=b["allow_until"], max_windows_per_unit=b.get("max_windows_per_unit"),
        over_factor=b["over_factor"], seed=b["build_seed"])


def run_arm(cfg, arm, op_normalize, seeds, device, quick):
    print(f"\n########## normalisation arm: {arm} (op_normalize={op_normalize}) ##########",
          flush=True)
    bm = build_bm(cfg, op_normalize)
    C, T = bm["meta"]["n_channels"], cfg["build"]["T"]
    pidx = predicate_index(C)
    baselines = cfg["experiment"]["baselines"][:1] if quick else cfg["experiment"]["baselines"]

    pools = {"FD001 (held-out)": bm["test_indist"]}
    for t in TARGETS:
        pools[t] = [i for i in bm["test_shift"] if i.condition == t]
    print("  pools: " + ", ".join(f"{k}:{len(v)}" for k, v in pools.items()), flush=True)

    # ---- invariants per build (AGENTS.md sec.2) ----
    inv = {}
    for name, insts in pools.items():
        if not insts:
            continue
        bal = real_balance_report(insts)
        orc = oracle_accuracy(insts, pidx)["answer_accuracy"]
        leak = leakage_probe(insts)
        leak_d2 = mean_std([v["accuracy"] if isinstance(v, dict) else v
                            for d, v in leak.items() if isinstance(d, int) and d >= 2])[0]
        inv[name] = {"yes_frac": round(bal["yes_frac"], 3), "oracle": round(orc, 4),
                     "leak_d2": round(leak_d2, 3) if leak_d2 == leak_d2 else None,
                     "n": len(insts)}
        print(f"    [{name}] yes_frac={inv[name]['yes_frac']} oracle={inv[name]['oracle']} "
              f"leak(d>=2)={inv[name]['leak_d2']}", flush=True)

    methods = baselines + ["NS-TQA", "STL-only"]
    cell = {m: {k: [] for k in pools} for m in methods}
    pf1 = {k: [] for k in pools}

    for name, insts in pools.items():          # STL-only: deterministic, once
        if insts:
            cell["STL-only"][name].append(
                stl_only_evaluate(insts, C, a_level=cfg["build"]["a_level"],
                                  smooth_k=cfg["build"]["smooth_k"])["answer_accuracy"])

    for seed in seeds:
        print(f"  seed {seed}: training ...", flush=True)
        base = {n: train_baseline(n, bm["train"], C, T, cfg["lstm"], device, seed, verbose=False)
                for n in baselines}
        pres = train_perception(
            bm["train"], n_channels=C, hidden=cfg["perception"]["hidden"],
            kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
            per_channel=cfg["perception"]["per_channel"],
            epochs=10 if quick else cfg["perception"]["epochs"],
            batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
            weight_decay=cfg["perception"]["weight_decay"],
            device_pref=cfg.get("device", "cpu"), seed=seed, verbose=False)
        nst = LearnedNSTQA(pres.model, n_channels=C)
        for name, insts in pools.items():
            if not insts:
                continue
            for bn, model in base.items():
                cell[bn][name].append(eval_baseline(model, insts, C, T, device)["answer_accuracy"])
            cell["NS-TQA"][name].append(eval_nstqa(nst, insts)["answer_accuracy"])
            pf1[name].append(predicate_metrics(pres.model, insts)["macro_f1"])

    return {"pools": list(pools), "cell": cell, "pf1": pf1, "inv": inv,
            "methods": methods, "C": C}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "cmapss_regimes.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seeds = cfg["experiment"]["seeds"][:1 if args.quick else None]
    device = select_device(cfg.get("device", "cpu"))
    arms = {"global": False, "per-regime": True}
    res = {a: run_arm(cfg, a, on, seeds, device, args.quick) for a, on in arms.items()}
    _write(cfg, res, seeds, ROOT / cfg["run_root"])


def _write(cfg, res, seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    L = [f"# C-MAPSS shift decomposition + per-regime normalisation (WO-3A) — {len(seeds)} seeds", "",
         "FD001 is the train subset. The targets factorise the shift:",
         "",
         "| | 1 fault mode | 2 fault modes |",
         "|---|---|---|",
         "| **1 op-regime** | FD001 (train) | **FD003** = fault-mode shift |",
         "| **6 op-regimes** | **FD002** = regime shift | **FD004** = both |",
         ""]
    for arm, D in res.items():
        L.append(f"## Normalisation: {arm}   (C={D['C']})")
        cols = D["pools"]
        L += ["", "| method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for m in D["methods"]:
            cells = []
            for c in cols:
                mu, sd = mean_std(D["cell"][m][c])
                cells.append("—" if mu != mu else
                             (f"{mu:.3f}" if m == "STL-only" else f"{mu:.3f}±{sd:.3f}"))
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        f1 = [mean_std(D["pf1"][c])[0] for c in cols]
        L.append("| _perception macro-F1_ | " + " | ".join(
            ("—" if v != v else f"_{v:.3f}_") for v in f1) + " |")
        L += ["", "_invariants (per pool):_ " + "; ".join(
            f"{k}: yes={v['yes_frac']}, oracle={v['oracle']}, leak(d≥2)={v['leak_d2']}"
            for k, v in D["inv"].items()), ""]

    # the headline comparison: does per-regime normalisation fix the REGIME axis?
    L += ["## Effect of per-regime normalisation (NS-TQA)", "",
          "| target | shift type | global | per-regime | Δ |", "|---|---|---|---|---|"]
    kind = {"FD002": "operating regime", "FD003": "fault mode", "FD004": "both"}
    for t in TARGETS:
        g = mean_std(res["global"]["cell"]["NS-TQA"].get(t, []))[0]
        p = mean_std(res["per-regime"]["cell"]["NS-TQA"].get(t, []))[0]
        d = p - g if (g == g and p == p) else float("nan")
        L.append(f"| {t} | {kind[t]} | {g:.3f} | {p:.3f} | {d:+.3f} |")
    table = "\n".join(L)
    (out_dir / "cmapss_regimes.md").write_text(table)
    (out_dir / "cmapss_regimes.json").write_text(json.dumps(
        {"seeds": seeds, "results": {a: {"cell": D["cell"], "pf1": D["pf1"], "inv": D["inv"]}
                                     for a, D in res.items()}}, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote -> {out_dir}/cmapss_regimes.{{md,json}}")


if __name__ == "__main__":
    main()
