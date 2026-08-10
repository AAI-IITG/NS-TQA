"""36 - Sensitivity of the NS-TQA gap to design choices (WO-4).

The headline claim is that NS-TQA's robustness gap over end-to-end baselines is not an
artefact of one lucky hyper-parameter setting. We vary each grounding/window factor ONE
AT A TIME around the default, rebuild the (non-circular) benchmark, retrain NS-TQA and
the best end-to-end baseline (TCN), and record shifted answer accuracy over seeds:

  hi_q/lo_q  in {(0.80,0.20),(0.85,0.15),(0.90,0.10)}   quantile thresholds
  a_level    in {2,4,8}                                  grounding sharpness
  T          in {32,48,64}                               window length
  smooth_k   in {3,5,9}                                  grounding smoothing

For each cell we report NS-TQA and TCN shifted accuracy and the gap; a paired Wilcoxon
(NS-TQA vs TCN across seeds) annotates each. The claim to verify: the gap survives every
setting (and if some setting kills it, that is reported, not hidden).

Run:  python scripts/36_run_sensitivity.py [--config configs/sensitivity.yaml] [--datasets xjtu,cmapss] [--quick]
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

from benchmark.baseline import select_device
from benchmark.necessity import eval_baseline, eval_nstqa, oracle_accuracy, train_baseline
from benchmark.realdata import build_real_benchmark
from models.nstqa_learned import LearnedNSTQA
from perception.grounding import predicate_index
from perception.learned import train_perception
from utils.stats import wilcoxon

FACTORS = {
    "hi_lo_q": [(0.80, 0.20), (0.85, 0.15), (0.90, 0.10)],
    "a_level": [2.0, 4.0, 8.0],
    "T": [32, 48, 64],
    "smooth_k": [3, 5, 9],
}


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


def build_bm(dcfg, overrides):
    a, b = dcfg["adapter"], dict(dcfg["build"])
    hi_q = a.get("hi_q", 0.85); lo_q = a.get("lo_q", 0.15)
    b.update({k: v for k, v in overrides.items() if k in b})
    if "hi_q" in overrides:
        hi_q, lo_q = overrides["hi_q"], overrides["lo_q"]
    return build_real_benchmark(
        build_adapter(a), T=b["T"], stride=b["stride"], depths=tuple(b["depths"]),
        shift="condition", train_conditions=tuple(b["train_conditions"]),
        test_conditions=tuple(b["test_conditions"]), indist_holdout_frac=b["indist_holdout_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=hi_q, lo_q=lo_q, smooth_k=b["smooth_k"], a_level=b["a_level"],
        allow_until=b["allow_until"], max_windows_per_unit=b.get("max_windows_per_unit"),
        over_factor=b["over_factor"], seed=b["build_seed"])


def eval_cell(cfg, dcfg, overrides, seeds, device, quick):
    bm = build_bm(dcfg, overrides)
    C = bm["meta"]["n_channels"]; T = bm["meta"]["T"]
    pidx = predicate_index(C)
    if not bm["test_shift"]:
        return None
    orc = oracle_accuracy(bm["test_shift"], pidx)["answer_accuracy"]
    ns, tc = [], []
    for seed in seeds:
        tcn = train_baseline("tcn", bm["train"], C, T, cfg["lstm"], device, seed, verbose=False)
        pres = train_perception(
            bm["train"], n_channels=C, hidden=cfg["perception"]["hidden"],
            kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
            per_channel=cfg["perception"]["per_channel"],
            epochs=10 if quick else cfg["perception"]["epochs"],
            batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
            weight_decay=cfg["perception"]["weight_decay"],
            device_pref=cfg.get("device", "cpu"), seed=seed, verbose=False)
        nst = LearnedNSTQA(pres.model, n_channels=C)
        ns.append(eval_nstqa(nst, bm["test_shift"])["answer_accuracy"])
        tc.append(eval_baseline(tcn, bm["test_shift"], C, T, device)["answer_accuracy"])
    w = wilcoxon(ns, tc) if len(ns) >= 2 else {"p": float("nan")}
    return {"ns": ns, "tcn": tc, "oracle": round(orc, 4),
            "ns_ms": mean_std(ns), "tcn_ms": mean_std(tc), "p": w["p"],
            "gap": mean_std(ns)[0] - mean_std(tc)[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "sensitivity.yaml"))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--factors", default=None, help="comma subset of hi_lo_q,a_level,T,smooth_k")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seeds = cfg["experiment"]["seeds"][:2 if args.quick else None]
    device = select_device(cfg.get("device", "cpu"))
    dsets = list(cfg["datasets"].keys())
    if args.datasets:
        dsets = [d for d in dsets if d in set(args.datasets.split(","))]
    factors = FACTORS if not args.factors else {k: FACTORS[k] for k in args.factors.split(",")}

    results = {}
    for dname in dsets:
        dcfg = cfg["datasets"][dname]
        default = {"hi_q": dcfg["adapter"].get("hi_q", 0.85), "lo_q": dcfg["adapter"].get("lo_q", 0.15),
                   "a_level": dcfg["build"]["a_level"], "T": dcfg["build"]["T"],
                   "smooth_k": dcfg["build"]["smooth_k"]}
        results[dname] = {}
        for fac, values in factors.items():
            for v in values:
                if fac == "hi_lo_q":
                    ov = {"hi_q": v[0], "lo_q": v[1]}; label = f"{v[0]}/{v[1]}"
                else:
                    ov = {fac: v}; label = str(v)
                is_default = all(default.get(k) == vv for k, vv in ov.items())
                print(f"\n[{dname}] {fac}={label}{' (default)' if is_default else ''}", flush=True)
                cell = eval_cell(cfg, dcfg, ov, seeds, device, args.quick)
                if cell:
                    print(f"    NS-TQA={cell['ns_ms'][0]:.3f} TCN={cell['tcn_ms'][0]:.3f} "
                          f"gap={cell['gap']:+.3f} p={cell['p']:.3f} oracle={cell['oracle']}", flush=True)
                    results[dname].setdefault(fac, {})[label] = {**cell, "default": is_default}
    _write(cfg, results, seeds, ROOT / cfg["run_root"])


def _write(cfg, results, seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    L = [f"# Sensitivity of the NS-TQA gap (WO-4) — {len(seeds)} seeds", "",
         "NS-TQA vs the best end-to-end baseline (TCN), shifted answer accuracy, varying each "
         "grounding/window factor one at a time. `gap` = NS-TQA − TCN; `p` = paired Wilcoxon.", ""]
    min_gap = {}
    for dname, facs in results.items():
        L.append(f"## {dname}")
        for fac, cells in facs.items():
            L += ["", f"### {fac}", "", "| value | NS-TQA | TCN | gap | Wilcoxon p | oracle |",
                  "|---|---|---|---|---|---|"]
            for label, c in cells.items():
                d = " *(default)*" if c["default"] else ""
                L.append(f"| {label}{d} | {c['ns_ms'][0]:.3f}±{c['ns_ms'][1]:.3f} | "
                         f"{c['tcn_ms'][0]:.3f}±{c['tcn_ms'][1]:.3f} | {c['gap']:+.3f} | "
                         f"{c['p']:.3f} | {c['oracle']} |")
                min_gap[dname] = min(min_gap.get(dname, 9), c["gap"])
        L.append("")
    L.append("## Summary")
    for dname, g in min_gap.items():
        verdict = "gap SURVIVES every setting" if g > 0 else f"gap DISAPPEARS in some setting (min {g:+.3f})"
        L.append(f"- **{dname}**: minimum NS-TQA−TCN gap across the grid = {g:+.3f} → {verdict}.")
    table = "\n".join(L)
    (out_dir / "sensitivity.md").write_text(table)
    (out_dir / "sensitivity.json").write_text(json.dumps(
        {"seeds": seeds, "results": results}, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote -> {out_dir}/sensitivity.{{md,json}}")


if __name__ == "__main__":
    main()
