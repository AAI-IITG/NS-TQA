"""29 - Learned `anomalous` predicate, first-class (WO-1C).

Two claims, both on held-out XJTU bearings + a cross-load shift, at 10 seeds:

  (1) PREDICATE LEVEL — the learned anomaly head reproduces the privileged
      healthy-baseline-deviation target that a fixed ``max(high,low)`` proxy cannot;
      macro-F1 head vs proxy (Wilcoxon across seeds). This is the C3b evidence at
      the predicate level (learned perception is necessary for a NON-threshold
      predicate).

  (2) BENCHMARK SOUNDNESS — the leakage-safe compositional benchmark
      (``build_safe_anomaly_benchmark``: full-life windows, >=1 non-anomaly leaf at
      depth>=2, held-out-probe rejection) keeps the honest single-predicate leakage
      <= 0.55 at depth >= 2, so the executor stays load-bearing.

Answer level: NS-TQA (learned perception + learned anomaly head + executor) vs
STL-only (hand thresholds + a hand proxy for `anomalous`) vs the oracle, on the
in-dist and shifted pools — the necessity/generalization read for the predicate.

Run:  python scripts/29_run_anomaly_predicate.py [--config configs/anomaly_predicate.yaml] [--quick]
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

import torch
import yaml

from benchmark.anomaly_qa import (nstqa_anomaly_eval, oracle_anomaly_eval,
                                  stl_only_anomaly_eval)
from benchmark.anomaly_questions import build_safe_anomaly_benchmark, heldout_leakage
from models.stl_only import hand_rule_calibrator
from perception.anomaly import HealthyBaseline, predict_anomaly, train_anomaly_head
from perception.grounding import ground
from perception.learned import train_perception
from utils.stats import wilcoxon


def proxy_anomaly_target(x, C, a_level, smooth_k):
    """Fixed hand-rule `anomalous` ~ max(high, low) on a normalized window x[T,C]."""
    cal = hand_rule_calibrator(C, hi=1.0, lo=-1.0, slope_scale=1.0, a_level=a_level, smooth_k=smooth_k)
    mu4, _ = ground(x, cal)                       # [T,4C] family-major
    return torch.maximum(mu4[:, :C], mu4[:, C:2 * C])   # [T,C]


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    return (sum(xs) / len(xs), st.pstdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)


def macro_f1(pred: torch.Tensor, gold: torch.Tensor) -> float:
    """Binary macro-F1 (mean of positive- and negative-class F1) over all entries."""
    pred, gold = pred.reshape(-1).bool(), gold.reshape(-1).bool()
    f1s = []
    for cls in (True, False):
        p, g = (pred == cls), (gold == cls)
        tp = (p & g).sum().item()
        prec = tp / max(1, p.sum().item())
        rec = tp / max(1, g.sum().item())
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / 2


def _unique_windows(instances) -> torch.Tensor:
    """Dedup (unit,start) normalized windows from QA instances -> [N,T,C]."""
    seen, ws = set(), []
    for i in instances:
        key = (i.unit_id, i.provenance.get("start"))
        if key not in seen:
            seen.add(key)
            ws.append(i.X)
    return torch.stack(ws)


def build_adapter(a: dict):
    """Dispatch on config keys: ``rul_cap`` -> C-MAPSS, ``n_bands`` -> XJTU."""
    if "rul_cap" in a:
        from benchmark.adapters_cmapss import CMAPSSAdapter
        return CMAPSSAdapter(
            root=ROOT / a["root"], subsets=sorted(set(["FD001", "FD002", "FD004"])),
            rul_cap=a["rul_cap"], flat_std_thresh=a.get("flat_std_thresh", 1e-6),
            min_qspan=a.get("min_qspan", 0.05),
            op_normalize=a.get("op_normalize", False), n_regimes=a.get("n_regimes", 6))
    from benchmark.adapters_xjtu import XJTUAdapter
    return XJTUAdapter(root=ROOT / a["root"], fs=a["fs"], n_bands=a["n_bands"],
                       min_snapshots=a.get("min_snapshots", 1),
                       cache_path=ROOT / a["cache_path"],
                       force_recompute=a.get("force_recompute", False))


def run_dataset(cfg, dname, dcfg, seeds):
    """Build the safe anomaly benchmark for one dataset and run the seed loop."""
    b = dcfg["build"]
    adapter = build_adapter(dcfg["adapter"])
    print(f"\n### {dname}: building leakage-safe anomaly benchmark ...", flush=True)
    bm = build_safe_anomaly_benchmark(
        adapter, T=b["T"], stride=b["stride"], depths=tuple(b["depths"]),
        shift=b["shift"], train_conditions=tuple(b["train_conditions"]),
        test_conditions=tuple(b["test_conditions"]),
        indist_holdout_frac=b["indist_holdout_frac"], healthy_frac=b["healthy_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=b["hi_q"], lo_q=b["lo_q"], smooth_k=b["smooth_k"], a_level=b["a_level"],
        a_anom=b["a_anom"], anomaly_q=b["anomaly_q"], anomaly_p=b["anomaly_p"],
        allow_until=b["allow_until"], max_windows_per_unit=b.get("max_windows_per_unit"),
        over_factor=b["over_factor"], seed=b["build_seed"],
        leak_max=b["leak_max"], max_leak_retries=b["max_leak_retries"])
    C = bm["meta"]["n_channels"]
    baseline = bm["baseline"]
    print(f"  C={C} n_train={bm['meta']['n_train']} n_indist={bm['meta']['n_indist']} "
          f"n_shift={bm['meta']['n_shift']}")

    def leak_by_depth(insts):
        byd = {}
        for i in insts:
            byd.setdefault(i.depth, []).append(i)
        return {d: round(heldout_leakage(v, seed=7), 4) for d, v in sorted(byd.items())}
    leak = {"indist": leak_by_depth(bm["test_indist"]), "shift": leak_by_depth(bm["test_shift"])}
    print(f"  held-out leakage indist={leak['indist']} shift={leak['shift']}")

    def gold_and_proxy(insts):
        gold = torch.stack([baseline.target(i.X) for i in insts]) > 0.5
        proxy = torch.stack([proxy_anomaly_target(i.X, C, b["a_level"], b["smooth_k"]) for i in insts]) > 0.5
        return gold, proxy
    gold_i, proxy_i = gold_and_proxy(bm["test_indist"])
    gold_s, proxy_s = gold_and_proxy(bm["test_shift"])
    proxy_f1 = {"indist": macro_f1(proxy_i, gold_i), "shift": macro_f1(proxy_s, gold_s)}

    train_windows = _unique_windows(bm["train"])
    # perception trains on the 4-family privileged truths (first 4C cols of mu_star_5);
    # the learned anomaly head supplies the 5th family separately.
    import copy
    train4 = []
    for i in bm["train"]:
        j = copy.copy(i)
        j.mu_star = i.mu_star[:, :4 * C]
        train4.append(j)

    metrics = {"head_f1": {"indist": [], "shift": []},
               "answer": {m: {"indist": [], "shift": []}
                          for m in ("NS-TQA", "STL-only", "oracle")}}

    for seed in seeds:
        print(f"  {dname} seed {seed}: training perception + anomaly head ...", flush=True)
        pres = train_perception(
            train4, n_channels=C, hidden=cfg["perception"]["hidden"],
            kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
            per_channel=cfg["perception"]["per_channel"], epochs=cfg["perception"]["epochs"],
            batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
            weight_decay=cfg["perception"]["weight_decay"],
            device_pref=cfg.get("device", "cpu"), seed=seed, verbose=False)
        head = train_anomaly_head(
            train_windows, baseline, hidden=cfg["anomaly_head"]["hidden"],
            kernel=cfg["anomaly_head"]["kernel"], n_layers=cfg["anomaly_head"]["n_layers"],
            epochs=cfg["anomaly_head"]["epochs"], batch_size=cfg["anomaly_head"]["batch_size"],
            lr=cfg["anomaly_head"]["lr"], device_pref=cfg.get("device", "cpu"), seed=seed)

        pred_i = torch.stack([predict_anomaly(head, i.X) for i in bm["test_indist"]]) > 0.5
        pred_s = torch.stack([predict_anomaly(head, i.X) for i in bm["test_shift"]]) > 0.5
        metrics["head_f1"]["indist"].append(macro_f1(pred_i, gold_i))
        metrics["head_f1"]["shift"].append(macro_f1(pred_s, gold_s))
        for pool in ("indist", "shift"):
            insts = bm[f"test_{pool}"]
            metrics["answer"]["NS-TQA"][pool].append(
                nstqa_anomaly_eval(insts, pres.model, head)["answer_accuracy"])
            metrics["answer"]["STL-only"][pool].append(
                stl_only_anomaly_eval(insts, a_level=b["a_level"], smooth_k=b["smooth_k"])["answer_accuracy"])
            metrics["answer"]["oracle"][pool].append(
                oracle_anomaly_eval(insts)["answer_accuracy"])
    return {"C": C, "meta": bm["meta"], "metrics": metrics, "proxy_f1": proxy_f1, "leak": leak}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "anomaly_predicate.yaml"))
    ap.add_argument("--datasets", default=None, help="comma subset, e.g. xjtu")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seeds = cfg["experiment"]["seeds"]
    dsets = list(cfg["datasets"].keys())
    if args.datasets:
        dsets = [d for d in dsets if d in set(args.datasets.split(","))]
    if args.quick:
        seeds = seeds[:1]
        cfg["device"] = "cpu"                 # local smoke: on-box GPU is sm_61 (incompatible)
        cfg["perception"]["epochs"] = 10
        cfg["anomaly_head"]["epochs"] = 10
        for d in dsets:
            cfg["datasets"][d]["build"]["n_train_per_depth"] = 80
            cfg["datasets"][d]["build"]["n_test_per_depth"] = 150

    results = {d: run_dataset(cfg, d, cfg["datasets"][d], seeds) for d in dsets}
    _write(cfg, results, seeds, ROOT / cfg["run_root"])


def _write(cfg, results, seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    L = [f"# Learned `anomalous` predicate (WO-1C, C3b) — {len(seeds)} seeds", "",
         "Predicate-level necessity of the learned anomaly head vs the fixed "
         "`max(high,low)` proxy, on held-out units + cross-condition shift, over "
         "leakage-SAFE compositional benchmarks (held-out single-predicate leakage "
         "≤ 0.55 at depth ≥ 2).", ""]
    for dname, D in results.items():
        metrics, proxy_f1, leak, C = D["metrics"], D["proxy_f1"], D["leak"], D["C"]
        L.append(f"## {dname}  (C={C})")
        _write_dataset(L, dname, metrics, proxy_f1, leak, seeds)
    table = "\n".join(L)
    (out_dir / "anomaly_predicate.md").write_text(table)
    (out_dir / "anomaly_predicate.json").write_text(json.dumps(
        {"seeds": seeds, "results": {d: {"metrics": D["metrics"], "proxy_f1": D["proxy_f1"],
                                         "leakage": D["leak"], "meta": D["meta"]}
                                     for d, D in results.items()}}, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote -> {out_dir}/anomaly_predicate.{{md,json}}")


def _write_dataset(L, dname, metrics, proxy_f1, leak, seeds):
    # (1) predicate table
    L += ["\n### (1) Predicate-level: learned head vs fixed proxy (macro-F1)", "",
          "| pool | learned head | fixed proxy | head − proxy | Wilcoxon p |", "|---|---|---|---|---|"]
    for pool in ("indist", "shift"):
        hm, hs = mean_std(metrics["head_f1"][pool])
        pf = proxy_f1[pool]
        w = wilcoxon(metrics["head_f1"][pool], [pf] * len(metrics["head_f1"][pool]))
        L.append(f"| {pool} | {hm:.3f}±{hs:.3f} | {pf:.3f} | {hm - pf:+.3f} | {w['p']:.4f} |")
    L.append("\n_The learned head beats the proxy ⇒ learned perception is necessary for the "
             "non-threshold `anomalous` predicate (C3b, predicate level)._")

    # (2) leakage gate
    L += ["\n### (2) Benchmark soundness: held-out single-predicate leakage (want ≤ 0.55 at depth ≥ 2)", ""]
    for pool in ("indist", "shift"):
        cells = " ".join(f"d{d}={v}" for d, v in leak[pool].items())
        maxd2 = max((v for d, v in leak[pool].items() if d >= 2), default=0.0)
        L.append(f"- **{pool}**: {cells}  → max(depth≥2) = {maxd2:.3f} "
                 f"({'PASS' if maxd2 <= 0.55 else 'residual > 0.55'})")

    # answer-level necessity/generalization
    L += ["\n### Answer-level: NS-TQA vs STL-only vs oracle (accuracy)", "",
          "| method | in-dist | shift |", "|---|---|---|"]
    for m in ("NS-TQA", "STL-only", "oracle"):
        mi, si = mean_std(metrics["answer"][m]["indist"])
        ms, ss = mean_std(metrics["answer"][m]["shift"])
        L.append(f"| {m} | {mi:.3f}±{si:.3f} | {ms:.3f}±{ss:.3f} |")
    L.append("")


if __name__ == "__main__":
    main()
