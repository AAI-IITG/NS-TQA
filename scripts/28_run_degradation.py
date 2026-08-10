"""28 - Degraded-sensing robustness study (WO-1B): learned vs fixed grounding.

At what sensing-degradation severity, if any, does a LEARNED perception head beat a
fixed-threshold STL-only grounding? Degradation (noise / dropped samples /
calibration drift) is applied to the model INPUT of the TEST pools only; the
privileged label stays defined on the CLEAN grounding (non-circular — see
``src/benchmark/degrade.py``). For each dataset we build the benchmark once, train
each method, and evaluate every method on the clean pool and on every degraded pool.

Methods compared:
  * STL-only            fixed hand thresholds + executor (no learning, deterministic)
  * NS-TQA              learned perception (trained on CLEAN train windows) + executor
  * NS-TQA-aug          learned perception trained WITH light noise augmentation (the
                        deployment argument): perception sees SNR-``aug_snr_db`` noise
  * lstm/transformer/tcn end-to-end baselines (best reported)

Run:  python scripts/28_run_degradation.py [--config configs/degradation.yaml]
              [--datasets cmapss,xjtu] [--quick]
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

from benchmark.baseline import select_device
from benchmark.degrade import degrade_instances
from benchmark.necessity import (eval_baseline, eval_nstqa, oracle_accuracy,
                                 train_baseline)
from benchmark.realdata import build_real_benchmark
from models.nstqa_learned import LearnedNSTQA
from models.stl_only import stl_only_evaluate
from perception.grounding import predicate_index
from perception.learned import train_perception
from utils.stats import wilcoxon


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return (float("nan"), 0.0)
    return (sum(xs) / len(xs), stats.pstdev(xs) if len(xs) > 1 else 0.0)


def build_adapter(a: dict):
    """Dispatch on config keys: ``rul_cap`` -> C-MAPSS, ``n_bands`` -> XJTU."""
    if "rul_cap" in a:
        from benchmark.adapters_cmapss import CMAPSSAdapter
        subsets = sorted(set(["FD001", "FD002", "FD004"]))
        return CMAPSSAdapter(
            root=ROOT / a["root"], subsets=subsets, rul_cap=a["rul_cap"],
            flat_std_thresh=a.get("flat_std_thresh", 1e-6), min_qspan=a.get("min_qspan", 0.05))
    from benchmark.adapters_xjtu import XJTUAdapter
    return XJTUAdapter(root=ROOT / a["root"], fs=a["fs"], n_bands=a["n_bands"],
                       min_snapshots=a.get("min_snapshots", 1),
                       cache_path=ROOT / a["cache_path"],
                       force_recompute=a.get("force_recompute", False))


def build_bm(dcfg: dict):
    a, b = dcfg["adapter"], dcfg["build"]
    adapter = build_adapter(a)
    bm = build_real_benchmark(
        adapter, T=b["T"], stride=b["stride"], depths=tuple(b["depths"]),
        shift=b["shift"], train_conditions=tuple(b["train_conditions"]),
        test_conditions=tuple(b["test_conditions"]),
        indist_holdout_frac=b["indist_holdout_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=a["hi_q"], lo_q=a["lo_q"], smooth_k=b["smooth_k"], a_level=b["a_level"],
        allow_until=b["allow_until"], max_windows_per_unit=b.get("max_windows_per_unit"),
        over_factor=b["over_factor"], seed=b["build_seed"])
    return bm, b


def make_degraded_pools(clean_pools: dict, sweep: dict, build_seed: int) -> dict:
    """(regime, family, sev_label) -> degraded instances. Degraded ONCE, shared by
    all methods & model seeds (corruption is a property of the benchmark)."""
    pools = {}
    for regime, insts in clean_pools.items():
        if not insts:
            continue
        for family, levels in sweep.items():
            for lv in levels:
                params = {k: v for k, v in lv.items() if k != "label"}
                pools[(regime, family, lv["label"])] = degrade_instances(
                    insts, family, seed=build_seed, **params)
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "degradation.yaml"))
    ap.add_argument("--datasets", default=None, help="comma list subset, e.g. xjtu")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    exp = cfg["experiment"]
    seeds, baselines = exp["seeds"], exp["baselines"]
    aug_snr = exp["aug_snr_db"]
    sweep = cfg["sweep"]
    device = select_device(cfg.get("device", "cpu"))
    dsets = list(cfg["datasets"].keys())
    if args.datasets:
        dsets = [d for d in dsets if d in set(args.datasets.split(","))]
    if args.quick:
        seeds = seeds[:1]
        baselines = ["lstm"]
        cfg["lstm"]["epochs"] = 5
        cfg["perception"]["epochs"] = 10
        sweep = {"noise": sweep["noise"][:2], "drift": sweep["drift"][:1]}

    methods = ["STL-only", "NS-TQA", "NS-TQA-aug"] + baselines
    # results[dataset][regime][family][sev_label][method] = [per-seed acc]  (+ "clean")
    results: dict = {}
    out_dir = ROOT / cfg["run_root"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for dname in dsets:
        print(f"\n########## dataset: {dname} ##########", flush=True)
        bm, b = build_bm(cfg["datasets"][dname])
        C, T = bm["meta"]["n_channels"], bm["meta"]["T"]
        pidx = predicate_index(C)
        bs = b["build_seed"]
        clean_pools = {r: bm[f"test_{r}"] for r in ("indist", "shift") if bm[f"test_{r}"]}
        print(f"  C={C} T={T} pools: " + ", ".join(f"{k}:{len(v)}" for k, v in clean_pools.items()))
        print(f"  oracle: " + ", ".join(
            f"{k}={oracle_accuracy(v, pidx)['answer_accuracy']:.3f}" for k, v in clean_pools.items()))
        deg_pools = make_degraded_pools(clean_pools, sweep, bs)

        # all pools to score: clean (label "clean") + degraded
        eval_pools = {(r, "clean", "clean"): insts for r, insts in clean_pools.items()}
        eval_pools.update(deg_pools)

        res = {}  # (regime,family,sev)->method->list
        for key in eval_pools:
            res[key] = {m: [] for m in methods}

        # --- STL-only: deterministic, evaluate once per pool ---
        for key, insts in eval_pools.items():
            res[key]["STL-only"].append(
                stl_only_evaluate(insts, C, a_level=b["a_level"], smooth_k=b["smooth_k"])["answer_accuracy"])

        # --- per-seed: train baselines + clean/aug perception, eval on all pools ---
        aug_train = degrade_instances(bm["train"], "noise", seed=bs, snr_db=aug_snr)
        for seed in seeds:
            print(f"  seed {seed}: training ...", flush=True)
            base_models = {name: train_baseline(name, bm["train"], C, T, cfg["lstm"],
                                                device, seed, verbose=False)
                           for name in baselines}
            pc = train_perception(bm["train"], n_channels=C, hidden=cfg["perception"]["hidden"],
                                  kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
                                  per_channel=cfg["perception"]["per_channel"], epochs=cfg["perception"]["epochs"],
                                  batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
                                  weight_decay=cfg["perception"]["weight_decay"],
                                  device_pref=cfg.get("device", "cpu"), seed=seed, verbose=False)
            pa = train_perception(aug_train, n_channels=C, hidden=cfg["perception"]["hidden"],
                                  kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
                                  per_channel=cfg["perception"]["per_channel"], epochs=cfg["perception"]["epochs"],
                                  batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
                                  weight_decay=cfg["perception"]["weight_decay"],
                                  device_pref=cfg.get("device", "cpu"), seed=seed, verbose=False)
            nst_c, nst_a = LearnedNSTQA(pc.model, n_channels=C), LearnedNSTQA(pa.model, n_channels=C)
            for key, insts in eval_pools.items():
                res[key]["NS-TQA"].append(eval_nstqa(nst_c, insts)["answer_accuracy"])
                res[key]["NS-TQA-aug"].append(eval_nstqa(nst_a, insts)["answer_accuracy"])
                for name, model in base_models.items():
                    res[key][name].append(eval_baseline(model, insts, C, T, device)["answer_accuracy"])
        results[dname] = {"res": res, "C": C, "T": T,
                          "clean_pools": {k: len(v) for k, v in clean_pools.items()}}

    _write_outputs(results, methods, baselines, sweep, seeds, out_dir)


def _best_e2e(cell: dict, baselines: list):
    means = [(mean_std(cell[b])[0]) for b in baselines]
    means = [m for m in means if m == m]
    return max(means) if means else float("nan")


def _write_outputs(results, methods, baselines, sweep, seeds, out_dir):
    blocks = [f"# Degraded-sensing robustness (WO-1B) — {len(seeds)} model seeds",
              "", "Degradation applied to TEST inputs only; labels defined on the clean "
              "privileged grounding (non-circular). STL-only is deterministic (1 eval). "
              "`p(NS>STL)` is a one-sample signed-rank test of the per-seed NS-TQA accuracies "
              "against the STL-only constant.", ""]
    for dname, D in results.items():
        res = D["res"]
        regimes = sorted({k[0] for k in res})
        blocks.append(f"## {dname}  (C={D['C']}, T={D['T']}, pools={D['clean_pools']})")
        # clean reference row
        for regime in regimes:
            ck = (regime, "clean", "clean")
            clean_line = " · ".join(
                f"{m} {mean_std(res[ck][m])[0]:.3f}" for m in methods)
            blocks.append(f"\n**{regime} — clean:** {clean_line}")
            for family in sweep:
                levels = [lv["label"] for lv in sweep[family]]
                head = f"| method | " + " | ".join(levels) + " |"
                sep = "|---" * (len(levels) + 1) + "|"
                lines = [f"\n#### {dname} / {regime} / {family}", "", head, sep]
                for m in methods:
                    cells = []
                    for lab in levels:
                        key = (regime, family, lab)
                        mu, sd = mean_std(res[key][m])
                        cells.append("—" if mu != mu else (f"{mu:.3f}" if m == "STL-only"
                                                           else f"{mu:.3f}±{sd:.3f}"))
                    lines.append(f"| {m} | " + " | ".join(cells) + " |")
                # best-e2e + Wilcoxon NS vs STL per severity
                be = [(_best_e2e(res[(regime, family, lab)], baselines)) for lab in levels]
                lines.append(f"| best-e2e | " + " | ".join(
                    ("—" if v != v else f"{v:.3f}") for v in be) + " |")
                wl = []
                for lab in levels:
                    key = (regime, family, lab)
                    ns = res[key]["NS-TQA"]
                    stl = res[key]["STL-only"][0] if res[key]["STL-only"] else float("nan")
                    if len(ns) >= 2 and stl == stl:
                        w = wilcoxon(ns, [stl] * len(ns))
                        d = mean_std(ns)[0] - stl
                        wl.append(f"{'+' if d>=0 else ''}{d:.3f} (p={w['p']:.3f})")
                    else:
                        wl.append("—")
                lines.append(f"| NS−STL (p) | " + " | ".join(wl) + " |")
                blocks.append("\n".join(lines))
        blocks.append("")

    # ---- verdict summary: any severity where NS-TQA(or aug) significantly > STL? ----
    wins = []
    for dname, D in results.items():
        res = D["res"]
        for key in res:
            regime, family, lab = key
            if family == "clean":
                continue
            stl = res[key]["STL-only"][0] if res[key]["STL-only"] else float("nan")
            for arm in ("NS-TQA", "NS-TQA-aug"):
                ns = res[key][arm]
                if len(ns) >= 2 and stl == stl:
                    w = wilcoxon(ns, [stl] * len(ns))
                    d = mean_std(ns)[0] - stl
                    if d > 0 and w["p"] < 0.05:
                        wins.append(f"{dname}/{regime}/{family}/{lab}: {arm} +{d:.3f} (p={w['p']:.3f})")
    blocks.append("## Verdict — severities where learned > STL-only (p<0.05)")
    blocks.append("")
    blocks.append("\n".join(f"- {w}" for w in wins) if wins else
                  "_None: STL-only ≥ learned at every tested severity (learned perception "
                  "does not win under degradation at these settings)._")

    table = "\n".join(blocks)
    (out_dir / "degradation.md").write_text(table)
    (out_dir / "degradation.json").write_text(json.dumps({
        "seeds": seeds, "methods": methods, "sweep": sweep,
        "results": {d: {"C": D["C"], "T": D["T"], "clean_pools": D["clean_pools"],
                        "res": {"|".join(map(str, k)): {m: v for m, v in cell.items()}
                                for k, cell in D["res"].items()}}
                    for d, D in results.items()},
    }, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote -> {out_dir}/degradation.{{md,json}}")


if __name__ == "__main__":
    main()
