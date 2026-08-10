"""18 - Answer-only-supervised NS-TQA ablation (pre-empts the "unfair mu* supervision" critique).

The headline NS-TQA trains perception with dense per-predicate supervision against
the privileged ``mu_star``; the end-to-end baselines only ever see answer labels.
A reviewer will ask whether NS-TQA's edge is the architecture or that extra
supervision. This ablation removes the advantage: it trains a SECOND perception
net from ANSWER LABELS ONLY, backpropagating through the differentiable soft
executor (``train_perception_answer_only``), then evaluates it with the HARD
executor like every other NS-TQA. If answer-only NS-TQA still beats the
answer-only baselines, the neuro-symbolic structure is doing the work.

Reports both NS-TQA variants (answer accuracy + perception macro-F1) on in-dist
and shift, vs the oracle upper bound; baseline numbers come from script 14.

Run:  python scripts/18_answer_only_ablation.py [--config configs/xjtu_necessity.yaml]
                                               [--seeds N] [--quick]
"""
import argparse
import json
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

from benchmark.necessity import oracle_accuracy
from benchmark.realdata import build_real_benchmark
from models.nstqa_learned import LearnedNSTQA
from perception.grounding import predicate_index
from perception.learned import (predicate_metrics, train_perception,
                                train_perception_answer_only)

build_adapter = import_module("15_run_faithfulness").build_adapter


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return (float("nan"), 0.0)
    return (sum(xs) / len(xs), stats.pstdev(xs) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "xjtu_necessity.yaml"))
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    a, b, p, exp = cfg["adapter"], cfg["build"], cfg["perception"], cfg["experiment"]
    seeds = exp["seeds"]
    if args.seeds is not None:
        seeds = seeds[:args.seeds]
    epochs = p["epochs"]
    if args.quick:
        seeds, epochs = seeds[:1], 15

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
    print(f"{adapter.name}: channels={C} epochs={epochs} seeds={seeds} "
          f"train={bm['meta']['n_train']} indist={bm['meta']['n_indist']} shift={bm['meta']['n_shift']}")

    common = dict(n_channels=C, hidden=p["hidden"], kernel=p["kernel"],
                  n_layers=p["n_layers"], per_channel=p["per_channel"], epochs=epochs,
                  batch_size=p["batch_size"], lr=p["lr"], weight_decay=p["weight_decay"],
                  device_pref=cfg.get("device", "cpu"), verbose=False)

    variants = {"NS-TQA (mu* supervised)": train_perception,
                "NS-TQA (answer-only)": train_perception_answer_only}
    acc = {v: {reg: [] for _, reg in regimes} for v in variants}
    f1 = {v: {reg: [] for _, reg in regimes} for v in variants}

    for seed in seeds:
        for vname, trainer in variants.items():
            print(f"seed {seed}: training {vname} ...", flush=True)
            pres = trainer(bm["train"], seed=seed, **common)
            nst = LearnedNSTQA(pres.model, n_channels=C)
            for key, reg in regimes:
                acc[vname][reg].append(nst.evaluate_instances(bm[key])["answer_accuracy"])
                f1[vname][reg].append(predicate_metrics(pres.model, bm[key])["macro_f1"])

    oracle = {reg: oracle_accuracy(bm[key], pidx)["answer_accuracy"] for key, reg in regimes}

    out_dir = ROOT / cfg["run_root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = " | ".join(f"{reg} acc" for _, reg in regimes)
    f1cols = " | ".join(f"{reg} F1" for _, reg in regimes)
    L = [f"# Answer-only-supervised NS-TQA ablation ({len(seeds)} seeds): "
         f"{bm['meta']['dataset']} shift={b['shift']}", "",
         f"| Method | {cols} | {f1cols} |",
         "|---" * (1 + 2 * len(regimes)) + "|"]
    for v in variants:
        a_cells = " | ".join(f"{mean_std(acc[v][reg])[0]:.3f} ± {mean_std(acc[v][reg])[1]:.3f}"
                             for _, reg in regimes)
        f_cells = " | ".join(f"{mean_std(f1[v][reg])[0]:.3f}" for _, reg in regimes)
        L.append(f"| {v} | {a_cells} | {f_cells} |")
    orc = " | ".join(f"{oracle[reg]:.3f}" for _, reg in regimes)
    L.append(f"| oracle (upper bound) | {orc} | " + " | ".join("—" for _ in regimes) + " |")
    L += ["", "*Baseline reference (answer-only, script 14): best baseline TCN ≈ 0.66/0.68 on "
          "XJTU cross-load. Answer-only NS-TQA uses the SAME supervision as the baselines.*"]
    table = "\n".join(L)
    (out_dir / "answer_only_ablation.md").write_text(table)
    (out_dir / "answer_only_ablation.json").write_text(json.dumps({
        "acc": {v: {reg: list(mean_std(acc[v][reg])) for _, reg in regimes} for v in variants},
        "perception_f1": {v: {reg: list(mean_std(f1[v][reg])) for _, reg in regimes} for v in variants},
        "oracle": oracle, "seeds": seeds, "dataset": bm["meta"]["dataset"], "shift": b["shift"],
    }, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote table + json to {out_dir}")


if __name__ == "__main__":
    main()
