"""21 - Answer-level anomaly-question benchmark (Phase G payoff).

Builds a non-circular QA benchmark whose programs use the learned `anomalous`
predicate (5-family vocabulary), then compares, at the ANSWER level:

  * NS-TQA  : learned 4-family perception + learned anomaly head -> executor
  * STL-only: fixed hand-rule grounding + a level proxy for `anomalous` -> executor
  * oracle  : executor on the privileged 5-family mu*

Expected: NS-TQA >> STL-only on anomaly-using programs, because the hand-rule has no
faithful `anomalous` rule -- the setting where learned perception is necessary on
real signals. Also reports a depth-wise leakage probe.

Run:  python scripts/21_run_anomaly_qa.py [--config configs/xjtu_necessity.yaml] [--seeds N] [--quick]
"""
import argparse
import dataclasses
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

from benchmark.anomaly_qa import (anomaly_leakage_probe, build_anomaly_benchmark,
                                  nstqa_anomaly_eval, oracle_anomaly_eval,
                                  stl_only_anomaly_eval)
from perception.anomaly import train_anomaly_head
from perception.learned import train_perception

build_adapter = import_module("15_run_faithfulness").build_adapter


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return (float("nan"), 0.0)
    return (sum(xs) / len(xs), stats.pstdev(xs) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "xjtu_necessity.yaml"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--anomaly-p", type=float, default=1.0,
                    help="fraction of program leaves that are `anomalous` (1.0 = anomaly-focused)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    a, b, p = cfg["adapter"], cfg["build"], cfg["perception"]
    seeds = list(range(args.seeds))
    pe = p["epochs"]
    if args.quick:
        seeds, pe = [0], 12

    adapter = build_adapter(a, b)
    print(f"building anomaly benchmark ({adapter.name}, shift={b['shift']}) ...", flush=True)
    bm = build_anomaly_benchmark(
        adapter, T=b["T"], stride=b["stride"], depths=tuple(b["depths"]),
        shift=b["shift"], train_conditions=tuple(b["train_conditions"]),
        test_conditions=tuple(b["test_conditions"]),
        indist_holdout_frac=b["indist_holdout_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=a["hi_q"], lo_q=a["lo_q"], smooth_k=b["smooth_k"], a_level=b["a_level"],
        anomaly_p=args.anomaly_p,
        allow_until=b["allow_until"], max_windows_per_unit=b.get("max_windows_per_unit"),
        over_factor=b["over_factor"], seed=b["build_seed"],
    )
    C = bm["meta"]["n_channels"]
    baseline = bm["baseline"]
    regimes = [("test_indist", "indist")] + ([("test_shift", "shift")] if bm["meta"]["n_shift"] else [])
    print(f"  channels={C} train={bm['meta']['n_train']} "
          + " ".join(f"{r}={len(bm[k])}" for k, r in regimes))

    # STL-only and oracle are seed-independent (no training)
    stl = {r: stl_only_anomaly_eval(bm[k], smooth_k=b["smooth_k"], a_level=b["a_level"])
           for k, r in regimes}
    orc = {r: oracle_anomaly_eval(bm[k]) for k, r in regimes}
    leak = {r: anomaly_leakage_probe(bm[k]) for k, r in regimes}

    # NS-TQA: train 4-family perception (on the 4C slice of mu*) + anomaly head, per seed
    train4 = [dataclasses.replace(i, mu_star=i.mu_star[:, :4 * C]) for i in bm["train"]]
    Xtrain = __import__("torch").stack([i.X for i in bm["train"]])
    ns = {r: [] for _, r in regimes}
    for seed in seeds:
        print(f"  seed {seed}: training perception + anomaly head ...", flush=True)
        pres = train_perception(
            train4, n_channels=C, hidden=p["hidden"], kernel=p["kernel"],
            n_layers=p["n_layers"], per_channel=p["per_channel"], epochs=pe,
            batch_size=p["batch_size"], lr=p["lr"], weight_decay=p["weight_decay"],
            device_pref=cfg.get("device", "cpu"), seed=seed, verbose=False)
        head = train_anomaly_head(Xtrain, baseline, hidden=p["hidden"], kernel=p["kernel"],
                                  n_layers=p["n_layers"], epochs=pe, lr=p["lr"],
                                  device_pref=cfg.get("device", "cpu"), seed=seed)
        for k, r in regimes:
            ns[r].append(nstqa_anomaly_eval(bm[k], pres.model, head)["answer_accuracy"])

    out_dir = ROOT / cfg["run_root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    L = [f"# Anomaly-question benchmark ({len(seeds)} seeds): {adapter.name} shift={b['shift']}",
         "", "| Method | " + " | ".join(r for _, r in regimes) + " |",
         "|---" * (len(regimes) + 1) + "|"]
    L.append("| **NS-TQA (learned perc.+anomaly head)** | "
             + " | ".join(f"{mean_std(ns[r])[0]:.3f} ± {mean_std(ns[r])[1]:.3f}" for _, r in regimes) + " |")
    L.append("| STL-only (hand rules + level proxy) | "
             + " | ".join(f"{stl[r]['answer_accuracy']:.3f}" for _, r in regimes) + " |")
    L.append("| oracle (privileged mu*) | "
             + " | ".join(f"{orc[r]['answer_accuracy']:.3f}" for _, r in regimes) + " |")
    L += ["", "### Leakage probe (best single privileged predicate; want ~0.5 for depth>=2)", "",
          "| regime | " + " | ".join(f"d{d}" for d in b["depths"]) + " |",
          "|---" * (len(b["depths"]) + 1) + "|"]
    for _, r in regimes:
        L.append(f"| {r} | " + " | ".join(f"{leak[r].get(d, float('nan')):.2f}" for d in b["depths"]) + " |")
    table = "\n".join(L)
    (out_dir / "anomaly_qa.md").write_text(table)
    (out_dir / "anomaly_qa.json").write_text(json.dumps({
        "nstqa": {r: list(mean_std(ns[r])) for _, r in regimes},
        "stl_only": {r: stl[r]["answer_accuracy"] for _, r in regimes},
        "oracle": {r: orc[r]["answer_accuracy"] for _, r in regimes},
        "leakage": leak, "seeds": seeds, "dataset": adapter.name,
    }, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote to {out_dir}")


if __name__ == "__main__":
    main()
