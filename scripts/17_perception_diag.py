"""17 - Perception-quality diagnostic (why is XJTU macro-F1 low?).

Trains the NS-TQA perception net on a benchmark and reports per-family predicate
F1 (high/low/rising/falling) on TRAIN vs in-dist vs shift. The three-way split
separates the candidate causes of a low macro-F1:

  * low TRAIN F1            -> underfitting / capacity / epochs (raise these);
  * train high, indist low  -> the (single-bearing) held-out split is the bottleneck;
  * train high, shift low   -> genuine operating-condition shift in that family;
  * one family low across all-> that predicate (e.g. rising/falling on noisy HIs)
                               needs a better feature/smoothing, not more training.

Run:  python scripts/17_perception_diag.py [--config configs/xjtu_necessity.yaml]
                                           [--epochs N] [--hidden H]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from benchmark.realdata import build_real_benchmark
from perception.grounding import PRED_FAMILIES
from perception.learned import predicate_metrics, train_perception

# reuse the adapter dispatch from the faithfulness runner
sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module
build_adapter = import_module("15_run_faithfulness").build_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "xjtu_necessity.yaml"))
    ap.add_argument("--epochs", type=int, default=None, help="override perception epochs")
    ap.add_argument("--hidden", type=int, default=None, help="override perception hidden")
    ap.add_argument("--indist-frac", type=float, default=None,
                    help="override indist_holdout_frac (more held-out bearings)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    a, b, p = cfg["adapter"], cfg["build"], cfg["perception"]
    if args.indist_frac is not None:
        b["indist_holdout_frac"] = args.indist_frac
    epochs = args.epochs if args.epochs is not None else p["epochs"]
    hidden = args.hidden if args.hidden is not None else p["hidden"]

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
    print(f"{adapter.name}: channels={C}  epochs={epochs} hidden={hidden}  "
          f"train={bm['meta']['n_train']} indist={bm['meta']['n_indist']} shift={bm['meta']['n_shift']}")

    pres = train_perception(
        bm["train"], n_channels=C, hidden=hidden, kernel=p["kernel"],
        n_layers=p["n_layers"], per_channel=p["per_channel"], epochs=epochs,
        batch_size=p["batch_size"], lr=p["lr"], weight_decay=p["weight_decay"],
        device_pref=cfg.get("device", "cpu"), seed=args.seed, verbose=False,
    )

    splits = [("train", bm["train"]), ("indist", bm["test_indist"])]
    if bm["meta"]["n_shift"] > 0:
        splits.append(("shift", bm["test_shift"]))
    reports = {name: predicate_metrics(pres.model, insts) for name, insts in splits}

    hdr = "| split | " + " | ".join(PRED_FAMILIES) + " | macro | elem-acc |"
    print("\n" + hdr)
    print("|---" * (len(PRED_FAMILIES) + 3) + "|")
    for name, _ in splits:
        r = reports[name]
        fam = " | ".join(f"{r[f]['f1']:.3f}" for f in PRED_FAMILIES)
        print(f"| {name} | {fam} | {r['macro_f1']:.3f} | {r['elementwise_acc']:.3f} |")

    # one-line read of the likely cause
    tr, ind = reports["train"]["macro_f1"], reports["indist"]["macro_f1"]
    worst = min(PRED_FAMILIES, key=lambda f: reports["indist"][f]["f1"])
    print(f"\ntrain macro-F1={tr:.3f}  indist macro-F1={ind:.3f}  "
          f"(gap={tr - ind:+.3f}); weakest family on indist = {worst} "
          f"({reports['indist'][worst]['f1']:.3f})")


if __name__ == "__main__":
    main()
