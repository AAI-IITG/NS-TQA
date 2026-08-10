"""09 - Generate the planted-spurious necessity benchmark.

Builds three populations (train, test_indist, test_shift) where the answer is
determined ONLY by a depth-d STL program over CAUSAL channels, and a separate
SPURIOUS channel offers a depth-0 shortcut that is correlated with the answer in
train/indist but broken under shift. Caches the benchmark to disk after passing
class-balance and shortcut guards.

Run:
  python scripts/09_generate_spurious.py
  python scripts/09_generate_spurious.py --shift-mode shift_flip
  python scripts/09_generate_spurious.py --config configs/spurious.yaml --seed 1
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
import pickle
import time

import yaml

from benchmark.spurious import balance_report, generate_spurious_benchmark


def _report(name: str, instances: list) -> dict:
    rep = balance_report(instances)
    print(
        f"  {name:12s} n={rep['n']:5d}  yes_frac={rep['yes_frac']:.3f}  "
        f"spurious_shortcut_acc={rep['spurious_shortcut_acc']:.3f}"
    )
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "spurious.yaml"))
    ap.add_argument("--shift-mode", default=None,
                    help="override config shift_mode (shift_decorr | shift_flip)")
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    ap.add_argument("--out", default=None, help="override output pickle path")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.shift_mode is not None:
        cfg["shift_mode"] = args.shift_mode
    if args.seed is not None:
        cfg["seed"] = args.seed

    print(f"generating planted-spurious benchmark (shift_mode={cfg['shift_mode']}, "
          f"seed={cfg['seed']}) ...")
    t0 = time.time()
    bm = generate_spurious_benchmark(
        n_train=cfg["n_train"], n_test=cfg["n_test"], n_causal=cfg["n_causal"],
        T=cfg["T"], depths=cfg["depths"], spurious_gap=cfg["spurious_gap"],
        causal_noise=cfg["causal_noise"], spurious_noise=cfg["spurious_noise"],
        shift_mode=cfg["shift_mode"], allow_until=cfg["allow_until"], seed=cfg["seed"],
    )
    print(f"  generated in {time.time() - t0:.1f}s | meta: {bm['meta']}")

    reps = {
        "train": _report("train", bm["train"]),
        "test_indist": _report("test_indist", bm["test_indist"]),
        "test_shift": _report("test_shift", bm["test_shift"]),
    }

    # ---- guards: refuse to save a benchmark that cannot test the hypothesis ----
    lo, hi = cfg["min_yes_frac"], cfg["max_yes_frac"]
    for name, rep in reps.items():
        if not (lo <= rep["yes_frac"] <= hi):
            raise SystemExit(
                f"DEGENERATE split {name!r}: yes_frac={rep['yes_frac']:.3f} "
                f"outside [{lo},{hi}]. Adjust generation before training."
            )
    if reps["train"]["spurious_shortcut_acc"] < cfg["min_train_shortcut"]:
        raise SystemExit(
            f"spurious shortcut not learnable in train "
            f"(acc={reps['train']['spurious_shortcut_acc']:.3f} < "
            f"{cfg['min_train_shortcut']}); the necessity contrast needs the "
            f"shortcut to exist in train."
        )
    if reps["test_shift"]["spurious_shortcut_acc"] > cfg["max_shift_shortcut"]:
        raise SystemExit(
            f"spurious shortcut still usable under shift "
            f"(acc={reps['test_shift']['spurious_shortcut_acc']:.3f} > "
            f"{cfg['max_shift_shortcut']}); shift did not break the shortcut."
        )

    # ---- atomic save ----
    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = Path(args.out) if args.out else out_dir / f"spurious_{cfg['shift_mode']}.pkl"
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    payload = {"train": bm["train"], "test_indist": bm["test_indist"],
               "test_shift": bm["test_shift"], "meta": bm["meta"],
               "reports": reps}
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f)
    tmp_path.replace(final_path)

    report_path = final_path.with_suffix(".report.json")
    with open(report_path, "w") as f:
        json.dump({"meta": bm["meta"], "reports": reps}, f, indent=2)

    print(f"  guards passed.")
    print(f"  saved benchmark to {final_path}")
    print(f"  saved report to    {report_path}")


if __name__ == "__main__":
    main()