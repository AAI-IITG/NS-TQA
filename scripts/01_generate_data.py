"""01 - Generate the Level-1 synthetic benchmark.

Creates planted-rule instances (X, mu*, question, phi*, answer*) with the
learnability guard enabled, validates non-degeneracy, and caches to disk.

Run:  python scripts/01_generate_data.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pickle

from benchmark.synthetic import generate_synthetic, class_balance


def main():
    out_dir = ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 2000
    n_channels = 4
    T = 48
    seed = 0

    print(f"generating {n} synthetic instances (C={n_channels}, T={T}) ...")
    instances, cal = generate_synthetic(
        n=n, n_channels=n_channels, T=T, seed=seed, enforce_learnable=True
    )

    bal = class_balance(instances)
    print(f"  produced {bal['n']} instances | yes={bal['yes']} no={bal['no']} "
          f"(yes_frac={bal['yes_frac']:.3f})")

    # non-degeneracy guard -- refuse to save a benchmark that is essentially one class
    if not (0.10 < bal["yes_frac"] < 0.90):
        raise SystemExit(
            f"DEGENERATE benchmark (yes_frac={bal['yes_frac']:.3f}); "
            "adjust templates/excitation before training."
        )

    # split
    n_tr = int(0.8 * len(instances))
    train, test = instances[:n_tr], instances[n_tr:]

    with open(out_dir / "synthetic_train.pkl", "wb") as f:
        pickle.dump(train, f)
    with open(out_dir / "synthetic_test.pkl", "wb") as f:
        pickle.dump(test, f)
    with open(out_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(cal, f)

    print(f"  saved {len(train)} train / {len(test)} test to {out_dir}")
    print("  guard passed: benchmark is non-degenerate and learnable.")


if __name__ == "__main__":
    main()
