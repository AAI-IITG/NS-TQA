"""02 - Build semi-synthetic QA over real C-MAPSS signals.

Loads C-MAPSS, calibrates perception on training windows, then for each window
samples templated questions, executes phi* on the grounded state to compute
answers, and caches (X, question, phi*, answer*, critical_t) instances.

Run:  python scripts/02_build_cmapss_qa.py
(requires C-MAPSS files under configs/data_cmapss.yaml: root)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pickle
import time

import yaml

from benchmark.cmapss import load_cmapss, window_engine_ids, windows
from benchmark.templates import generate_questions
from executor.hard_logic import evaluate
from perception.grounding import Calibrator, ground


def build_instance(x, rul_value, question, cal, engine_id=None):
    """Build one compact QA instance from a signal window."""
    mu, pidx = ground(x, cal)
    rho, tr = evaluate(mu, question.program, pidx, anchor=0)
    return {
        "X": x.clone(),
        "rul": float(rul_value),
        "engine_id": engine_id,
        "question": question,
        "phi_star": question.program,
        "answer_star": bool(rho > 0),
        "rho_star": float(rho),
        "critical_t": tr.critical_t,
    }


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "data_cmapss.yaml"))
    pcfg = yaml.safe_load(open(ROOT / "configs" / "perception.yaml"))

    root = ROOT / cfg["root"]
    if not (root / f"train_{cfg['subset']}.txt").exists():
        raise SystemExit(
            f"C-MAPSS files not found under {root}. Place train_{cfg['subset']}.txt there."
        )

    print(f"loading C-MAPSS {cfg['subset']} ...")
    ds = load_cmapss(root, subset=cfg["subset"], rul_cap=cfg["rul_cap"],
                     flat_std_thresh=cfg["flat_std_thresh"])
    print(f"  engines={len(ds.trajectories)} channels={ds.n_channels} "
          f"dropped={ds.dropped_channels}")

    X, rul = windows(ds, seq_len=cfg["seq_len"], stride=cfg["stride"])
    engine_ids = window_engine_ids(ds, seq_len=cfg["seq_len"], stride=cfg["stride"])
    print(f"  windows: {tuple(X.shape)}")

    cal = Calibrator.fit(X, hi_q=pcfg["hi_q"], lo_q=pcfg["lo_q"], smooth_k=pcfg["smooth_k"])
    cal.a_level = pcfg["a_level"]

    C = ds.n_channels
    T = cfg["seq_len"]
    questions = generate_questions(len(X), n_channels=C, T=T, seed=0)

    instances = []
    start = time.time()
    yes = 0
    for i in range(len(X)):
        instance = build_instance(X[i], rul[i], questions[i], cal, engine_ids[i])
        instances.append(instance)
        yes += int(instance["answer_star"])
        n_done = i + 1
        if n_done % 1000 == 0 or n_done == len(X):
            elapsed = time.time() - start
            print(
                f"  built {n_done}/{len(X)} QA instances "
                f"| yes_frac={yes/n_done:.3f} | elapsed={elapsed:.1f}s"
            )

    print(f"  built {len(instances)} QA instances | yes_frac={yes/len(instances):.3f}")

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / f"cmapss_{cfg['subset']}_qa.pkl"
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump({"instances": instances, "calibrator": cal,
                     "channel_names": ds.channel_names,
                     "engine_ids": ds.engine_ids}, f)
    tmp_path.replace(final_path)
    print(f"  saved to {final_path}")


if __name__ == "__main__":
    main()
