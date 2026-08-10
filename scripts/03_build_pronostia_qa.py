"""03 - Build semi-synthetic QA over real PRONOSTIA bearing feature signals.

Loads complete PRONOSTIA run-to-failure bearing folders, converts raw vibration
snapshots into feature trajectories, calibrates perception on feature windows,
then executes each structured STL question over the grounded symbolic state.

Run:  python scripts/03_build_pronostia_qa.py
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

from benchmark.cmapss import windows
from benchmark.pronostia import load_pronostia
from benchmark.templates import generate_questions
from executor.hard_logic import evaluate
from perception.grounding import Calibrator, ground


def build_instance(x, rul_value, question, cal, bearing_id):
    """Build one compact PRONOSTIA QA instance from a feature window."""
    mu, pidx = ground(x, cal)
    rho, tr = evaluate(mu, question.program, pidx, anchor=0)
    return {
        "X": x.clone(),
        "rul": float(rul_value),
        "bearing_id": bearing_id,
        "question": question,
        "phi_star": question.program,
        "answer_star": bool(rho > 0),
        "rho_star": float(rho),
        "critical_t": tr.critical_t,
    }


def _window_bearing_ids(ds, seq_len: int, stride: int) -> list[str]:
    ids = []
    for bearing_id, traj in zip(ds.bearing_ids, ds.trajectories):
        T = traj.shape[0]
        if T < seq_len:
            continue
        n = len(range(0, T - seq_len + 1, stride))
        ids.extend([bearing_id] * n)
    return ids


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "data_pronostia.yaml"))
    pcfg = yaml.safe_load(open(ROOT / "configs" / "perception.yaml"))

    root = ROOT / cfg["root"]
    if not root.exists():
        raise SystemExit(f"PRONOSTIA dataset root not found: {root}")

    log = lambda msg: print(msg, flush=True)

    log("loading PRONOSTIA ...")
    log(f"  root={root}")
    log(f"  splits={cfg['splits']} axis={cfg['axis']} n_bands={cfg['n_bands']}")
    ds = load_pronostia(
        root,
        splits=cfg["splits"],
        axis=cfg["axis"],
        n_bands=cfg["n_bands"],
        rul_cap=cfg["rul_cap"],
        flat_std_thresh=cfg["flat_std_thresh"],
        progress=lambda msg: log(f"  {msg}"),
    )
    log(
        f"  bearings={len(ds.trajectories)} channels={ds.n_channels} "
        f"dropped={ds.dropped_channels}"
    )
    for bearing_id, traj in zip(ds.bearing_ids, ds.trajectories):
        log(f"    {bearing_id}: snapshots={traj.shape[0]}")

    X, rul = windows(ds, seq_len=cfg["seq_len"], stride=cfg["stride"])
    bearing_ids = _window_bearing_ids(ds, seq_len=cfg["seq_len"], stride=cfg["stride"])
    log(f"  windows: {tuple(X.shape)}")

    cal = Calibrator.fit(X, hi_q=pcfg["hi_q"], lo_q=pcfg["lo_q"], smooth_k=pcfg["smooth_k"])
    cal.a_level = pcfg["a_level"]

    C = ds.n_channels
    T = cfg["seq_len"]
    questions = generate_questions(len(X), n_channels=C, T=T, seed=0)

    instances = []
    start = time.time()
    yes = 0
    for i in range(len(X)):
        instance = build_instance(X[i], rul[i], questions[i], cal, bearing_ids[i])
        instances.append(instance)
        yes += int(instance["answer_star"])
        n_done = i + 1
        if n_done % 1000 == 0 or n_done == len(X):
            elapsed = time.time() - start
            log(
                f"  built {n_done}/{len(X)} QA instances "
                f"| yes_frac={yes/n_done:.3f} | elapsed={elapsed:.1f}s"
            )

    log(f"  built {len(instances)} QA instances | yes_frac={yes/len(instances):.3f}")

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    split_tag = "-".join(s.lower() for s in cfg["splits"])
    final_path = out / f"pronostia_{split_tag}_qa.pkl"
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(
            {
                "instances": instances,
                "calibrator": cal,
                "channel_names": ds.channel_names,
                "bearing_ids": ds.bearing_ids,
                "splits": ds.splits,
                "rul_units": ds.rul_units,
            },
            f,
        )
    tmp_path.replace(final_path)
    log(f"  saved to {final_path}")


if __name__ == "__main__":
    main()
