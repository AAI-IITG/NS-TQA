"""07 - Evaluate the faithful symbolic NS-TQA oracle path.

Run:
  python scripts/07_evaluate_symbolic_oracle.py --dataset-path data/processed/cmapss_FD001_qa.pkl
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
import time

import torch

from benchmark.baseline import load_qa_pickle
from models.nstqa import NSTQA
from utils.oracle_metrics import assemble_oracle_metrics


def dataset_tag(path: Path) -> str:
    """Convert an artifact path to a stable run tag."""
    name = path.stem
    return name[:-3] if name.endswith("_qa") else name


def _as_tensor(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default=str(ROOT / "data" / "processed" / "cmapss_FD001_qa.pkl"),
    )
    parser.add_argument(
        "--run-root",
        default=str(ROOT / "runs" / "symbolic_oracle"),
    )
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    artifact = load_qa_pickle(dataset_path)
    if "calibrator" not in artifact:
        raise SystemExit(f"artifact has no saved calibrator: {dataset_path}")

    instances = artifact["instances"]
    if args.max_instances is not None:
        instances = instances[: args.max_instances]

    tag = dataset_tag(dataset_path)
    oracle = NSTQA(artifact["calibrator"])
    records = []
    start = time.time()

    for i, instance in enumerate(instances):
        X = _as_tensor(instance["X"])
        phi = instance["phi_star"]
        pred = oracle.answer(X, phi)
        question = instance.get("question")
        records.append(
            {
                "pred_answer": bool(pred["answer"]),
                "gold_answer": bool(instance["answer_star"]),
                "pred_rho": float(pred["rho"]),
                "gold_rho": float(instance["rho_star"]),
                "pred_critical_t": pred["critical_t"],
                "gold_critical_t": instance["critical_t"],
                "template": getattr(question, "template", None),
                "engine_id": instance.get("engine_id"),
                "bearing_id": instance.get("bearing_id"),
            }
        )
        n_done = i + 1
        if n_done % 1000 == 0 or n_done == len(instances):
            elapsed = time.time() - start
            yes = sum(r["pred_answer"] for r in records) / max(1, len(records))
            print(
                f"  evaluated {n_done}/{len(instances)} "
                f"| pred_yes_frac={yes:.3f} | elapsed={elapsed:.1f}s",
                flush=True,
            )

    metrics = assemble_oracle_metrics(
        records,
        dataset_path=str(dataset_path.relative_to(ROOT)),
        dataset_tag=tag,
    )
    metrics["run_type"] = "symbolic_oracle"

    out_path = Path(args.run_root) / tag / "metrics_all.json"
    _atomic_write_json(out_path, metrics)

    print(json.dumps(metrics, indent=2))
    print(f"saved metrics to {out_path}")


if __name__ == "__main__":
    main()
