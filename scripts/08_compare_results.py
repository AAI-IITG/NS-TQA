"""08 - Compare direct-answer baselines against the symbolic oracle.

Run:
  python scripts/08_compare_results.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _dataset_from_metrics(metrics: dict, fallback: str) -> str:
    tag = metrics.get("dataset_tag")
    if tag:
        return tag
    dataset_path = metrics.get("dataset_path")
    if dataset_path:
        name = Path(dataset_path).stem
        return name[:-3] if name.endswith("_qa") else name
    return fallback


def _baseline_row(path: Path, metrics: dict) -> dict:
    split_meta = metrics.get("split_meta") or {"split_mode": "random"}
    split_mode = split_meta.get("split_mode", "random")
    group_key = split_meta.get("group_key")
    method = "LSTM direct-answer"
    setting = split_mode if group_key is None else f"{split_mode}:{group_key}"
    binary = metrics.get("binary", {})
    return {
        "dataset": _dataset_from_metrics(metrics, path.parent.name),
        "method": method,
        "setting": setting,
        "split": metrics.get("split", "test"),
        "n": metrics.get("n"),
        "answer_accuracy": metrics.get("answer_accuracy"),
        "balanced_accuracy": binary.get("balanced_accuracy"),
        "f1": binary.get("f1"),
        "recall": binary.get("recall"),
        "conjunction_score": metrics.get("conjunction_score"),
        "metrics_path": str(path.relative_to(ROOT)),
    }


def _oracle_row(path: Path, metrics: dict) -> dict:
    binary = metrics.get("binary", {})
    return {
        "dataset": _dataset_from_metrics(metrics, path.parent.name),
        "method": "NS-TQA symbolic oracle",
        "setting": "all:phi_star",
        "split": metrics.get("split", "all"),
        "n": metrics.get("n"),
        "answer_accuracy": metrics.get("answer_accuracy"),
        "balanced_accuracy": binary.get("balanced_accuracy"),
        "f1": binary.get("f1"),
        "recall": binary.get("recall"),
        "conjunction_score": metrics.get("conjunction_score"),
        "metrics_path": str(path.relative_to(ROOT)),
    }


def _markdown(rows: list[dict]) -> str:
    headers = [
        "dataset",
        "method",
        "setting",
        "split",
        "n",
        "answer_accuracy",
        "balanced_accuracy",
        "f1",
        "recall",
        "conjunction_score",
    ]
    lines = [
        "# Result Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(h)) for h in headers) + " |")
    lines.extend(
        [
            "",
            "Notes:",
            "- LSTM direct-answer rows are ablation baselines; they predict `answer_star` directly.",
            "- NS-TQA symbolic oracle executes the stored `phi_star` with the saved calibrator.",
            "- `conjunction_score` is only meaningful for the faithful symbolic path here.",
        ]
    )
    return "\n".join(lines) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", default=str(ROOT / "runs" / "baseline"))
    parser.add_argument(
        "--oracle-root", default=str(ROOT / "runs" / "symbolic_oracle")
    )
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "summary"))
    args = parser.parse_args()

    rows = []
    baseline_root = Path(args.baseline_root)
    oracle_root = Path(args.oracle_root)
    for path in sorted(baseline_root.glob("*/metrics_test.json")):
        rows.append(_baseline_row(path, _load_json(path)))
    for path in sorted(oracle_root.glob("*/metrics_all.json")):
        rows.append(_oracle_row(path, _load_json(path)))

    rows.sort(key=lambda r: (r["dataset"], r["method"], r["setting"]))
    summary = {
        "n_rows": len(rows),
        "rows": rows,
    }

    out_dir = Path(args.out_dir)
    json_path = out_dir / "results.json"
    md_path = out_dir / "results.md"
    _atomic_write_json(json_path, summary)
    _atomic_write_text(md_path, _markdown(rows))

    print(_markdown(rows))
    print(f"saved summary to {md_path} and {json_path}")


if __name__ == "__main__":
    main()
