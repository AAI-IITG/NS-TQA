"""10 - Necessity experiment: end-to-end LSTM vs faithful NS-TQA.

Trains, on the SAME planted-spurious benchmark (script 09):
  * an end-to-end LSTM that maps (signal, encoded program) -> answer, and is
    free to shortcut on the spurious channel; and
  * the faithful NS-TQA path: learned perception (X -> predicate truths) then
    the deterministic STL executor running phi*.

Then evaluates both on test_indist and test_shift, broken down by program depth,
and writes a results table plus two headline figures:
  * necessity bar chart  (indist vs shift, both methods, with shortcut + oracle)
  * depth curve          (accuracy vs program depth, both methods, both regimes)

Run:
  python scripts/10_run_necessity.py
  python scripts/10_run_necessity.py --benchmark data/synthetic/spurious_shift_flip.pkl
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from benchmark.baseline import select_device
from benchmark.spurious import balance_report
from executor.grammar import Always, And, Eventually, Node, Not, Or, Predicate, Until
from executor.hard_logic import evaluate
from models.baselines import LSTMBaseline
from models.nstqa_learned import LearnedNSTQA
from perception.grounding import PRED_FAMILIES, predicate_index
from perception.learned import predicate_metrics, train_perception
from utils.oracle_metrics import binary_metrics, group_accuracy

OPS = ["Eventually", "Always", "Until", "And", "Or", "Not"]


# --------------------------------------------------------------------------- #
# program encoding for the end-to-end baseline (fair: gets the program, not the
# executor). Works for arbitrary depth-d trees, unlike baseline.encode_question.
# --------------------------------------------------------------------------- #

def _op_depth(node: Node) -> int:
    if isinstance(node, Predicate):
        return 0
    if isinstance(node, (Not, Eventually, Always)):
        return 1 + _op_depth(node.child)
    if isinstance(node, (And, Or, Until)):
        return 1 + max(_op_depth(node.left), _op_depth(node.right))
    raise TypeError(type(node).__name__)


def encode_program(phi: Node, n_channels: int, T: int) -> torch.Tensor:
    op_counts = {o: 0 for o in OPS}
    pred_mask = [0.0] * (len(PRED_FAMILIES) * n_channels)  # family-major presence
    a_list, b_list = [], []

    def walk(n: Node) -> None:
        if isinstance(n, Predicate):
            fi = PRED_FAMILIES.index(n.name)
            pred_mask[fi * n_channels + n.channel] = 1.0
            return
        op_counts[type(n).__name__] += 1
        if isinstance(n, (Eventually, Always, Until)):
            a_list.append(n.a)
            b_list.append(n.b)
        for c in n._children():
            walk(c)

    walk(phi)
    denom = max(1, T - 1)
    mean_a = (sum(a_list) / len(a_list) / denom) if a_list else 0.0
    mean_b = (sum(b_list) / len(b_list) / denom) if b_list else 0.0
    feats = (
        [float(op_counts[o]) for o in OPS]
        + pred_mask
        + [float(_op_depth(phi)), float(sum(pred_mask)), mean_a, mean_b]
    )
    return torch.tensor(feats, dtype=torch.float32)


def program_query_dim(n_channels: int) -> int:
    return len(OPS) + len(PRED_FAMILIES) * n_channels + 4


# --------------------------------------------------------------------------- #
# LSTM baseline train / eval
# --------------------------------------------------------------------------- #

def _stack_lstm(instances: list, n_channels: int, T: int):
    X = torch.stack([i.X for i in instances])
    q = torch.stack([encode_program(i.phi_star, n_channels, T) for i in instances])
    y = torch.tensor([float(i.answer_star) for i in instances])
    depth = [i.depth for i in instances]
    return X, q, y, depth


def train_lstm(train_instances, n_channels, T, cfg, device, seed):
    torch.manual_seed(seed)
    model = LSTMBaseline(
        n_channels=n_channels, query_dim=program_query_dim(n_channels),
        hidden=cfg["hidden"], dropout=cfg["dropout"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    X, q, y, _ = _stack_lstm(train_instances, n_channels, T)
    X, q, y = X.to(device), q.to(device), y.to(device)
    n = X.shape[0]
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=device)
        total, correct = 0.0, 0
        for s in range(0, n, cfg["batch_size"]):
            idx = perm[s : s + cfg["batch_size"]]
            opt.zero_grad(set_to_none=True)
            logits = model(X[idx], q[idx])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[idx])
            loss.backward()
            if cfg.get("grad_clip_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            opt.step()
            total += float(loss.detach()) * len(idx)
            correct += int(((logits.sigmoid() >= 0.5) == y[idx].bool()).sum())
        if epoch % cfg.get("log_every", 10) == 0 or epoch == cfg["epochs"] - 1:
            print(f"  [lstm] epoch {epoch:3d} bce={total/n:.4f} train_acc={correct/n:.3f}")
    return model


@torch.no_grad()
def eval_lstm(model, instances, n_channels, T, device):
    model.eval()
    X, q, y, depth = _stack_lstm(instances, n_channels, T)
    logits = model(X.to(device), q.to(device))
    pred = [bool(p) for p in (logits.sigmoid() >= 0.5).cpu().tolist()]
    gold = [bool(g) for g in y.bool().tolist()]
    return {
        "answer_accuracy": sum(p == g for p, g in zip(pred, gold)) / max(1, len(gold)),
        "binary": binary_metrics(pred, gold),
        "by_depth": group_accuracy(depth, pred, gold),
    }


# --------------------------------------------------------------------------- #
# NS-TQA eval (faithful path)
# --------------------------------------------------------------------------- #

def eval_nstqa(nst: LearnedNSTQA, instances):
    out = nst.evaluate_instances(instances)
    recs = out["records"]
    pred = [r["pred_answer"] for r in recs]
    gold = [r["gold_answer"] for r in recs]
    depth = [r["depth"] for r in recs]
    return {
        "answer_accuracy": out["answer_accuracy"],
        "binary": binary_metrics(pred, gold),
        "by_depth": group_accuracy(depth, pred, gold),
    }


def oracle_accuracy(instances, pidx):
    """Executor on planted mu_star -> upper bound (should be 1.0)."""
    pred = [bool(evaluate(i.mu_star, i.phi_star, pidx, 0)[0] > 0) for i in instances]
    gold = [bool(i.answer_star) for i in instances]
    by_depth = group_accuracy([i.depth for i in instances], pred, gold)
    acc = sum(p == g for p, g in zip(pred, gold)) / max(1, len(gold))
    return acc, by_depth


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def _depth_row(by_depth: dict, depths: list[int]) -> list[str]:
    return [f"{by_depth.get(str(d), {}).get('accuracy', float('nan')):.3f}" for d in depths]


def write_results(run_dir: Path, meta: dict, results: dict, depths: list[int]):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "results.json", "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    head = ["method", "split", "n", "accuracy", "balanced_acc", "f1"] + [f"acc@d{d}" for d in depths]
    lines = ["# Necessity experiment results", "",
             f"benchmark: `{meta['benchmark']}` (shift_mode={meta['shift_mode']}), "
             f"n_causal={meta['n_causal']}, T={meta['T']}, depths={depths}", "",
             "| " + " | ".join(head) + " |",
             "| " + " | ".join(["---"] * len(head)) + " |"]

    def row(method, split, r):
        b = r["binary"]
        cells = [method, split, str(r["n"]), f"{r['answer_accuracy']:.3f}",
                 f"{b['balanced_accuracy']:.3f}", f"{b['f1']:.3f}"] + _depth_row(r["by_depth"], depths)
        lines.append("| " + " | ".join(cells) + " |")

    for method in ("LSTM end-to-end", "NS-TQA (learned perception + executor)"):
        for split in ("indist", "shift"):
            row(method, split, results[method][split])
    # reference rows
    for split in ("indist", "shift"):
        r = results["oracle (executor on mu_star)"][split]
        lines.append("| " + " | ".join(
            ["oracle (mu_star)", split, str(r["n"]), f"{r['answer_accuracy']:.3f}", "-", "-"]
            + _depth_row(r["by_depth"], depths)) + " |")
    for split in ("indist", "shift"):
        sc = results["spurious shortcut"][split]
        lines.append("| " + " | ".join(
            ["spurious shortcut", split, str(sc["n"]), f"{sc['accuracy']:.3f}", "-", "-"]
            + ["-"] * len(depths)) + " |")

    lines += ["", "Perception macro-F1: "
              f"indist={results['perception_f1']['indist']:.3f}, "
              f"shift={results['perception_f1']['shift']:.3f} "
              "(equal by design: identical causal channels across splits)."]
    (run_dir / "results.md").write_text("\n".join(lines) + "\n")


def plot_necessity_bar(run_dir: Path, results: dict, shift_mode: str):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    splits = ["indist", "shift"]
    x = range(len(splits))
    w = 0.35
    lstm = [results["LSTM end-to-end"][s]["answer_accuracy"] for s in splits]
    nst = [results["NS-TQA (learned perception + executor)"][s]["answer_accuracy"] for s in splits]
    ax.bar([i - w / 2 for i in x], lstm, w, label="LSTM end-to-end", color="#B0413E")
    ax.bar([i + w / 2 for i in x], nst, w, label="NS-TQA (ours)", color="#2C7A7B")
    short = [results["spurious shortcut"][s]["accuracy"] for s in splits]
    ax.plot(x, short, "o--", color="#888", label="spurious shortcut")
    ax.axhline(0.5, ls=":", color="black", lw=1, label="chance")
    ax.set_xticks(list(x)); ax.set_xticklabels(["in-distribution", f"shift ({shift_mode})"])
    ax.set_ylabel("answer accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Necessity: end-to-end collapses under shift; NS-TQA holds")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(run_dir / "fig_necessity_bar.png", dpi=300); plt.close(fig)


def plot_depth_curve(run_dir: Path, results: dict, depths: list[int]):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    def series(method, split):
        bd = results[method][split]["by_depth"]
        return [bd.get(str(d), {}).get("accuracy", float("nan")) for d in depths]
    ax.plot(depths, series("LSTM end-to-end", "indist"), "o-", color="#B0413E", label="LSTM indist")
    ax.plot(depths, series("LSTM end-to-end", "shift"), "o--", color="#B0413E", alpha=0.6, label="LSTM shift")
    ax.plot(depths, series("NS-TQA (learned perception + executor)", "indist"), "s-", color="#2C7A7B", label="NS-TQA indist")
    ax.plot(depths, series("NS-TQA (learned perception + executor)", "shift"), "s--", color="#2C7A7B", alpha=0.6, label="NS-TQA shift")
    ax.axhline(0.5, ls=":", color="black", lw=1)
    ax.set_xlabel("program depth"); ax.set_ylabel("answer accuracy")
    ax.set_xticks(depths); ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy vs compositional depth")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(run_dir / "fig_depth_curve.png", dpi=300); plt.close(fig)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "necessity.yaml"))
    ap.add_argument("--benchmark", default=None, help="override benchmark pickle path")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.benchmark is not None:
        cfg["benchmark"] = args.benchmark
    if args.seed is not None:
        cfg["seed"] = args.seed

    bench_path = ROOT / cfg["benchmark"]
    with open(bench_path, "rb") as f:
        bm = pickle.load(f)
    meta = bm["meta"]
    C, T, depths = meta["n_channels"], meta["T"], meta["depths"]
    device = select_device(cfg.get("device", "cpu"))
    print(f"loaded {bench_path.name}: train={len(bm['train'])} "
          f"test_indist={len(bm['test_indist'])} test_shift={len(bm['test_shift'])} "
          f"| C={C} T={T} depths={depths} device={device}")

    t0 = time.time()
    # --- end-to-end LSTM ---
    print("training LSTM end-to-end baseline ...")
    lstm = train_lstm(bm["train"], C, T, cfg["lstm"], device, cfg["seed"])
    lstm_in = eval_lstm(lstm, bm["test_indist"], C, T, device)
    lstm_sh = eval_lstm(lstm, bm["test_shift"], C, T, device)
    lstm_in["n"] = len(bm["test_indist"]); lstm_sh["n"] = len(bm["test_shift"])

    # --- faithful NS-TQA ---
    print("training NS-TQA learned perception ...")
    pres = train_perception(
        bm["train"], n_channels=C, hidden=cfg["perception"]["hidden"],
        kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
        per_channel=cfg["perception"]["per_channel"], epochs=cfg["perception"]["epochs"],
        batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
        weight_decay=cfg["perception"]["weight_decay"],
        device_pref=cfg.get("device", "cpu"), seed=cfg["seed"],
        log_every=cfg["perception"]["log_every"],
    )
    nst = LearnedNSTQA(pres.model, n_channels=C)
    nst_in = eval_nstqa(nst, bm["test_indist"])
    nst_sh = eval_nstqa(nst, bm["test_shift"])
    nst_in["n"] = len(bm["test_indist"]); nst_sh["n"] = len(bm["test_shift"])

    # --- references ---
    pidx = predicate_index(C)
    orc_in_acc, orc_in_bd = oracle_accuracy(bm["test_indist"], pidx)
    orc_sh_acc, orc_sh_bd = oracle_accuracy(bm["test_shift"], pidx)
    pf_in = predicate_metrics(pres.model, bm["test_indist"])["macro_f1"]
    pf_sh = predicate_metrics(pres.model, bm["test_shift"])["macro_f1"]

    results = {
        "LSTM end-to-end": {"indist": lstm_in, "shift": lstm_sh},
        "NS-TQA (learned perception + executor)": {"indist": nst_in, "shift": nst_sh},
        "oracle (executor on mu_star)": {
            "indist": {"n": len(bm["test_indist"]), "answer_accuracy": orc_in_acc, "by_depth": orc_in_bd},
            "shift": {"n": len(bm["test_shift"]), "answer_accuracy": orc_sh_acc, "by_depth": orc_sh_bd},
        },
        "spurious shortcut": {
            "indist": {"n": len(bm["test_indist"]),
                       "accuracy": balance_report(bm["test_indist"])["spurious_shortcut_acc"]},
            "shift": {"n": len(bm["test_shift"]),
                      "accuracy": balance_report(bm["test_shift"])["spurious_shortcut_acc"]},
        },
        "perception_f1": {"indist": pf_in, "shift": pf_sh},
    }

    run_dir = ROOT / cfg["run_root"] / meta["shift_mode"]
    meta_out = {"benchmark": str(bench_path.relative_to(ROOT)), **meta, "elapsed_sec": time.time() - t0}
    write_results(run_dir, meta_out, results, depths)
    plot_necessity_bar(run_dir, results, meta["shift_mode"])
    plot_depth_curve(run_dir, results, depths)

    print("\n=== headline ===")
    print(f"LSTM    indist={lstm_in['answer_accuracy']:.3f}  shift={lstm_sh['answer_accuracy']:.3f}  "
          f"(drop={lstm_in['answer_accuracy']-lstm_sh['answer_accuracy']:+.3f})")
    print(f"NS-TQA  indist={nst_in['answer_accuracy']:.3f}  shift={nst_sh['answer_accuracy']:.3f}  "
          f"(drop={nst_in['answer_accuracy']-nst_sh['answer_accuracy']:+.3f})")
    print(f"oracle  indist={orc_in_acc:.3f}  shift={orc_sh_acc:.3f}   perception_f1 indist={pf_in:.3f} shift={pf_sh:.3f}")
    print(f"saved table + figures to {run_dir}")


if __name__ == "__main__":
    main()