"""19 - Cross-RIG transfer matrix (Phase C): do health predicates transfer across machines?

Train the faithful NS-TQA (and the end-to-end baselines) on ONE bearing rig and
evaluate on physically different rigs. All rigs (XJTU, FEMTO/PRONOSTIA, NASA IMS)
are combined via ``MultiRigAdapter`` with ``condition = rig name``, so the standard
non-circular cross-condition build does the work. For each train rig we train the
models ONCE per seed and evaluate on every rig (the train rig's held-out bearings
= the diagonal; the other rigs = off-diagonal transfer), filling a 3x3 matrix of
NS-TQA answer accuracy (and the best baseline for contrast).

The claim: NS-TQA transfers across rigs better than end-to-end baselines, because
the named predicate vocabulary is machine-agnostic while black boxes overfit
rig-specific signal statistics.

Run:  python scripts/19_run_cross_rig.py [--config configs/cross_rig.yaml] [--quick]
"""
import argparse
import json
import statistics as stats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from benchmark.adapters_femto import FEMTOAdapter
from benchmark.adapters_ims import IMSAdapter
from benchmark.adapters_multirig import MultiRigAdapter
from benchmark.adapters_xjtu import XJTUAdapter
from benchmark.baseline import select_device
from benchmark.necessity import (eval_baseline, eval_nstqa, oracle_accuracy,
                                 train_baseline)
from benchmark.realdata import build_real_benchmark
from models.nstqa_learned import LearnedNSTQA
from models.stl_only import stl_only_evaluate
from perception.grounding import predicate_index
from perception.learned import predicate_metrics, train_perception


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return (float("nan"), 0.0)
    return (sum(xs) / len(xs), stats.pstdev(xs) if len(xs) > 1 else 0.0)


def build_multirig(cfg) -> MultiRigAdapter:
    a = cfg["adapters"]
    nb = a["n_bands"]
    children = []
    for rig in cfg["rigs"]:
        if rig == "xjtu":
            children.append(XJTUAdapter(root=ROOT / a["xjtu_root"], n_bands=nb))
        elif rig == "femto":
            children.append(FEMTOAdapter(root=ROOT / a["femto_root"], n_bands=nb))
        elif rig == "ims":
            tests = tuple(a.get("ims_tests") or ()) or None
            children.append(IMSAdapter(root=ROOT / a["ims_root"], n_bands=nb, tests=tests))
        else:
            raise ValueError(f"unknown rig {rig!r}")
    return MultiRigAdapter(children)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "cross_rig.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    b, exp = cfg["build"], cfg["experiment"]
    rigs = list(cfg["rigs"])
    seeds, baselines = exp["seeds"], exp["baselines"]
    if args.quick:
        seeds = seeds[:1]
        baselines = ["lstm"]
        cfg["lstm"]["epochs"] = 5
        cfg["perception"]["epochs"] = 10

    adapter = build_multirig(cfg)
    print(f"loading rigs {rigs} ...", flush=True)
    series = adapter.load()
    present = adapter.conditions(series)
    rigs = [r for r in rigs if r in present]                 # keep only rigs actually on disk
    C = series[0].C
    T = b["T"]
    device = select_device(cfg.get("device", "cpu"))
    pidx = predicate_index(C)
    print(f"  channels={C}; units/rig=" +
          ", ".join(f"{r}:{sum(1 for s in series if s.condition==r)}" for r in rigs))

    # STL-only = fixed hand-rule grounding + the SAME executor, NO learning. It is
    # deterministic (no seed/training), so it is evaluated once per cell. Comparing
    # NS-TQA vs STL-only across rigs is the neuro-necessity (C3b) test: if STL-only
    # matches NS-TQA cross-rig, the *learned* perception is dispensable under the
    # current (train-rig) labeling -> motivates the concept-transfer variant (H.3).
    methods = baselines + ["NS-TQA", "STL-only"]
    # cell[(method, train, test)] -> list of per-seed accuracies
    cell: dict = {m: {tr: {te: [] for te in rigs} for tr in rigs} for m in methods}
    pf1: dict = {tr: {te: [] for te in rigs} for tr in rigs}   # NS-TQA perception macro-F1

    a = cfg["adapters"]
    for train_rig in rigs:
        others = [r for r in rigs if r != train_rig]
        print(f"\n=== train on {train_rig} -> test on {[train_rig] + others} ===", flush=True)
        bm = build_real_benchmark(
            adapter, T=T, stride=b["stride"], depths=tuple(b["depths"]),
            shift="condition", train_conditions=(train_rig,), test_conditions=tuple(others),
            indist_holdout_frac=b["indist_holdout_frac"],
            n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
            hi_q=a["hi_q"], lo_q=a["lo_q"], smooth_k=b["smooth_k"], a_level=b["a_level"],
            allow_until=b["allow_until"], max_windows_per_unit=b.get("max_windows_per_unit"),
            over_factor=b["over_factor"], seed=b["build_seed"],
        )
        # evaluation pools: diagonal (held-out same rig) + each other rig
        pools = {train_rig: bm["test_indist"]}
        for te in others:
            pools[te] = [i for i in bm["test_shift"] if i.condition == te]
        print("  pool sizes: " + ", ".join(f"{k}:{len(v)}" for k, v in pools.items()))

        # STL-only: deterministic, no training -> evaluate once per cell.
        for te, insts in pools.items():
            if insts:
                cell["STL-only"][train_rig][te].append(
                    stl_only_evaluate(insts, C, a_level=b["a_level"],
                                      smooth_k=b["smooth_k"])["answer_accuracy"])

        for seed in seeds:
            print(f"  seed {seed}: training models on {train_rig} ...", flush=True)
            base_models = {name: train_baseline(name, bm["train"], C, T, cfg["lstm"],
                                                device, seed, verbose=False)
                           for name in baselines}
            pres = train_perception(
                bm["train"], n_channels=C, hidden=cfg["perception"]["hidden"],
                kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
                per_channel=cfg["perception"]["per_channel"], epochs=cfg["perception"]["epochs"],
                batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
                weight_decay=cfg["perception"]["weight_decay"],
                device_pref=cfg.get("device", "cpu"), seed=seed, verbose=False)
            nst = LearnedNSTQA(pres.model, n_channels=C)
            for te, insts in pools.items():
                if not insts:
                    continue
                for name, model in base_models.items():
                    cell[name][train_rig][te].append(
                        eval_baseline(model, insts, C, T, device)["answer_accuracy"])
                cell["NS-TQA"][train_rig][te].append(
                    eval_nstqa(nst, insts)["answer_accuracy"])
                pf1[train_rig][te].append(predicate_metrics(pres.model, insts)["macro_f1"])

    # ---- write the matrices ----
    out_dir = ROOT / cfg["run_root"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def matrix_md(method: str) -> str:
        head = "| train ＼ test | " + " | ".join(rigs) + " |"
        sep = "|---" * (len(rigs) + 1) + "|"
        lines = [f"### {method} — answer accuracy (diagonal = same-rig held-out)", "", head, sep]
        for tr in rigs:
            cells = []
            for te in rigs:
                m, s = mean_std(cell[method][tr][te])
                cells.append("—" if m != m else f"{m:.3f}")
            lines.append(f"| **{tr}** | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    blocks = [f"# Cross-rig transfer ({len(seeds)} seeds): {' / '.join(rigs)}", ""]
    blocks += [matrix_md(m) for m in methods]
    # NS-TQA perception macro-F1 matrix (how well the predicates themselves transfer)
    f1head = "| train ＼ test | " + " | ".join(rigs) + " |"
    f1lines = ["### NS-TQA perception macro-F1 (predicate transfer)", "", f1head,
               "|---" * (len(rigs) + 1) + "|"]
    for tr in rigs:
        f1lines.append(f"| **{tr}** | " + " | ".join(
            ("—" if mean_std(pf1[tr][te])[0] != mean_std(pf1[tr][te])[0]
             else f"{mean_std(pf1[tr][te])[0]:.3f}") for te in rigs) + " |")
    blocks.append("\n".join(f1lines))

    # ---- neuro-necessity (C3b) readout: NS-TQA vs STL-only, diagonal vs off-diagonal ----
    def diag_mean(method):
        vals = [mean_std(cell[method][tr][tr])[0] for tr in rigs]
        vals = [v for v in vals if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    def offdiag_mean(method):
        vals = [mean_std(cell[method][tr][te])[0] for tr in rigs for te in rigs if tr != te]
        vals = [v for v in vals if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    ns_off, stl_off = offdiag_mean("NS-TQA"), offdiag_mean("STL-only")
    ns_dia, stl_dia = diag_mean("NS-TQA"), diag_mean("STL-only")
    margin = ns_off - stl_off
    verdict = ("learned perception HELPS cross-rig (C3b supported under current labeling)"
               if margin > 0.01 else
               "STL-only matches/beats NS-TQA cross-rig -> current labeling does NOT support C3b; "
               "run the concept-transfer variant (Phase H.3)")
    readout = "\n".join([
        "### Neuro-necessity readout (C3b): NS-TQA vs STL-only",
        "",
        "| metric | NS-TQA | STL-only | NS-TQA − STL-only |",
        "|---|---|---|---|",
        f"| diagonal (same-rig) mean | {ns_dia:.3f} | {stl_dia:.3f} | {ns_dia - stl_dia:+.3f} |",
        f"| off-diagonal (transfer) mean | {ns_off:.3f} | {stl_off:.3f} | {margin:+.3f} |",
        "",
        f"**Verdict:** {verdict}.",
        "",
        "_Note: under the current build, test-rig privileged labels come from the **train-rig** "
        "calibrated grounding (`realdata.py` fits normalizer+calibrator on train only), so a fixed "
        "threshold rule can reproduce them. A negative/zero margin here is expected and motivates the "
        "concept-transfer variant (test-rig-fit labels), where STL-only's fixed thresholds should fail._",
    ])
    blocks.append(readout)

    table = "\n\n".join(blocks)

    (out_dir / "transfer_matrix.md").write_text(table)
    (out_dir / "transfer_matrix.json").write_text(json.dumps({
        "rigs": rigs, "seeds": seeds, "methods": methods,
        "accuracy": {m: {tr: {te: list(mean_std(cell[m][tr][te])) for te in rigs}
                         for tr in rigs} for m in methods},
        "perception_f1": {tr: {te: list(mean_std(pf1[tr][te])) for te in rigs} for tr in rigs},
        "n_channels": C,
    }, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote matrices to {out_dir}")


if __name__ == "__main__":
    main()
