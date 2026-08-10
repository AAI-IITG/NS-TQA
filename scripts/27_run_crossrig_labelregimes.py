"""27 - Cross-rig matrix under BOTH label regimes (WO-1A, journal extension).

The conference cross-rig table (scripts/19) is built under the **L-src** regime:
test-rig privileged labels come from the TRAIN-rig calibrated grounding, so a fixed
threshold rule (STL-only) reproduces them and the cross-rig win is attributable to
the executor + shared predicate vocabulary (C4), not to the *learned* perception
(C3b). WO-1A adds the **L-tgt** regime: each test rig's labels are defined by a
grounding fit on that rig's OWN training-fraction bearings (never the evaluated
windows). Under L-tgt a train-rig threshold rule need not track the labels, so if
learned perception still transfers the concept while STL-only degrades, that is the
answer-level evidence for learned-perception necessity.

This script runs the full 3x3 cross-rig matrix for methods {best end-to-end,
NS-TQA, STL-only} under BOTH regimes and writes a combined table + JSON. STL-only is
deterministic (1 "seed"); the learned methods use the configured model seeds.
See docs/labeling_regimes.md.

Run:  python scripts/27_run_crossrig_labelregimes.py [--config configs/crossrig_labelregimes.yaml]
                                                     [--regimes both|L-src|L-tgt]
                                                     [--seeds N] [--seed-subset i[,j,...]] [--quick]

Output: runs/crossrig_stlonly/crossrig_labelregimes.{md,json}
"""
import argparse
import json
import platform
import socket
import statistics as stats
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import yaml

from benchmark.adapters_femto import FEMTOAdapter
from benchmark.adapters_ims import IMSAdapter
from benchmark.adapters_multirig import MultiRigAdapter
from benchmark.adapters_xjtu import XJTUAdapter
from benchmark.baseline import select_device
from benchmark.necessity import eval_baseline, eval_nstqa, train_baseline
from benchmark.realdata import build_real_benchmark
from models.nstqa_learned import LearnedNSTQA
from models.stl_only import stl_only_evaluate
from perception.grounding import predicate_index
from perception.learned import predicate_metrics, train_perception

REGIMES = ("L-src", "L-tgt")


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


def run_regime(regime, cfg, adapter, rigs, C, T, device, seeds, baselines):
    """Fill the cross-rig cells for one label regime. Returns (cell, pf1)."""
    b, a = cfg["build"], cfg["adapters"]
    methods = baselines + ["NS-TQA", "STL-only"]
    cell = {m: {tr: {te: [] for te in rigs} for tr in rigs} for m in methods}
    pf1 = {tr: {te: [] for te in rigs} for tr in rigs}

    for train_rig in rigs:
        others = [r for r in rigs if r != train_rig]
        print(f"\n[{regime}] === train on {train_rig} -> {[train_rig] + others} ===", flush=True)
        bm = build_real_benchmark(
            adapter, T=T, stride=b["stride"], depths=tuple(b["depths"]),
            shift="condition", train_conditions=(train_rig,), test_conditions=tuple(others),
            indist_holdout_frac=b["indist_holdout_frac"],
            n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
            hi_q=a["hi_q"], lo_q=a["lo_q"], smooth_k=b["smooth_k"], a_level=b["a_level"],
            allow_until=b["allow_until"], max_windows_per_unit=b.get("max_windows_per_unit"),
            over_factor=b["over_factor"], seed=b["build_seed"],
            label_regime=regime, label_fit_frac=b.get("label_fit_frac", 0.5),
        )
        pools = {train_rig: bm["test_indist"]}
        for te in others:
            pools[te] = [i for i in bm["test_shift"] if i.condition == te]
        print("  pool sizes: " + ", ".join(f"{k}:{len(v)}" for k, v in pools.items()), flush=True)

        # STL-only: deterministic, no training -> once per cell.
        for te, insts in pools.items():
            if insts:
                cell["STL-only"][train_rig][te].append(
                    stl_only_evaluate(insts, C, a_level=b["a_level"],
                                      smooth_k=b["smooth_k"])["answer_accuracy"])

        for seed in seeds:
            print(f"  seed {seed}: training on {train_rig} ...", flush=True)
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
                cell["NS-TQA"][train_rig][te].append(eval_nstqa(nst, insts)["answer_accuracy"])
                pf1[train_rig][te].append(predicate_metrics(pres.model, insts)["macro_f1"])
    return cell, pf1, methods


def matrix_md(title, cell, method, rigs):
    head = "| train ＼ test | " + " | ".join(rigs) + " |"
    sep = "|---" * (len(rigs) + 1) + "|"
    lines = [f"#### {title}", "", head, sep]
    for tr in rigs:
        cells = [("—" if mean_std(cell[method][tr][te])[0] != mean_std(cell[method][tr][te])[0]
                  else f"{mean_std(cell[method][tr][te])[0]:.3f}") for te in rigs]
        lines.append(f"| **{tr}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def diag_off(cell, method, rigs):
    dia = [mean_std(cell[method][tr][tr])[0] for tr in rigs]
    off = [mean_std(cell[method][tr][te])[0] for tr in rigs for te in rigs if tr != te]
    dia = [v for v in dia if v == v]
    off = [v for v in off if v == v]
    return (sum(dia) / len(dia) if dia else float("nan"),
            sum(off) / len(off) if off else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "crossrig_labelregimes.yaml"))
    ap.add_argument("--regimes", default="both", choices=["both", "L-src", "L-tgt"])
    ap.add_argument("--seeds", type=int, default=None, help="override number of model seeds")
    ap.add_argument("--seed-subset", default=None,
                    help="comma-separated seed indices to run (for GPU sharding)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    exp = cfg["experiment"]
    rigs = list(cfg["rigs"])
    seeds, baselines = list(exp["seeds"]), exp["baselines"]
    if args.seeds is not None:
        seeds = list(range(args.seeds))
    if args.seed_subset is not None:
        want = {int(s) for s in args.seed_subset.split(",")}
        seeds = [s for s in seeds if s in want]
    if args.quick:
        seeds = seeds[:1]
        baselines = ["lstm"]
        cfg["lstm"]["epochs"] = 5
        cfg["perception"]["epochs"] = 10
    regimes = REGIMES if args.regimes == "both" else (args.regimes,)

    adapter = build_multirig(cfg)
    print(f"loading rigs {rigs} ...", flush=True)
    series = adapter.load()
    present = adapter.conditions(series)
    rigs = [r for r in rigs if r in present]
    C = series[0].C
    T = cfg["build"]["T"]
    device = select_device(cfg.get("device", "cpu"))
    _ = predicate_index(C)
    print(f"  channels={C}; seeds={seeds}; regimes={regimes}", flush=True)

    t0 = time.time()
    results = {}
    for regime in regimes:
        cell, pf1, methods = run_regime(regime, cfg, adapter, rigs, C, T, device, seeds, baselines)
        results[regime] = {"cell": cell, "pf1": pf1, "methods": methods}
    wall = time.time() - t0
    peak_vram = (torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)

    # ---- write combined table ----
    out_dir = ROOT / cfg.get("run_root", "runs/crossrig_stlonly")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_e2e = "NS-TQA"  # placeholder; per-cell best baseline handled below

    blocks = [f"# Cross-rig transfer under both label regimes (WO-1A)",
              "",
              f"Rigs: {' / '.join(rigs)}. Model seeds: {len(seeds)} "
              f"({seeds}). STL-only deterministic (1 evaluation).",
              "",
              "See `docs/labeling_regimes.md`. **L-src** = test labels from the train-rig "
              "grounding (conference regime); **L-tgt** = test labels from each test rig's "
              "own training-fraction grounding.", ""]

    for regime in regimes:
        cell = results[regime]["cell"]
        methods = results[regime]["methods"]
        blocks.append(f"## Regime {regime}")
        for m in methods:
            blocks.append(matrix_md(f"{m} — answer accuracy (diag = same-rig held-out)",
                                    cell, m, rigs))
        # neuro-necessity readout for this regime
        ns_d, ns_o = diag_off(cell, "NS-TQA", rigs)
        st_d, st_o = diag_off(cell, "STL-only", rigs)
        blocks.append("\n".join([
            f"#### Neuro-necessity readout ({regime}): NS-TQA vs STL-only",
            "", "| metric | NS-TQA | STL-only | NS-TQA − STL-only |", "|---|---|---|---|",
            f"| diagonal (same-rig) | {ns_d:.3f} | {st_d:.3f} | {ns_d - st_d:+.3f} |",
            f"| off-diagonal (transfer) | {ns_o:.3f} | {st_o:.3f} | {ns_o - st_o:+.3f} |",
        ]))

    # cross-regime verdict (only meaningful when both regimes ran)
    if set(regimes) == set(REGIMES):
        _, ns_o_src = diag_off(results["L-src"]["cell"], "NS-TQA", rigs)
        _, st_o_src = diag_off(results["L-src"]["cell"], "STL-only", rigs)
        _, ns_o_tgt = diag_off(results["L-tgt"]["cell"], "NS-TQA", rigs)
        _, st_o_tgt = diag_off(results["L-tgt"]["cell"], "STL-only", rigs)
        m_src, m_tgt = ns_o_src - st_o_src, ns_o_tgt - st_o_tgt
        if m_tgt > 0.01 and m_tgt > m_src:
            claim = ("STL-only fails cross-rig under L-tgt while learned perception transfers "
                     "the concept -> answer-level learned-perception necessity (C3b).")
        else:
            claim = ("STL-only transfers under L-tgt too -> the executor carries cross-rig "
                     "regardless; learned-perception necessity stays at predicate/synthetic level.")
        blocks.append("\n".join([
            "## Extracted claim (WO-1A caption)", "",
            f"NS-TQA − STL-only off-diagonal margin: L-src {m_src:+.3f}, L-tgt {m_tgt:+.3f}.", "",
            f"**{claim}** (no spin — the numbers decide.)"]))

    table = "\n\n".join(blocks)
    (out_dir / "crossrig_labelregimes.md").write_text(table)
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(), "python": platform.python_version(),
        "torch": torch.__version__, "device": str(device),
        "rigs": rigs, "seeds": seeds, "regimes": list(regimes),
        "wall_clock_s": round(wall, 1), "peak_gpu_bytes": int(peak_vram),
        "config": cfg,
        "accuracy": {reg: {m: {tr: {te: list(mean_std(results[reg]["cell"][m][tr][te]))
                                    for te in rigs} for tr in rigs}
                           for m in results[reg]["methods"]} for reg in regimes},
        "perception_f1": {reg: {tr: {te: list(mean_std(results[reg]["pf1"][tr][te]))
                                     for te in rigs} for tr in rigs} for reg in regimes},
        "n_channels": C,
    }
    tmp = out_dir / "crossrig_labelregimes.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2, default=str))
    tmp.replace(out_dir / "crossrig_labelregimes.json")
    print("\n" + table)
    print(f"\nwrote -> {out_dir}/crossrig_labelregimes.{{md,json}}  ({wall:.0f}s)")


if __name__ == "__main__":
    main()
