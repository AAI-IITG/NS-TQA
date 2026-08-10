"""24 - LLM-bridge baseline on the XJTU benchmark (Phase I.3).

Reproduces the ITFormer / Time-MQA *paradigm* (frozen pretrained LM reads the
question as natural language; a TS encoder is bridged into the LM space; a
trainable cross-modal transformer answers yes/no) and scores it on OUR axes:

  1. natural cross-load generalization  (in-dist vs shifted accuracy), and
  2. the planted-shortcut necessity test (does it collapse like the other
     end-to-end baselines, or is it structurally protected?).

The frozen LM is all-MiniLM-L6-v2 (local HF cache); only the bridge + TS encoder
+ head train. Results are reported as "an ITFormer-style bridge", not "ITFormer".

Run:  python scripts/24_run_llm_bridge.py [--config configs/xjtu_necessity.yaml]
                                          [--epochs 30] [--device cuda] [--quick]
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

import torch
import yaml

from benchmark.baseline import select_device
from benchmark.realdata import build_real_benchmark
from benchmark.spurious_real import attach_spurious_channel
from models.llm_bridge import eval_llm_bridge, load_frozen_lm, train_llm_bridge

LM_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return (float("nan"), 0.0)
    return (sum(xs) / len(xs), stats.pstdev(xs) if len(xs) > 1 else 0.0)


def build_adapter(a, b):
    """Construct the adapter from config keys: ``rul_cap`` -> C-MAPSS, else XJTU."""
    if "rul_cap" in a:                                       # C-MAPSS (tabular)
        from benchmark.adapters_cmapss import CMAPSSAdapter
        subsets = tuple(b["train_conditions"]) + tuple(b["test_conditions"])
        return CMAPSSAdapter(
            root=ROOT / a["root"], subsets=subsets, rul_cap=a["rul_cap"],
            flat_std_thresh=float(a["flat_std_thresh"]), min_qspan=a["min_qspan"],
            hi_q=a["hi_q"], lo_q=a["lo_q"])
    from benchmark.adapters_xjtu import XJTUAdapter           # XJTU (vibration)
    return XJTUAdapter(root=ROOT / a["root"], fs=a["fs"], n_bands=a["n_bands"],
                       min_snapshots=a.get("min_snapshots", 1),
                       cache_path=ROOT / a["cache_path"],
                       force_recompute=a.get("force_recompute", False))


def build_bm(cfg):
    a, b = cfg["adapter"], cfg["build"]
    adapter = build_adapter(a, b)
    return build_real_benchmark(
        adapter, T=b["T"], stride=b["stride"], depths=tuple(b["depths"]),
        shift=b["shift"], train_conditions=tuple(b["train_conditions"]),
        test_conditions=tuple(b["test_conditions"]),
        indist_holdout_frac=b["indist_holdout_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=a["hi_q"], lo_q=a["lo_q"], smooth_k=b["smooth_k"], a_level=b["a_level"],
        allow_until=b["allow_until"], max_windows_per_unit=b.get("max_windows_per_unit"),
        over_factor=b["over_factor"], seed=b["build_seed"])


def train_eval(bm, lm, tok, device, seeds, epochs):
    """Train the bridge per seed on bm['train']; return (indist, shift) acc lists."""
    C = bm["train"][0].X.shape[-1]
    indist, shift = [], []
    has_shift = bool(bm["test_shift"])
    for seed in seeds:
        model = train_llm_bridge(bm["train"], C, lm, tok, device, epochs=epochs, seed=seed)
        indist.append(eval_llm_bridge(model, bm["test_indist"], lm, tok, device)["answer_accuracy"])
        shift.append(eval_llm_bridge(model, bm["test_shift"], lm, tok, device)["answer_accuracy"]
                     if has_shift else None)
    return indist, shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "xjtu_necessity.yaml"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    pref = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = select_device(pref)
    seeds = cfg["experiment"]["seeds"]
    modes = cfg.get("spurious", {}).get("shift_modes", ["shift_decorr", "shift_flip"])
    strength = cfg.get("spurious", {}).get("strength", 3.0)
    epochs = args.epochs
    if args.quick:
        seeds = seeds[:1]
        epochs = 30
        modes = modes[:1]

    print(f"device={device}; loading frozen LM {LM_NAME} ...", flush=True)
    tok, lm = load_frozen_lm(LM_NAME, device)
    print(f"  frozen LM params: {sum(p.numel() for p in lm.parameters()):,} (not trained)")

    is_cmapss = "rul_cap" in cfg["adapter"]
    dset = "C-MAPSS (FD001->FD002/4)" if is_cmapss else "XJTU cross-load"
    # reference numbers from the main tables, per dataset
    ref = ({"NS-TQA": ("0.955", "0.925"), "TCN": ("0.810", "0.803"), "STL": "0.983"}
           if is_cmapss else
           {"NS-TQA": ("0.821", "0.831"), "TCN": ("0.663", "0.679"), "STL": "0.912"})
    print(f"building {dset} benchmark ...", flush=True)
    bm = build_bm(cfg)
    C = bm["train"][0].X.shape[-1]
    print(f"  channels={C}; train={len(bm['train'])} indist={len(bm['test_indist'])} "
          f"shift={len(bm['test_shift'])}; seeds={list(seeds)} epochs={epochs}")

    # ---- 1. natural cross-load generalization ----
    print("\n=== natural cross-load generalization ===", flush=True)
    g_in, g_sh = train_eval(bm, lm, tok, device, seeds, epochs)
    gi, gis = mean_std(g_in)
    gs, gss = mean_std(g_sh)
    print(f"  LLM-bridge: in-dist {gi:.3f}±{gis:.3f} | shift {gs:.3f}±{gss:.3f}")

    # ---- 2. planted-shortcut necessity ----
    spur = {}
    for mode in modes:
        print(f"\n=== planted shortcut: {mode} (strength {strength}) ===", flush=True)
        sbm = attach_spurious_channel(bm, shift_mode=mode, strength=strength, seed=0)
        s_in, s_sh = train_eval(sbm, lm, tok, device, seeds, epochs)
        si, _ = mean_std(s_in)
        ss, _ = mean_std(s_sh)
        spur[mode] = (si, ss)
        print(f"  LLM-bridge: in-dist {si:.3f} | shift {ss:.3f}  "
              f"(end-to-end models collapse to the shortcut here)")

    # ---- write table ----
    out_dir = ROOT / cfg["run_root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    L = [f"# LLM-bridge (ITFormer-style) on {dset} ({len(seeds)} seeds, {epochs} ep)",
         "",
         "_Frozen all-MiniLM-L6-v2 + TS encoder + trainable cross-modal bridge -> yes/no. "
         "Reference numbers (NS-TQA, best end-to-end TCN, STL-only) are from the main tables._", "",
         "### Natural distribution-shift generalization (answer accuracy)", "",
         "| Method | in-dist/in-regime | shift |", "|---|---|---|",
         f"| **LLM-bridge (ours, ITFormer-style)** | {gi:.3f}±{gis:.3f} | {gs:.3f}±{gss:.3f} |",
         f"| NS-TQA (ref) | {ref['NS-TQA'][0]} | {ref['NS-TQA'][1]} |",
         f"| best end-to-end TCN (ref) | {ref['TCN'][0]} | {ref['TCN'][1]} |",
         f"| STL-only (ref) | — | {ref['STL']} |", "",
         "### Planted-shortcut necessity (answer accuracy)", "",
         "| Shortcut shift | LLM-bridge in-dist | LLM-bridge shift |", "|---|---|---|"]
    for mode in modes:
        si, ss = spur[mode]
        L.append(f"| {mode} | {si:.3f} | {ss:.3f} |")
    L += ["",
          "_If the in-dist column is ~1.0 and the shift column drops to chance "
          "(`shift_decorr`) or ~0 (`shift_flip`), the LLM-bridge has taken the spurious "
          "shortcut and collapses under shift --- like LSTM/Transformer/TCN, and unlike "
          "NS-TQA, which is structurally unable to read the spurious channel._"]
    table = "\n".join(L)
    (out_dir / "llm_bridge.md").write_text(table)
    (out_dir / "llm_bridge.json").write_text(json.dumps({
        "device": str(device), "seeds": list(seeds), "epochs": epochs,
        "generalization": {"indist": list(mean_std(g_in)), "shift": list(mean_std(g_sh))},
        "spurious": {m: {"indist": spur[m][0], "shift": spur[m][1]} for m in modes},
    }, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote {out_dir / 'llm_bridge.md'}")


if __name__ == "__main__":
    main()
