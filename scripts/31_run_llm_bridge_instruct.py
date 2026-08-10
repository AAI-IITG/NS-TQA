"""31 - Faithful LLM-bridge baseline (WO-2A): frozen Qwen2.5-Instruct on our benchmark.

A frozen instruct LLM (Qwen2.5-1.5B/7B-Instruct) with a trainable time-series encoder
projected into its embedding space, scored by the `` yes``/`` no`` next-token logits
(``src/models/llm_bridge_instruct.py``). Trained on the SAME train split + answer
labels as the other baselines; nothing privileged. Reports in-dist vs shifted answer
accuracy per dataset, and (optionally) the planted-shortcut necessity collapse.

Environment (cluster): apptainer ``llm-runtime.sif`` + ``PYTHONPATH=$HOME/hf_shim``
(huggingface_hub shim) + ``HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1``; model read from a
local path.

Run:  python scripts/31_run_llm_bridge_instruct.py --config configs/llm_bridge_instruct.yaml [--sanity] [--datasets xjtu]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import yaml

from benchmark.realdata import build_real_benchmark
from benchmark.spurious_real import attach_spurious_channel
from models.llm_bridge_instruct import (eval_instruct_bridge, load_frozen_instruct_lm,
                                        train_instruct_bridge)


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    return (sum(xs) / len(xs), st.pstdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)


def build_adapter(a):
    if "rul_cap" in a:
        from benchmark.adapters_cmapss import CMAPSSAdapter
        return CMAPSSAdapter(root=ROOT / a["root"], subsets=sorted(set(["FD001", "FD002", "FD004"])),
                             rul_cap=a["rul_cap"], flat_std_thresh=a.get("flat_std_thresh", 1e-6),
                             min_qspan=a.get("min_qspan", 0.05),
                             op_normalize=a.get("op_normalize", False))
    from benchmark.adapters_xjtu import XJTUAdapter
    return XJTUAdapter(root=ROOT / a["root"], fs=a["fs"], n_bands=a["n_bands"],
                       min_snapshots=a.get("min_snapshots", 1), cache_path=ROOT / a["cache_path"])


def build_bm(dcfg):
    a, b = dcfg["adapter"], dcfg["build"]
    bm = build_real_benchmark(
        build_adapter(a), T=b["T"], stride=b["stride"], depths=tuple(b["depths"]),
        shift=b["shift"], train_conditions=tuple(b["train_conditions"]),
        test_conditions=tuple(b["test_conditions"]), indist_holdout_frac=b["indist_holdout_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=a.get("hi_q", 0.85), lo_q=a.get("lo_q", 0.15), smooth_k=b["smooth_k"],
        a_level=b["a_level"], allow_until=b["allow_until"],
        max_windows_per_unit=b.get("max_windows_per_unit"), over_factor=b["over_factor"],
        seed=b["build_seed"])
    return bm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "llm_bridge_instruct.yaml"))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--sanity", action="store_true", help="overfit 200 instances, report TRAIN acc")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tp = cfg["train"]
    use_lora = cfg.get("use_lora", False)
    print(f"loading frozen LM {cfg['model_path']} ({cfg.get('dtype','fp32')}) on {device} ...", flush=True)
    lm, tok = load_frozen_instruct_lm(cfg["model_path"], device, dtype=cfg.get("dtype", "fp32"))
    n_lm = sum(p.numel() for p in lm.parameters())
    print(f"  LM params={n_lm/1e9:.2f}B  hidden={lm.config.hidden_size}  lora={use_lora}", flush=True)

    def fresh_lm():
        """LoRA injects adapters INTO the base model in place, so reusing one LM across
        seeds stacks adapters and warm-starts later seeds (seeds would not be
        independent). Reload a clean LM per training call whenever LoRA is on."""
        if not use_lora:
            return lm, tok
        m, t = load_frozen_instruct_lm(cfg["model_path"], device, dtype=cfg.get("dtype", "fp32"))
        return m, t

    dsets = list(cfg["datasets"].keys())
    if args.datasets:
        dsets = [d for d in dsets if d in set(args.datasets.split(","))]

    # ---- overfit sanity: fit 200 instances, must reach >0.95 TRAIN acc ----
    if args.sanity:
        bm = build_bm(cfg["datasets"][dsets[0]])
        insts = bm["train"][:200]
        print(f"overfit sanity on {len(insts)} {dsets[0]} instances ...", flush=True)
        m, t = fresh_lm()
        model = train_instruct_bridge(insts, bm["meta"]["n_channels"], m, t, device,
                                      epochs=tp.get("sanity_epochs", 40), lr=tp["lr"],
                                      batch_size=tp["batch_size"], grad_accum=tp["grad_accum"],
                                      n_tokens=cfg["n_tokens"], use_lora=use_lora,
                                      seed=0, verbose=True)
        acc = eval_instruct_bridge(model, insts, device)["answer_accuracy"]
        print(f"\nSANITY train acc = {acc:.3f}  -> {'PASS (>0.95)' if acc > 0.95 else 'FAIL'}")
        return

    seeds = cfg["experiment"]["seeds"]
    results = {}
    for dname in dsets:
        bm = build_bm(cfg["datasets"][dname])
        C = bm["meta"]["n_channels"]
        print(f"\n### {dname}: C={C} train={len(bm['train'])} indist={len(bm['test_indist'])} "
              f"shift={len(bm['test_shift'])}", flush=True)
        indist, shift = [], []
        # planted-shortcut necessity (optional): does the bridge lock onto the shortcut?
        do_short = cfg.get("run_shortcut", False) and bm["test_shift"]
        sc_indist, sc_shift = [], []
        for seed in seeds:
            print(f"  {dname} seed {seed}: training bridge ...", flush=True)
            m, t = fresh_lm()                    # clean LM per seed (LoRA is in-place)
            model = train_instruct_bridge(bm["train"], C, m, t, device,
                                          epochs=tp["epochs"], lr=tp["lr"], batch_size=tp["batch_size"],
                                          grad_accum=tp["grad_accum"], n_tokens=cfg["n_tokens"],
                                          use_lora=use_lora, seed=seed)
            indist.append(eval_instruct_bridge(model, bm["test_indist"], device)["answer_accuracy"])
            if bm["test_shift"]:
                shift.append(eval_instruct_bridge(model, bm["test_shift"], device)["answer_accuracy"])
            del model, m
            torch.cuda.empty_cache()
            if do_short:
                bs = attach_spurious_channel(bm, shift_mode="shift_flip", strength=cfg.get("spurious_strength", 3.0), seed=seed)
                m2, t2 = fresh_lm()              # clean LM for the shortcut arm too
                ms = train_instruct_bridge(bs["train"], bs["meta"]["n_channels"], m2, t2, device,
                                           epochs=tp["epochs"], lr=tp["lr"], batch_size=tp["batch_size"],
                                           grad_accum=tp["grad_accum"], n_tokens=cfg["n_tokens"],
                                           use_lora=use_lora, seed=seed)
                sc_indist.append(eval_instruct_bridge(ms, bs["test_indist"], device)["answer_accuracy"])
                sc_shift.append(eval_instruct_bridge(ms, bs["test_shift"], device)["answer_accuracy"])
                del ms, m2
                torch.cuda.empty_cache()
        results[dname] = {"indist": indist, "shift": shift,
                          "sc_indist": sc_indist, "sc_shift": sc_shift, "C": C}

    _write(cfg, results, seeds, n_lm, ROOT / cfg["run_root"])


def _write(cfg, results, seeds, n_lm, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(cfg["model_path"]).name
    L = [f"# LLM-bridge (faithful, WO-2A): frozen {name} ({n_lm/1e9:.2f}B), {len(seeds)} seeds", "",
         "Frozen instruct LLM + trainable TS encoder/projection, scored by yes/no next-token "
         "logits. Same train split + answer labels as the other baselines. "
         "Compare against NS-TQA / STL-only / best end-to-end in the main tables.", "",
         "| dataset | in-dist | shift |", "|---|---|---|"]
    for d, r in results.items():
        mi, si = mean_std(r["indist"]); ms, ss = mean_std(r["shift"])
        shift_s = "—" if ms != ms else f"{ms:.3f}±{ss:.3f}"
        L.append(f"| {d} | {mi:.3f}±{si:.3f} | {shift_s} |")
    if any(r["sc_shift"] for r in results.values()):
        L += ["", "### Planted-shortcut necessity (does the bridge collapse to the shortcut?)",
              "", "| dataset | in-dist (w/ shortcut) | shift (flip) |", "|---|---|---|"]
        for d, r in results.items():
            if r["sc_shift"]:
                mi, _ = mean_std(r["sc_indist"]); ms, _ = mean_std(r["sc_shift"])
                L.append(f"| {d} | {mi:.3f} | {ms:.3f} |")
    table = "\n".join(L)
    (out_dir / "llm_bridge_instruct.md").write_text(table)
    (out_dir / "llm_bridge_instruct.json").write_text(json.dumps(
        {"model": name, "n_params": n_lm, "seeds": seeds, "results": results}, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote -> {out_dir}/llm_bridge_instruct.{{md,json}}")


if __name__ == "__main__":
    main()
