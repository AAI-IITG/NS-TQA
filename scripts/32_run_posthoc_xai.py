"""32 - By-construction vs post-hoc explanation faithfulness (WO-2B, the money table).

Extends the Phase-J.5 comparison (``scripts/26``) to MULTIPLE post-hoc attribution
methods and BOTH datasets, on identical faithfulness axes. NS-TQA is explained by its
executor (by construction); black-box baselines are explained post-hoc with
gradient x input, Integrated Gradients, SHAP (if available), and attention rollout
(Transformer). Every explanation is reduced to the SAME tuple (critical interval +
supporting channel; ``src/explain/extract.py``) and scored by the SAME metrics
(supporting-channel accuracy, interval IoU vs the oracle interval, mask-and-re-answer
counterfactual validity), with a random-interval IoU chance row and Wilcoxon tests.

Run:  python scripts/32_run_posthoc_xai.py [--config configs/posthoc_xai.yaml]
             [--datasets xjtu,cmapss] [--quick]
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
from benchmark.necessity import encode_program, train_baseline
from benchmark.realdata import build_real_benchmark
from explain.extract import masked_counterfactual, saliency_to_explanation
from explain.posthoc import (attention_rollout, gradient_input,
                             integrated_gradients, shap_gradient)
from models.nstqa_learned import LearnedNSTQA
from perception.learned import train_perception
from utils.faithfulness import (_counterfactual_flips, _expected_random_iou,
                                _interval, _iou, governing_leaf, leakage_probe)
from utils.stats import wilcoxon

RADIUS = 2
POSTHOC = ["grad-input", "integrated-gradients", "SHAP", "attention-rollout"]


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    return (sum(xs) / len(xs), stats.pstdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)


def build_adapter(a):
    if "rul_cap" in a:
        from benchmark.adapters_cmapss import CMAPSSAdapter
        return CMAPSSAdapter(root=ROOT / a["root"], subsets=sorted(set(["FD001", "FD002", "FD004"])),
                             rul_cap=a["rul_cap"], flat_std_thresh=a.get("flat_std_thresh", 1e-6),
                             min_qspan=a.get("min_qspan", 0.05), op_normalize=a.get("op_normalize", False))
    from benchmark.adapters_xjtu import XJTUAdapter
    return XJTUAdapter(root=ROOT / a["root"], fs=a["fs"], n_bands=a["n_bands"],
                       min_snapshots=a.get("min_snapshots", 1), cache_path=ROOT / a["cache_path"])


def build_bm(dcfg):
    a, b = dcfg["adapter"], dcfg["build"]
    return build_real_benchmark(
        build_adapter(a), T=b["T"], stride=b["stride"], depths=tuple(b["depths"]),
        shift=b["shift"], train_conditions=tuple(b["train_conditions"]),
        test_conditions=tuple(b["test_conditions"]), indist_holdout_frac=b["indist_holdout_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=a.get("hi_q", 0.85), lo_q=a.get("lo_q", 0.15), smooth_k=b["smooth_k"],
        a_level=b["a_level"], allow_until=b["allow_until"],
        max_windows_per_unit=b.get("max_windows_per_unit"), over_factor=b["over_factor"],
        seed=b["build_seed"])


def logit_fn_for(model, q):
    return lambda Xb: model(Xb, q.expand(Xb.shape[0], -1))


def answer_fn_for(model, q, device):
    def f(X):
        with torch.no_grad():
            return model(X.unsqueeze(0).to(device), q.unsqueeze(0)).item() > 0.0
    return f


def run_dataset(cfg, dname, dcfg, seeds, device, quick):
    bm = build_bm(dcfg)
    C, T = bm["meta"]["n_channels"], dcfg["build"]["T"]
    insts = [i for i in bm["test_shift"] if i.mu_star is not None]
    print(f"\n### {dname}: C={C} T={T} shift-instances={len(insts)}", flush=True)
    methods = ["NS-TQA (by-construction)"] + POSTHOC
    per_seed = {m: {"support": [], "iou": [], "cf": []} for m in methods}
    iou_chance = []
    p_ep = 10 if quick else cfg["perception"]["epochs"]
    t_cfg = dict(cfg["lstm"], epochs=8 if quick else cfg["lstm"]["epochs"])
    bg = torch.stack([i.X for i in bm["train"][:32]])          # SHAP background

    for seed in seeds:
        print(f"  seed {seed}: training perception + TCN + Transformer ...", flush=True)
        pres = train_perception(bm["train"], n_channels=C, hidden=cfg["perception"]["hidden"],
                                kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
                                per_channel=cfg["perception"]["per_channel"], epochs=p_ep,
                                batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
                                weight_decay=cfg["perception"]["weight_decay"], device_pref="cpu",
                                seed=seed, verbose=False)
        nst = LearnedNSTQA(pres.model, n_channels=C)
        pidx = nst.pidx
        tcn = train_baseline("tcn", bm["train"], C, T, t_cfg, device, seed, verbose=False)
        trf = train_baseline("transformer", bm["train"], C, T, t_cfg, device, seed, verbose=False)
        acc = {m: {"support": [], "iou": [], "cf": []} for m in methods}

        for inst in insts:
            g_leaf, g_t = governing_leaf(inst.mu_star, inst.phi_star, pidx, 0)
            if g_leaf is None or g_t is None:
                continue
            gold = _interval(g_t, RADIUS, T)
            iou_chance.append(_expected_random_iou(gold, RADIUS, T))
            q = encode_program(inst.phi_star, C, T).to(device)

            # NS-TQA by-construction
            mu_hat = nst.perceive(inst.X)
            h_leaf, h_t = governing_leaf(mu_hat, inst.phi_star, pidx, 0)
            if h_leaf is not None:
                acc["NS-TQA (by-construction)"]["support"].append(float(h_leaf.channel == g_leaf.channel))
                acc["NS-TQA (by-construction)"]["iou"].append(_iou(_interval(h_t, RADIUS, T), gold))
                acc["NS-TQA (by-construction)"]["cf"].append(
                    float(_counterfactual_flips(mu_hat, inst.phi_star, pidx, h_leaf, h_t)))

            # post-hoc on the TCN (grad/IG/SHAP) + Transformer (attention)
            sal = {
                "grad-input": gradient_input(logit_fn_for(tcn, q), inst.X.to(device)),
                "integrated-gradients": integrated_gradients(logit_fn_for(tcn, q), inst.X.to(device),
                                                             steps=cfg.get("ig_steps", 32)),
                "SHAP": shap_gradient(tcn, q, inst.X, bg, device),
                "attention-rollout": attention_rollout(trf, q, inst.X, device),
            }
            for m in POSTHOC:
                ex = saliency_to_explanation(sal[m], RADIUS, T)
                if ex["channel"] is None:
                    continue
                acc[m]["support"].append(float(ex["channel"] == g_leaf.channel))
                acc[m]["iou"].append(_iou(ex["interval"], gold))
                base_model = trf if m == "attention-rollout" else tcn
                acc[m]["cf"].append(masked_counterfactual(answer_fn_for(base_model, q, device),
                                                          inst.X, ex["interval"]))
        for m in methods:
            for k in ("support", "iou", "cf"):
                per_seed[m][k].append(mean_std(acc[m][k])[0])

    leak = leakage_probe(insts)
    leak_d2 = mean_std([v for d, v in leak.items() if isinstance(d, int) and d >= 2])[0] if leak else float("nan")
    return {"C": C, "methods": methods, "per_seed": per_seed,
            "iou_chance": mean_std(iou_chance)[0], "leak_d2": leak_d2, "n_inst": len(insts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "posthoc_xai.yaml"))
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = select_device(cfg.get("device", "cpu"))
    seeds = cfg["experiment"]["seeds"][:1 if args.quick else None]
    dsets = list(cfg["datasets"].keys())
    if args.datasets:
        dsets = [d for d in dsets if d in set(args.datasets.split(","))]
    results = {d: run_dataset(cfg, d, cfg["datasets"][d], seeds, device, args.quick) for d in dsets}
    _write(results, seeds, ROOT / cfg["run_root"])


def _write(results, seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    L = [f"# By-construction vs post-hoc explanation faithfulness (WO-2B) — {len(seeds)} seeds", "",
         "NS-TQA is explained by its executor; black boxes are explained post-hoc "
         "(gradient×input, Integrated Gradients, SHAP, attention rollout), reduced to the "
         "same (interval, channel) tuple and scored on the same axes. `p` = one-sample "
         "Wilcoxon of NS-TQA vs the method across seeds.", ""]
    for dname, D in results.items():
        ps = D["per_seed"]
        L.append(f"## {dname}  (C={D['C']}, shift instances={D['n_inst']}, "
                 f"IoU chance={D['iou_chance']:.3f}, leakage d≥2={D['leak_d2']:.3f})")
        L += ["", "| method | support acc | interval IoU | counterfactual |", "|---|---|---|---|"]
        ns = ps["NS-TQA (by-construction)"]
        for m in D["methods"]:
            r = ps[m]
            cells = []
            for k in ("support", "iou", "cf"):
                mu, sd = mean_std(r[k])
                tag = ""
                if m != "NS-TQA (by-construction)" and len(r[k]) >= 2 and len(ns[k]) == len(r[k]):
                    # directional: NS-TQA is hypothesised HIGHER on all three axes (one-sided)
                    w = wilcoxon(ns[k], r[k], alternative="greater")
                    tag = f" (p={w['p']:.3f})" if w["p"] == w["p"] else ""
                cells.append("—" if mu != mu else f"{mu:.3f}±{sd:.3f}{tag}")
            bold = "**" if m.startswith("NS-TQA") else ""
            L.append(f"| {bold}{m}{bold} | " + " | ".join(cells) + " |")
        L.append("")
    table = "\n".join(L)
    (out_dir / "posthoc_xai.md").write_text(table)
    (out_dir / "posthoc_xai.json").write_text(json.dumps(
        {"seeds": seeds, "results": results}, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote -> {out_dir}/posthoc_xai.{{md,json}}")


if __name__ == "__main__":
    main()
