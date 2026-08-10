"""40 - Expert-study pack (WO-5C, generation only).

Renders worked-explanation cards for a small blinded human study: for each selected
instance, the NS-TQA by-construction explanation (critical interval + supporting
predicate + answer + robustness) and a matched post-hoc saliency explanation of a
black-box baseline on the SAME instance, presented as anonymised ``Explanation A`` /
``Explanation B`` (order randomised per card). A questionnaire (5-point Likert +
A/B preference) accompanies the cards. The study itself -- shown to 3-5 PHM
practitioners -- is run by the authors; this script only generates the materials.

Run:  python scripts/40_make_expert_pack.py [--datasets xjtu,cmapss] [--per 3] [--seed 0]
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
import yaml

from benchmark.baseline import select_device
from benchmark.necessity import encode_program, train_baseline
from benchmark.realdata import build_real_benchmark
from executor.hard_logic import evaluate
from explain.extract import saliency_to_explanation
from explain.posthoc import gradient_input, integrated_gradients
from models.nstqa_learned import LearnedNSTQA
from perception.learned import train_perception
from utils.faithfulness import _interval, governing_leaf

RADIUS = 2
_FAMILY = {"high": "is high", "low": "is low", "rising": "is rising", "falling": "is falling"}
_AXIS = {"h": "Horizontal", "v": "Vertical"}
_FEAT = {"rms": "RMS", "kurtosis": "kurtosis", "peak": "peak amplitude", "crest": "crest factor",
         "fft_b0": "low-band energy", "fft_b1": "mid-band energy", "fft_b2": "high-band energy"}


def chan_name(names, c):
    n = names[c] if c < len(names) else f"ch{c}"
    if "_" in n:
        ax, _, ft = n.partition("_")
        return f"{_AXIS.get(ax, ax)} {_FEAT.get(ft, ft)}"
    return n


def humanize(node, names):
    from executor.grammar import (Always, And, Eventually, Not, Or, Predicate, Until)
    if isinstance(node, Predicate):
        return f"{chan_name(names, node.channel)} {_FAMILY.get(node.name, node.name)}"
    if isinstance(node, Not):
        return f"not ({humanize(node.child, names)})"
    if isinstance(node, And):
        return f"({humanize(node.left, names)}) and ({humanize(node.right, names)})"
    if isinstance(node, Or):
        return f"({humanize(node.left, names)}) or ({humanize(node.right, names)})"
    if isinstance(node, Eventually):
        return f"at some snapshot in [{node.a},{node.b}], {humanize(node.child, names)}"
    if isinstance(node, Always):
        return f"throughout [{node.a},{node.b}], {humanize(node.child, names)}"
    if isinstance(node, Until):
        return f"({humanize(node.left, names)}) until ({humanize(node.right, names)})"
    return "?"


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
        shift="condition", train_conditions=tuple(b["train_conditions"]),
        test_conditions=tuple(b["test_conditions"]), indist_holdout_frac=b["indist_holdout_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=a.get("hi_q", 0.85), lo_q=a.get("lo_q", 0.15), smooth_k=b["smooth_k"],
        a_level=b["a_level"], allow_until=b["allow_until"],
        max_windows_per_unit=b.get("max_windows_per_unit"), over_factor=b["over_factor"],
        seed=b["build_seed"])


def select(nst, insts, pidx, T, k, rng):
    picks = []
    for inst in insts:
        if inst.depth < 2:
            continue
        mu = nst.perceive(inst.X)
        rho, _ = evaluate(mu, inst.phi_star, pidx, 0)
        if (float(rho) > 0) != bool(inst.answer_star) or not bool(inst.answer_star):
            continue
        h_leaf, h_t = governing_leaf(mu, inst.phi_star, pidx, 0)
        g_leaf, _ = governing_leaf(inst.mu_star, inst.phi_star, pidx, 0)
        if h_leaf is None or g_leaf is None or h_leaf.channel != g_leaf.channel:
            continue
        col = pidx[(h_leaf.name, h_leaf.channel)]
        if float(mu[:, col].std()) < 0.1 or RADIUS + 1 > h_t or h_t > T - RADIUS - 2:
            continue
        picks.append((inst, mu, h_leaf, h_t, float(rho)))
    rng.shuffle(picks)
    return picks[:k]


def render_card(dname, idx, inst, mu, h_leaf, h_t, rho, tcn, names, C, T, out_dir, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = _interval(h_t, RADIUS, T)
    ch = h_leaf.channel
    # post-hoc saliency explanation of the TCN on the same instance
    q = encode_program(inst.phi_star, C, T)
    sal = integrated_gradients(lambda Xb: tcn(Xb, q.expand(Xb.shape[0], -1)), inst.X, steps=24)
    ph = saliency_to_explanation(sal, RADIUS, T)
    ph_ch, ph_iv = ph["channel"], ph["interval"]

    x = np.arange(T)
    # anonymise: randomise which panel is NS-TQA
    ns_left = rng.random() < 0.5
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    panels = [("A", axs[0]), ("B", axs[1])]
    which = {"A": ("nstqa" if ns_left else "posthoc"), "B": ("posthoc" if ns_left else "nstqa")}
    for tag, ax in panels:
        if which[tag] == "nstqa":
            ax.plot(x, inst.X[:, ch].numpy(), color="steelblue", marker=".", ms=3)
            ax.axvspan(lo - 0.5, hi + 0.5, color="gold", alpha=0.35)
            ax.set_ylabel(f"{chan_name(names, ch)}")
            ax.set_title(f"Explanation {tag}", fontsize=11, fontweight="bold")
        else:
            ax.plot(x, inst.X[:, ph_ch].numpy(), color="firebrick", marker=".", ms=3)
            if ph_iv:
                ax.axvspan(ph_iv[0] - 0.5, ph_iv[1] + 0.5, color="orange", alpha=0.30)
            ax.set_ylabel(f"{chan_name(names, ph_ch)}")
            ax.set_title(f"Explanation {tag}", fontsize=11, fontweight="bold")
        ax.set_xlabel("snapshot")
    fig.suptitle(f"{dname.upper()} case {idx}: unit {inst.unit_id}  |  "
                 f"Q: is it true that {humanize(inst.phi_star, names)}?  |  answer: "
                 f"{'YES' if inst.answer_star else 'NO'}", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    f = out_dir / f"card_{dname}_{idx}.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"idx": idx, "dataset": dname, "unit": inst.unit_id, "depth": inst.depth,
            "question": humanize(inst.phi_star, names), "answer": bool(inst.answer_star),
            "nstqa_panel": "A" if ns_left else "B",
            "nstqa": f"{chan_name(names, ch)} {_FAMILY.get(h_leaf.name, h_leaf.name)} in [{lo},{hi}]",
            "posthoc": f"{chan_name(names, ph_ch)} salient in {ph_iv}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "faithfulness.yaml"))
    ap.add_argument("--datasets", default="xjtu,cmapss")
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    rng = random.Random(args.seed)
    device = select_device("cpu")
    out_dir = ROOT / "runs" / "expert_pack"
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for dname in args.datasets.split(","):
        dcfg = cfg["datasets"][dname]
        bm = build_bm(dcfg)
        C, T = bm["meta"]["n_channels"], dcfg["build"]["T"]
        names = bm["meta"].get("channel_names") or build_adapter(dcfg["adapter"]).load()[0].channel_names
        pres = train_perception(bm["train"], n_channels=C, hidden=cfg["perception"]["hidden"],
                                kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
                                per_channel=cfg["perception"]["per_channel"], epochs=cfg["perception"]["epochs"],
                                batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
                                weight_decay=cfg["perception"]["weight_decay"], device_pref="cpu",
                                seed=args.seed, verbose=False)
        nst = LearnedNSTQA(pres.model, n_channels=C)
        tcn = train_baseline("tcn", bm["train"], C, T, dict(cfg.get("lstm", {"hidden": 64, "dropout": 0.2,
              "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "epochs": 40, "grad_clip_norm": 5.0,
              "log_every": 10})), device, args.seed, verbose=False)
        picks = select(nst, [i for i in bm["test_shift"] if i.mu_star is not None], nst.pidx, T, args.per, rng)
        for j, (inst, mu, hl, ht, rho) in enumerate(picks, 1):
            cards.append(render_card(dname, j, inst, mu, hl, ht, rho, tcn, names, C, T, out_dir, rng))
            print(f"  rendered card_{dname}_{j}.png", flush=True)

    # answer key + questionnaire
    key = ["# Expert-pack answer key (DO NOT show to raters)", ""]
    for c in cards:
        key.append(f"- {c['dataset']} case {c['idx']}: NS-TQA = Explanation {c['nstqa_panel']} "
                   f"(cited: {c['nstqa']}) ; post-hoc: {c['posthoc']}")
    (out_dir / "answer_key.md").write_text("\n".join(key))

    q = ["# Expert study questionnaire (blinded)", "",
         "For each case you are shown a machine-health question, its yes/no answer, and two "
         "candidate explanations (**A** and **B**), each highlighting a sensor channel and a time "
         "interval. Please rate each explanation and state a preference. (5 = strongly agree.)", "",
         "Per explanation (A and B), rate 1--5:",
         "- **Plausibility**: the highlighted evidence is a sensible reason for the answer.",
         "- **Actionability**: it points a maintenance engineer to something they could act on.",
         "- **Sufficiency**: the evidence shown is enough to justify the answer.", ""]
    for c in cards:
        q += [f"## {c['dataset'].upper()} case {c['idx']}  (`card_{c['dataset']}_{c['idx']}.png`)",
              f"Question: *is it true that {c['question']}?*  Answer: **{'YES' if c['answer'] else 'NO'}**", "",
              "| | Plausibility | Actionability | Sufficiency |", "|---|---|---|---|",
              "| Explanation A |  |  |  |", "| Explanation B |  |  |  |", "",
              "Which explanation do you prefer? [ A / B / no preference ]  ____", ""]
    (out_dir / "expert_questionnaire.md").write_text("\n".join(q))
    print(f"\nwrote {len(cards)} cards + questionnaire + answer key to {out_dir}")


if __name__ == "__main__":
    main()
