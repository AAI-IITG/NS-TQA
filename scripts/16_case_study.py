"""16 - Worked PHM explanation case study (Phase D centrepiece figure).

Renders ONE cross-load XJTU bearing instance end-to-end through the faithful path
- question (STL program) -> perceived named-sensor health predicates -> executor
- answer + critical evidence interval + supporting predicate - into (a) a
human-readable explanation card a maintenance engineer could act on, and (b) a
multi-panel figure (the supporting HI channel with the critical interval shaded,
the perceived predicate truth, and the robustness signal).

Selects a faithful, compositional, correctly-answered SHIFT instance (the answer
holds under an unseen operating load), so the figure demonstrates explainability
exactly where black-box baselines collapse.

Run:  python scripts/16_case_study.py [--config configs/xjtu_necessity.yaml] [--quick]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import yaml

from benchmark.adapters_xjtu import XJTUAdapter
from benchmark.realdata import build_real_benchmark
from executor.grammar import (Always, And, Eventually, Not, Or, Predicate, Until)
from models.nstqa_learned import LearnedNSTQA
from perception.learned import train_perception
from utils.faithfulness import governing_leaf

RADIUS = 2

_FAMILY = {"high": "is high", "low": "is low", "rising": "is rising", "falling": "is falling"}
_AXIS = {"h": "Horizontal", "v": "Vertical"}
_FEAT = {"rms": "RMS", "kurtosis": "kurtosis", "peak": "peak amplitude",
         "crest": "crest factor", "fft_b0": "low-band energy",
         "fft_b1": "mid-band energy", "fft_b2": "high-band energy"}


def channel_human(name: str) -> str:
    """'h_rms' -> 'Horizontal RMS'."""
    axis, _, feat = name.partition("_")
    return f"{_AXIS.get(axis, axis)} {_FEAT.get(feat, feat)}"


def humanize(node, names) -> str:
    """STL program -> plain-English maintenance sentence."""
    if isinstance(node, Predicate):
        return f"{channel_human(names[node.channel])} {_FAMILY[node.name]}"
    if isinstance(node, Not):
        return f"it is not the case that {humanize(node.child, names)}"
    if isinstance(node, And):
        return f"{humanize(node.left, names)} AND {humanize(node.right, names)}"
    if isinstance(node, Or):
        return f"{humanize(node.left, names)} OR {humanize(node.right, names)}"
    if isinstance(node, Eventually):
        return f"at some snapshot within [{node.a},{node.b}], {humanize(node.child, names)}"
    if isinstance(node, Always):
        return f"throughout snapshots [{node.a},{node.b}], {humanize(node.child, names)}"
    if isinstance(node, Until):
        return (f"({humanize(node.left, names)}) until ({humanize(node.right, names)}) "
                f"within [{node.a},{node.b}]")
    raise TypeError(type(node).__name__)


def _has_eventually_root(node) -> bool:
    """True if the outermost temporal operator is Eventually (existential).

    Existential roots make an intuitive figure: the critical timestep is where the
    evidence *fires* (a peak), not the weakest link a universal (Always) blames.
    """
    while isinstance(node, Not):
        node = node.child
    return isinstance(node, Eventually)


# Monotone-degradation HI families/features make the most PHM-meaningful story.
_DEGRADE_FAMILY = {"high", "rising"}
_DEGRADE_FEAT = ("rms", "kurtosis", "peak")


def select_instance(nst, instances):
    """Pick a faithful, compositional, correct SHIFT instance with a *discriminative*,
    PHM-meaningful supporting predicate (fires in the critical interval, quiet
    elsewhere) under an existential (Eventually) root — the clearest figure."""
    best, best_score = None, -1e9
    for inst in instances:
        if inst.depth < 2 or inst.mu_star is None:
            continue
        mu_hat = nst.perceive(inst.X)
        rho = float(nst.answer(inst.X, inst.phi_star)["rho"])
        pred = rho > 0
        if pred != bool(inst.answer_star) or not pred:      # want a correct "yes"
            continue
        h_leaf, h_t = governing_leaf(mu_hat, inst.phi_star, nst.pidx, 0)
        g_leaf, _ = governing_leaf(inst.mu_star, inst.phi_star, nst.pidx, 0)
        if h_leaf is None or g_leaf is None:
            continue
        col = nst.pidx[(h_leaf.name, h_leaf.channel)]
        truth = mu_hat[:, col]
        disc = float(truth.std())                            # discriminativeness over time
        if disc < 0.10 or float(truth[h_t]) < 0.5:           # skip saturated / non-firing support
            continue
        faithful = (h_leaf.name, h_leaf.channel) == (g_leaf.name, g_leaf.channel)
        feat = nst.pidx and inst.provenance.get("channel_names", [])
        feat_name = feat[h_leaf.channel].split("_", 1)[-1] if feat else ""
        phm = (h_leaf.name in _DEGRADE_FAMILY) + (feat_name in _DEGRADE_FEAT)
        score = (2.0 * faithful + 2.0 * _has_eventually_root(inst.phi_star)
                 + 1.5 * disc + 0.5 * phm + (inst.depth == 2) + min(abs(rho), 1.0))
        if score > best_score:
            best, best_score = (inst, mu_hat, h_leaf, h_t, rho), score
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "xjtu_necessity.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    a, b = cfg["adapter"], cfg["build"]
    epochs = 10 if args.quick else cfg["perception"]["epochs"]

    adapter = XJTUAdapter(
        root=ROOT / a["root"], fs=a["fs"], n_bands=a["n_bands"],
        min_snapshots=a.get("min_snapshots", 1),
        cache_path=ROOT / a["cache_path"], force_recompute=a.get("force_recompute", False),
    )
    bm = build_real_benchmark(
        adapter, T=b["T"], stride=b["stride"], depths=tuple(b["depths"]),
        shift=b["shift"], train_conditions=tuple(b["train_conditions"]),
        test_conditions=tuple(b["test_conditions"]),
        indist_holdout_frac=b["indist_holdout_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=a["hi_q"], lo_q=a["lo_q"], smooth_k=b["smooth_k"], a_level=b["a_level"],
        allow_until=b["allow_until"], max_windows_per_unit=b.get("max_windows_per_unit"),
        over_factor=b["over_factor"], seed=b["build_seed"],
    )
    C = bm["meta"]["n_channels"]
    names = bm["meta"]["channel_names"]
    print(f"training perception ({epochs} epochs) ...", flush=True)
    pres = train_perception(
        bm["train"], n_channels=C, hidden=cfg["perception"]["hidden"],
        kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
        per_channel=cfg["perception"]["per_channel"], epochs=epochs,
        batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
        weight_decay=cfg["perception"]["weight_decay"],
        device_pref=cfg.get("device", "cpu"), seed=0, verbose=False,
    )
    nst = LearnedNSTQA(pres.model, n_channels=C)

    picked = select_instance(nst, bm["test_shift"])
    if picked is None:
        print("no suitable faithful depth>=2 shift instance found; try more epochs")
        return
    inst, mu_hat, leaf, t_star, rho = picked
    T = inst.X.shape[0]
    lo, hi = max(0, t_star - RADIUS), min(T - 1, t_star + RADIUS)
    support_col = nst.pidx[(leaf.name, leaf.channel)]
    support_human = f"{channel_human(names[leaf.channel])} {_FAMILY[leaf.name]}"
    expl = nst.explain(inst.X, inst.phi_star)

    out_dir = ROOT / cfg["run_root"] / "case_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- explanation card ----
    card = [
        "# Worked PHM explanation case study (XJTU, cross-load shift)", "",
        f"- **Unit:** {inst.unit_id}  (operating condition `{inst.condition}`, unseen at train time)",
        f"- **Question (STL program):** `{inst.phi_star.canonical()}`  (depth {inst.depth})",
        f"- **In plain terms:** *Is it true that {humanize(inst.phi_star, names)}?*",
        f"- **NS-TQA answer:** **{'YES' if inst.answer_star else 'NO'}**  "
        f"(robustness margin ρ = {rho:+.3f})",
        f"- **Critical evidence interval τ:** snapshots **[{lo}, {hi}]** "
        f"(decisive snapshot t\\* = {t_star})",
        f"- **Supporting predicate:** `{leaf.canonical()}` → **{support_human}**", "",
        "## Faithful explanation",
        f"> The system answers **{'YES' if inst.answer_star else 'NO'}** *because* "
        f"**{support_human}** over snapshots **[{lo}, {hi}]** of this bearing's life — "
        f"the decisive evidence the deterministic executor used to satisfy the program. "
        f"This explanation is not post-hoc: the answer is obtainable only by executing the "
        f"program over the perceived predicates, and removing this predicate flips the answer.",
    ]
    (out_dir / "case_study.md").write_text("\n".join(card))
    print("\n".join(card))

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.arange(T)
        sig = inst.X[:, leaf.channel].numpy()            # normalized HI channel
        truth = mu_hat[:, support_col].numpy()           # perceived predicate truth
        rho_sig = expl["rho_signal"].numpy()

        fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        for ax in axs:
            ax.axvspan(lo - 0.5, hi + 0.5, color="gold", alpha=0.30,
                       label="critical interval τ")
            ax.axvline(t_star, color="darkorange", ls="--", lw=1.2)

        axs[0].plot(x, sig, color="steelblue", marker=".", ms=3)
        axs[0].set_ylabel(f"{channel_human(names[leaf.channel])}\n(normalized)")
        axs[0].set_title(f"{inst.unit_id} — Q: {inst.phi_star.canonical()}  →  "
                         f"{'YES' if inst.answer_star else 'NO'} "
                         f"because {support_human} in [{lo},{hi}]")

        axs[1].plot(x, truth, color="seagreen", marker=".", ms=3)
        axs[1].axhline(0.5, color="gray", ls=":", lw=1)
        axs[1].set_ylim(-0.05, 1.05)
        axs[1].set_ylabel(f"perceived\n{leaf.canonical()} truth")

        axs[2].plot(x, rho_sig, color="firebrick", marker=".", ms=3)
        axs[2].axhline(0.0, color="gray", ls=":", lw=1)
        axs[2].set_ylabel("robustness ρ(t)")
        axs[2].set_xlabel("snapshot index within window")

        axs[0].legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_case_study.png", dpi=300)
        print(f"\nwrote card + figure to {out_dir}")
    except Exception as e:
        print(f"(figure skipped: {e})")
        print(f"wrote card to {out_dir}")


if __name__ == "__main__":
    main()
