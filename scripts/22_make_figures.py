"""22 - Generate the paper's result figures into ``paper/figures/`` from runs/.

Produces (matplotlib, deterministic from the recorded run JSONs except the case
study, which trains one perception seed):

  fig_necessity_bar.png   synthetic necessity: baselines collapse, NS-TQA invariant
  fig_depth_curve.png     accuracy vs program depth (shift): NS-TQA flat
  fig_generalization.png  cross-condition generalization (C-MAPSS + XJTU)
  fig_crossrig.png        3x3 cross-rig transfer heatmap (NS-TQA)
  fig_case_study.png      a real bearing question -> answer + critical window (comprehensive)

Run:  python scripts/22_make_figures.py [--config configs/xjtu_necessity.yaml] [--quick] [--no-case-study]
"""
import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = ROOT / "paper" / "figures"
RUNS = ROOT / "runs"
PERC, EXEC, DATA, SHORT, GRAY = "#2C7A7B", "#C8841E", "#6B46C1", "#B0413E", "#9AA0A6"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})


def _load(p):
    p = RUNS / p
    return json.load(open(p)) if p.exists() else None


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / name, dpi=300); plt.close(fig)
    print(f"  wrote {name}")


# --------------------------------------------------------------------------- #
def fig_necessity_bar():
    dec = _load("necessity/shift_decorr/multiseed/results.json")
    flip = _load("necessity/shift_flip/multiseed/results.json")
    if not dec:
        print("  (skip necessity_bar: no run json)"); return
    methods = dec["meta"]["baselines"] + ["NS-TQA"]
    labels = [m.upper() if m != "NS-TQA" else m for m in methods]

    def acc(d, m, reg): return d["aggregate"][m][reg]["answer_accuracy"][0]
    def err(d, m, reg): return d["aggregate"][m][reg]["answer_accuracy"][1]
    ind = [acc(dec, m, "indist") for m in methods]
    sde = [acc(dec, m, "shift") for m in methods]
    sfl = [acc(flip, m, "shift") for m in methods] if flip else sde
    sde_e = [err(dec, m, "shift") for m in methods]

    x = np.arange(len(methods)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - w, ind, w, color=GRAY, label="in-distribution")
    ax.bar(x, sde, w, yerr=sde_e, capsize=3, color=EXEC, label="shift (decorrelated)")
    ax.bar(x + w, sfl, w, color=SHORT, label="shift (flipped)")
    ax.axhline(0.5, ls=":", c="k", lw=1, alpha=.6)
    ax.text(len(methods) - 1.4, 0.52, "chance", fontsize=8, color="k", alpha=.7)
    for i, m in enumerate(methods):
        ax.bar(x[i] - w, ind[i], w, color=(PERC if m == "NS-TQA" else GRAY))
        ax.bar(x[i], sde[i], w, color=(PERC if m == "NS-TQA" else EXEC))
        ax.bar(x[i] + w, sfl[i], w, color=(PERC if m == "NS-TQA" else SHORT))
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.05)
    ax.set_ylabel("answer accuracy")
    ax.set_title("Necessity (synthetic): end-to-end models collapse under shift; NS-TQA invariant")
    ax.legend(loc="center right", fontsize=8, framealpha=.9)
    _save(fig, "fig_necessity_bar.png")


def fig_depth_curve():
    dec = _load("necessity/shift_decorr/multiseed/results.json")
    if not dec:
        print("  (skip depth_curve)"); return
    methods = dec["meta"]["baselines"] + ["NS-TQA"]
    depths = sorted(int(d) for d in dec["aggregate"]["NS-TQA"]["shift"]["by_depth"])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for m in methods:
        bd = dec["aggregate"][m]["shift"]["by_depth"]
        ys = [bd[str(d)][0] for d in depths]
        es = [bd[str(d)][1] for d in depths]
        is_ns = m == "NS-TQA"
        ax.errorbar(depths, ys, yerr=es, marker="o", lw=2 if is_ns else 1.3,
                    color=PERC if is_ns else None, label=m if is_ns else m.upper(),
                    capsize=2, alpha=1 if is_ns else .8)
    ax.axhline(0.5, ls=":", c="k", lw=1, alpha=.6)
    ax.set_xticks(depths); ax.set_ylim(0, 1.02)
    ax.set_xlabel("program depth"); ax.set_ylabel("answer accuracy (shifted)")
    ax.set_title("Composition: NS-TQA stays flat across depth")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "fig_depth_curve.png")


def fig_generalization():
    cm = _load("cmapss_necessity/cmapss_necessity.json")
    xj = _load("xjtu_necessity/xjtu_generalization.json")
    if not (cm and xj):
        print("  (skip generalization)"); return

    def best_base(j, reg):
        return max(j["agg"][m][reg][0] for m in ("lstm", "transformer", "tcn"))
    groups = [("C-MAPSS\nFD001→FD002/4", cm), ("XJTU-SY\n35Hz→37.5/40", xj)]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, (title, j) in zip(axes, groups):
        regs = ["indist", "shift"]
        bb = [best_base(j, r) for r in regs]
        ns = [j["agg"]["NS-TQA"][r][0] for r in regs]
        nse = [j["agg"]["NS-TQA"][r][1] for r in regs]
        x = np.arange(2); w = 0.36
        ax.bar(x - w / 2, bb, w, color=GRAY, label="best end-to-end")
        ax.bar(x + w / 2, ns, w, yerr=nse, capsize=3, color=PERC, label="NS-TQA")
        ax.axhline(1.0, ls=":", c=EXEC, lw=1, alpha=.7)
        ax.text(1.1, 1.01, "oracle", fontsize=8, color=EXEC)
        ax.set_xticks(x); ax.set_xticklabels(["in-regime", "shifted"])
        ax.set_title(title, fontsize=10); ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("answer accuracy"); axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Generalization under natural operating-condition shift", y=1.02)
    _save(fig, "fig_generalization.png")


def fig_crossrig():
    cr = _load("cross_rig/transfer_matrix.json")
    if not cr:
        print("  (skip crossrig)"); return
    rigs = cr["rigs"]
    M = np.array([[cr["accuracy"]["NS-TQA"][tr][te][0] for te in rigs] for tr in rigs])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(M, cmap="YlGn", vmin=0.55, vmax=0.95)
    ax.set_xticks(range(len(rigs))); ax.set_xticklabels([r.upper() for r in rigs])
    ax.set_yticks(range(len(rigs))); ax.set_yticklabels([r.upper() for r in rigs])
    ax.set_xlabel("test rig"); ax.set_ylabel("train rig")
    for i in range(len(rigs)):
        for j in range(len(rigs)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=11,
                    fontweight="bold" if i == j else "normal")
    ax.set_title("Cross-rig transfer (NS-TQA accuracy)\ndiagonal = same-rig held-out", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="answer accuracy")
    _save(fig, "fig_crossrig.png")


# --------------------------------------------------------------------------- #
# worked PHM explanation case study, per dataset
# --------------------------------------------------------------------------- #
_FAMILY = {"high": "is high", "low": "is low", "rising": "is rising", "falling": "is falling"}
_VAX = {"h": "Horizontal", "v": "Vertical"}
_VFE = {"rms": "RMS", "kurtosis": "kurtosis", "peak": "peak amplitude", "crest": "crest factor",
        "fft_b0": "low-band energy", "fft_b1": "mid-band energy", "fft_b2": "high-band energy"}


def vibration_namer(name):
    axis, _, feat = name.partition("_")
    return f"{_VAX.get(axis, axis)} {_VFE.get(feat, feat)}"


def cmapss_namer(name):
    return ("operating setting " if name.startswith("op") else "sensor ") + name


def humanize_g(node, names, namer):
    from executor.grammar import (Always, And, Eventually, Not, Or, Predicate, Until)
    if isinstance(node, Predicate):
        return f"{namer(names[node.channel])} {_FAMILY[node.name]}"
    if isinstance(node, Not):
        return f"it is not the case that {humanize_g(node.child, names, namer)}"
    if isinstance(node, And):
        return f"{humanize_g(node.left, names, namer)} AND {humanize_g(node.right, names, namer)}"
    if isinstance(node, Or):
        return f"{humanize_g(node.left, names, namer)} OR {humanize_g(node.right, names, namer)}"
    if isinstance(node, Eventually):
        return f"at some point within [{node.a},{node.b}], {humanize_g(node.child, names, namer)}"
    if isinstance(node, Always):
        return f"throughout [{node.a},{node.b}], {humanize_g(node.child, names, namer)}"
    if isinstance(node, Until):
        return (f"({humanize_g(node.left, names, namer)}) until "
                f"({humanize_g(node.right, names, namer)}) within [{node.a},{node.b}]")
    raise TypeError(type(node).__name__)


def select_g(nst, instances, exclude=frozenset()):
    """Pick a faithful, Eventually-rooted, discriminative, correct YES instance whose
    supporting channel is not excluded (e.g. operating-setting columns)."""
    from utils.faithfulness import governing_leaf
    from executor.grammar import Eventually, Not
    best, bs = None, -1e9
    for inst in instances:
        if inst.depth < 2 or inst.mu_star is None:
            continue
        mu = nst.perceive(inst.X)
        rho = float(nst.answer(inst.X, inst.phi_star)["rho"])
        if (rho > 0) != bool(inst.answer_star) or rho <= 0:
            continue
        hl, ht = governing_leaf(mu, inst.phi_star, nst.pidx, 0)
        gl, _ = governing_leaf(inst.mu_star, inst.phi_star, nst.pidx, 0)
        if hl is None or gl is None or hl.channel in exclude:
            continue
        col = nst.pidx[(hl.name, hl.channel)]
        disc = float(mu[:, col].std())
        if disc < 0.08 or float(mu[ht, col]) < 0.5:
            continue
        faithful = (hl.name, hl.channel) == (gl.name, gl.channel)
        root = inst.phi_star
        while isinstance(root, Not):
            root = root.child
        score = 2 * faithful + 2 * isinstance(root, Eventually) + 1.5 * disc + (inst.depth == 2) + min(abs(rho), 1.0)
        if score > bs:
            best, bs = (inst, mu, hl, ht, rho), score
    return best


def case_study_figure(spec, quick):
    import torch
    from benchmark.realdata import build_real_benchmark
    from models.nstqa_learned import LearnedNSTQA
    from perception.learned import train_perception

    print(f"  case study [{spec['key']}]: building + training ...", flush=True)
    adapter = spec["make"]()
    bm = build_real_benchmark(
        adapter, T=spec["T"], stride=spec["stride"], depths=(1, 2, 3, 4),
        shift="condition", train_conditions=spec["train"], test_conditions=spec["test"],
        indist_holdout_frac=0.3, n_train_per_depth=200, n_test_per_depth=120,
        hi_q=0.85, lo_q=0.15, smooth_k=5, a_level=4.0, allow_until=False,
        max_windows_per_unit=200, over_factor=120, seed=0)
    C = bm["meta"]["n_channels"]; names = bm["meta"]["channel_names"]
    if not bm["test_shift"]:
        print(f"  (skip {spec['key']}: empty shift pool)"); return
    epochs = 12 if quick else 80
    pres = train_perception(bm["train"], n_channels=C, hidden=32, kernel=5, n_layers=2,
                            per_channel=True, epochs=epochs, batch_size=64, lr=1e-3,
                            weight_decay=0.0, device_pref="cpu", seed=0, verbose=False)
    nst = LearnedNSTQA(pres.model, n_channels=C)
    exclude = {i for i, n in enumerate(names) if spec.get("exclude_prefix") and n.startswith(spec["exclude_prefix"])}
    picked = select_g(nst, bm["test_shift"], exclude=exclude)
    if picked is None:
        print(f"  (no suitable instance for {spec['key']})"); return
    inst, mu_hat, leaf, t_star, rho = picked
    namer = spec["namer"]; T = inst.X.shape[0]; xlabel = spec["xlabel"]
    lo, hi = max(0, t_star - 2), min(T - 1, t_star + 2)
    col = nst.pidx[(leaf.name, leaf.channel)]
    support = f"{namer(names[leaf.channel])} {_FAMILY[leaf.name]}"
    sig = inst.X[:, leaf.channel].numpy()
    truth = mu_hat[:, col].numpy(); mu_star_truth = inst.mu_star[:, col].numpy()
    # bottom panel: per-timestep robustness of the SUPPORTING predicate (leaf rule
    # 2*mu-1). Unlike the whole-program robustness rho(phi,t) -- which, for an
    # eventually with a large offset, is only defined for early anchors and does not
    # align with the critical interval -- this spans the full window and crosses zero
    # / peaks inside tau, directly showing where the cited evidence fires.
    leaf_rho = 2.0 * truth - 1.0; x = np.arange(T)

    fig = plt.figure(figsize=(9.2, 6.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.2, 1.2], hspace=0.34)
    fig.text(0.02, 0.995, spec["label"], fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.02, 0.96, f"Q: “Is it true that {humanize_g(inst.phi_star, names, namer)}?”",
             fontsize=11, ha="left", va="top")
    fig.text(0.02, 0.93, f"unit {inst.unit_id} ({spec['cond_word']} {inst.condition}, unseen at train) "
             f"  •  program {inst.phi_star.canonical()}  (depth {inst.depth})",
             fontsize=8.3, color="0.4", ha="left", va="top")

    ax0 = fig.add_subplot(gs[0])
    ax0.axvspan(lo - .5, hi + .5, color=EXEC, alpha=.18, label="critical interval $\\tau$")
    ax0.axvline(t_star, color=EXEC, ls="--", lw=1.4)
    ax0.plot(x, sig, color=DATA, marker=".", ms=4, lw=1.6)
    ax0.set_ylabel(f"{namer(names[leaf.channel])}\n(normalized)")
    ans = "YES" if inst.answer_star else "NO"
    ax0.annotate(f"answer: {ans}  ($\\rho={rho:+.2f}$)\nbecause {support}\nover {xlabel}s [{lo},{hi}]",
                 xy=(t_star, sig[t_star]), xytext=(0.62, 0.16), textcoords="axes fraction",
                 fontsize=9.5, bbox=dict(boxstyle="round", fc="#FFF6E6", ec=EXEC, alpha=.95),
                 arrowprops=dict(arrowstyle="->", color=EXEC))
    ax0.legend(loc="upper left", fontsize=8)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.axvspan(lo - .5, hi + .5, color=EXEC, alpha=.18)
    ax1.plot(x, truth, color=PERC, marker=".", ms=3, lw=1.6, label="learned $\\hat\\mu$")
    ax1.plot(x, mu_star_truth, color="0.6", ls="--", lw=1.2, label="privileged $\\mu^\\star$")
    ax1.axhline(.5, color="k", ls=":", lw=.8, alpha=.6); ax1.set_ylim(-.05, 1.05)
    ax1.set_ylabel(f"perceived\n{leaf.canonical()}"); ax1.legend(fontsize=7.5, ncol=2, loc="lower left")

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.axvspan(lo - .5, hi + .5, color=EXEC, alpha=.18)
    ax2.plot(x, leaf_rho, color=SHORT, marker=".", ms=3, lw=1.6)
    ax2.axvline(t_star, color=EXEC, ls="--", lw=1.2, alpha=.7)
    ax2.axhline(0, color="k", ls=":", lw=.8, alpha=.6)
    ax2.set_ylabel("supporting-predicate\nrobustness $2\\hat\\mu-1$")
    ax2.set_xlabel(f"{xlabel} index within window")
    _save(fig, spec["out"])


def _specs():
    from benchmark.adapters_cmapss import CMAPSSAdapter
    from benchmark.adapters_femto import FEMTOAdapter
    from benchmark.adapters_ims import IMSAdapter
    from benchmark.adapters_xjtu import XJTUAdapter
    return {
        "cmapss": dict(key="cmapss", label="C-MAPSS (turbofan engine sensors)", out="fig_case_study_cmapss.png",
                       make=lambda: CMAPSSAdapter(root=ROOT / "data/raw/CMAPSS",
                                                  subsets=("FD001", "FD002", "FD004"), rul_cap=125,
                                                  flat_std_thresh=1e-6, min_qspan=0.05, hi_q=0.85, lo_q=0.15),
                       train=("FD001",), test=("FD002", "FD004"), namer=cmapss_namer,
                       T=48, stride=4, xlabel="cycle", cond_word="subset", exclude_prefix="op"),
        "xjtu": dict(key="xjtu", label="XJTU-SY (run-to-failure bearing vibration)", out="fig_case_study.png",
                     make=lambda: XJTUAdapter(), train=("35Hz12kN",), test=("37.5Hz11kN", "40Hz10kN"),
                     namer=vibration_namer, T=32, stride=2, xlabel="snapshot", cond_word="load", exclude_prefix=None),
        "femto": dict(key="femto", label="FEMTO/PRONOSTIA (run-to-failure bearing vibration)",
                      out="fig_case_study_femto.png", make=lambda: FEMTOAdapter(),
                      train=("1",), test=("2", "3"), namer=vibration_namer, T=32, stride=2,
                      xlabel="snapshot", cond_word="condition", exclude_prefix=None),
        "ims": dict(key="ims", label="NASA IMS (run-to-failure bearing vibration)",
                    out="fig_case_study_ims.png", make=lambda: IMSAdapter(),
                    train=("ims1",), test=("ims2", "ims3"), namer=vibration_namer, T=32, stride=2,
                    xlabel="snapshot", cond_word="rig", exclude_prefix=None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-case-study", action="store_true")
    ap.add_argument("--datasets", default="cmapss,xjtu,femto,ims",
                    help="comma list of case-study datasets to render")
    args = ap.parse_args()
    print("generating result figures ...")
    fig_necessity_bar(); fig_depth_curve(); fig_generalization(); fig_crossrig()
    if not args.no_case_study:
        specs = _specs()
        for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
            if ds in specs:
                try:
                    case_study_figure(specs[ds], args.quick)
                except Exception as e:
                    print(f"  (case study {ds} failed: {e})")
            else:
                print(f"  (unknown dataset {ds})")
    print(f"done -> {FIGDIR}")


if __name__ == "__main__":
    main()
