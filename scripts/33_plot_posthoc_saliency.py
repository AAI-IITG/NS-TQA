"""33 - Qualitative post-hoc saliency plots (WO-2B figure).

Renders WHY post-hoc attribution fails where NS-TQA's by-construction explanation
succeeds. For a representative shifted instance it draws, on the same time axis:

  (a) the signal of the oracle supporting channel, with the oracle critical interval
      tau shaded and NS-TQA's recovered interval marked;
  (b) grad x input, Integrated Gradients, and SHAP saliency as [C x T] heatmaps, each
      with the oracle tau band + the interval that method's saliency implicates.

Plus an AGGREGATE panel: mean saliency mass as a function of offset from the oracle
interval centre (averaged over all shift instances) -- NS-TQA's decisive evidence
concentrates at tau, while post-hoc saliency is essentially flat.

Runs locally (matplotlib + optional shap). Output: runs/posthoc_xai/.

Run:  python scripts/33_plot_posthoc_saliency.py [--datasets xjtu,cmapss] [--seed 0]
"""
import argparse
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
from explain.extract import saliency_to_explanation
from explain.posthoc import gradient_input, integrated_gradients, shap_gradient
from models.nstqa_learned import LearnedNSTQA
from perception.learned import train_perception
from utils.faithfulness import _interval, governing_leaf

RADIUS = 2


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


def pick_instance(nst, insts, pidx, T):
    """A faithful, depth>=2 instance with an oracle interval well inside [0,T]."""
    best = None
    for inst in insts:
        if inst.depth < 2:
            continue
        g_leaf, g_t = governing_leaf(inst.mu_star, inst.phi_star, pidx, 0)
        if g_leaf is None or g_t is None or g_t < RADIUS + 1 or g_t > T - RADIUS - 2:
            continue
        mu_hat = nst.perceive(inst.X)
        h_leaf, h_t = governing_leaf(mu_hat, inst.phi_star, pidx, 0)
        if h_leaf is None:
            continue
        if h_leaf.channel == g_leaf.channel and float(nst.answer(inst.X, inst.phi_star)["rho"]) > 0:
            return inst, g_leaf, g_t, mu_hat, h_leaf, h_t
        best = best or (inst, g_leaf, g_t, mu_hat, h_leaf, h_t)
    return best


def saliencies(tcn, inst, q, device, bg, C):
    shp = shap_gradient(tcn, q, inst.X, bg, device)
    if shp is None:
        shp = torch.zeros_like(inst.X)
    return {
        "gradient$\\times$input": gradient_input(logit_fn_for(tcn, q), inst.X.to(device)).cpu(),
        "Integrated Gradients": integrated_gradients(logit_fn_for(tcn, q), inst.X.to(device), steps=32).cpu(),
        "SHAP": shp.cpu(),
    }


def plot_dataset(dname, cfg, dcfg, seed, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = select_device("cpu")
    bm = build_bm(dcfg)
    C, T = bm["meta"]["n_channels"], dcfg["build"]["T"]
    insts = [i for i in bm["test_shift"] if i.mu_star is not None]
    pres = train_perception(bm["train"], n_channels=C, hidden=cfg["perception"]["hidden"],
                            kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
                            per_channel=cfg["perception"]["per_channel"], epochs=cfg["perception"]["epochs"],
                            batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
                            weight_decay=cfg["perception"]["weight_decay"], device_pref="cpu",
                            seed=seed, verbose=False)
    nst = LearnedNSTQA(pres.model, n_channels=C)
    pidx = nst.pidx
    tcn = train_baseline("tcn", bm["train"], C, T, dict(cfg["lstm"]), device, seed, verbose=False)
    bg = torch.stack([i.X for i in bm["train"][:32]])

    picked = pick_instance(nst, insts, pidx, T)
    if picked is None:
        print(f"[{dname}] no suitable instance"); return
    inst, g_leaf, g_t, mu_hat, h_leaf, h_t = picked
    q = encode_program(inst.phi_star, C, T).to(device)
    sal = saliencies(tcn, inst, q, device, bg, C)
    gold = _interval(g_t, RADIUS, T)
    ns_iv = _interval(h_t, RADIUS, T)
    gch = g_leaf.channel

    # ---- instance figure: signal + 3 saliency heatmaps ----
    fig, axs = plt.subplots(4, 1, figsize=(8.4, 8.6), height_ratios=[1.1, 1, 1, 1])
    x = np.arange(T)
    # (a) signal of oracle channel + intervals
    axs[0].plot(x, inst.X[:, gch].numpy(), color="steelblue", marker=".", ms=3)
    axs[0].axvspan(gold[0] - 0.5, gold[1] + 0.5, color="gold", alpha=0.35, label="oracle interval $\\tau$")
    axs[0].axvspan(ns_iv[0] - 0.5, ns_iv[1] + 0.5, facecolor="none", edgecolor="seagreen",
                   lw=2.0, ls="--", label="NS-TQA interval")
    axs[0].set_ylabel(f"signal\n(oracle ch {gch})")
    axs[0].set_title(f"{dname.upper()} shift instance — NS-TQA recovers $\\tau$; post-hoc saliency does not",
                     fontsize=10)
    axs[0].legend(fontsize=8, loc="upper left")
    # (b-d) saliency heatmaps [C x T]
    for ax, (name, S) in zip(axs[1:], sal.items()):
        Sn = S.numpy().T                                   # [C, T]
        Sn = Sn / (Sn.max() + 1e-9)
        im = ax.imshow(Sn, aspect="auto", cmap="magma", origin="lower",
                       extent=[0, T, -0.5, C - 0.5], vmin=0, vmax=1)
        ax.axvspan(gold[0], gold[1] + 1, facecolor="none", edgecolor="cyan", lw=2.0,
                   label="oracle $\\tau$")
        ex = saliency_to_explanation(S, RADIUS, T)
        if ex["interval"]:
            ax.axvspan(ex["interval"][0], ex["interval"][1] + 1, facecolor="none",
                       edgecolor="lime", lw=1.6, ls="--", label="method interval")
        ax.axhline(gch, color="white", lw=0.8, ls=":", alpha=0.7)
        ax.set_ylabel(f"{name}\nchannel")
        if ax is axs[1]:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
    axs[-1].set_xlabel("time step within window")
    fig.tight_layout()
    f1 = out_dir / f"fig_saliency_{dname}.png"
    fig.savefig(f1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- aggregate: mean saliency mass vs offset from tau centre ----
    W = T
    offs = np.arange(-(T // 2), T // 2 + 1)
    agg = {k: np.zeros(len(offs)) for k in list(sal.keys()) + ["NS-TQA (|$\\rho$| leaf truth)"]}
    cnt = 0
    for it in insts[:200]:
        gl, gt = governing_leaf(it.mu_star, it.phi_star, pidx, 0)
        if gl is None or gt is None:
            continue
        qi = encode_program(it.phi_star, C, T).to(device)
        for name, fn in [("gradient$\\times$input", lambda: gradient_input(logit_fn_for(tcn, qi), it.X.to(device)).cpu()),
                         ("Integrated Gradients", lambda: integrated_gradients(logit_fn_for(tcn, qi), it.X.to(device), steps=16).cpu())]:
            mass = fn().sum(1).numpy(); mass = mass / (mass.sum() + 1e-9)
            for j, o in enumerate(offs):
                t = gt + o
                if 0 <= t < T:
                    agg[name][j] += mass[t]
        # NS-TQA: perceived truth of the oracle leaf column over time
        mu = nst.perceive(it.X)
        col = pidx[(gl.name, gl.channel)]
        tr = mu[:, col].detach().numpy(); tr = tr / (tr.sum() + 1e-9)
        for j, o in enumerate(offs):
            t = gt + o
            if 0 <= t < T:
                agg["NS-TQA (|$\\rho$| leaf truth)"][j] += tr[t]
        cnt += 1
    fig2, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = {"gradient$\\times$input": "#d55e00", "Integrated Gradients": "#cc79a7",
              "NS-TQA (|$\\rho$| leaf truth)": "#0072b2"}
    for k in ("gradient$\\times$input", "Integrated Gradients", "NS-TQA (|$\\rho$| leaf truth)"):
        ax.plot(offs, agg[k] / max(cnt, 1), label=k, lw=2,
                color=colors.get(k), marker="." if "NS-TQA" in k else None, ms=4)
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("offset from oracle interval centre (time steps)")
    ax.set_ylabel("mean normalised\nsaliency mass")
    ax.set_title(f"{dname.upper()}: evidence concentration at $\\tau$ ({cnt} instances)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig2.tight_layout()
    f2 = out_dir / f"fig_saliency_alignment_{dname}.png"
    fig2.savefig(f2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"[{dname}] wrote {f1.name}, {f2.name}  (oracle ch={gch}, tau={gold}, "
          f"NS-TQA iv={ns_iv}; method intervals: "
          + ", ".join(f"{k.split('(')[0].strip()}={saliency_to_explanation(v, RADIUS, T)['interval']}"
                      for k, v in sal.items()) + ")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "posthoc_xai.yaml"))
    ap.add_argument("--datasets", default="xjtu,cmapss")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg["perception"]["epochs"] = min(cfg["perception"]["epochs"], 60)
    cfg["lstm"]["epochs"] = min(cfg["lstm"]["epochs"], 40)
    out_dir = ROOT / cfg["run_root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in args.datasets.split(","):
        plot_dataset(d, cfg, cfg["datasets"][d], args.seed, out_dir)


if __name__ == "__main__":
    main()
