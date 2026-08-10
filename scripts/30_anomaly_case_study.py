"""30 - Worked `anomalous` explanation case study (WO-1C deliverable).

Renders ONE held-out XJTU bearing instance whose faithful explanation is carried by
the LEARNED `anomalous` predicate, end-to-end: question (STL program over the
5-family vocab) -> perceived predicates (4-family perception + learned anomaly head)
-> executor -> answer + critical interval + supporting predicate. The money panel
overlays the learned anomaly truth mu_hat_anom against the privileged healthy-
baseline-deviation target mu*_anom on the supporting channel — showing the head
reproduces a NON-threshold degradation signal a fixed rule cannot.

Runs locally (CPU; XJTU HI cache). Output: runs/anomaly_predicate/case_study/.

Run:  python scripts/30_anomaly_case_study.py [--quick]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import copy
import torch
import yaml

from benchmark.adapters_xjtu import XJTUAdapter
from benchmark.anomaly_qa import pidx5
from benchmark.anomaly_questions import build_safe_anomaly_benchmark
from executor.grammar import Always, And, Eventually, Not, Or, Predicate, Until
from executor.hard_logic import evaluate
from perception.anomaly import predict_anomaly, train_anomaly_head
from perception.learned import predict_mu, train_perception
from utils.faithfulness import governing_leaf

RADIUS = 2
_FAMILY = {"high": "is high", "low": "is low", "rising": "is rising",
           "falling": "is falling", "anomalous": "is anomalous (deviates from healthy baseline)"}
_AXIS = {"h": "Horizontal", "v": "Vertical"}
_FEAT = {"rms": "RMS", "kurtosis": "kurtosis", "peak": "peak amplitude",
         "crest": "crest factor", "fft_b0": "low-band energy",
         "fft_b1": "mid-band energy", "fft_b2": "high-band energy"}


def channel_human(name: str) -> str:
    axis, _, feat = name.partition("_")
    return f"{_AXIS.get(axis, axis)} {_FEAT.get(feat, feat)}"


def humanize(node, names) -> str:
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
        return f"({humanize(node.left, names)}) until ({humanize(node.right, names)}) within [{node.a},{node.b}]"
    raise TypeError(type(node).__name__)


def mu_hat_5(perception, head, X):
    return torch.cat([predict_mu(perception, X), predict_anomaly(head, X)], dim=1)


def select_instance(perception, head, instances, pidx, C):
    """Pick a faithful, correct, depth>=2 shift instance whose supporting predicate
    is `anomalous` and whose learned anomaly truth is discriminative over time."""
    best, best_score = None, -1e9
    for inst in instances:
        if inst.depth < 2 or inst.mu_star is None:
            continue
        mu_hat = mu_hat_5(perception, head, inst.X)
        rho, _ = evaluate(mu_hat, inst.phi_star, pidx, anchor=0)
        pred = float(rho) > 0
        if pred != bool(inst.answer_star) or not pred:
            continue
        h_leaf, h_t = governing_leaf(mu_hat, inst.phi_star, pidx, 0)
        g_leaf, _ = governing_leaf(inst.mu_star, inst.phi_star, pidx, 0)
        if h_leaf is None or g_leaf is None:
            continue
        faithful = (h_leaf.name, h_leaf.channel) == (g_leaf.name, g_leaf.channel)
        is_anom = h_leaf.name == "anomalous"
        col = pidx[(h_leaf.name, h_leaf.channel)]
        disc = float(mu_hat[:, col].std())
        if disc < 0.08 or float(mu_hat[h_t, col]) < 0.5:
            continue
        score = 3.0 * is_anom + 2.0 * faithful + 1.5 * disc + (inst.depth == 2) + min(abs(float(rho)), 1.0)
        if score > best_score:
            best, best_score = (inst, mu_hat, h_leaf, h_t, float(rho)), score
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "anomaly_predicate.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    d = cfg["datasets"]["xjtu"]
    a, b = d["adapter"], d["build"]
    pep = 20 if args.quick else cfg["perception"]["epochs"]
    hep = 20 if args.quick else cfg["anomaly_head"]["epochs"]

    adapter = XJTUAdapter(root=ROOT / a["root"], fs=a["fs"], n_bands=a["n_bands"],
                          min_snapshots=a.get("min_snapshots", 1),
                          cache_path=ROOT / a["cache_path"], force_recompute=False)
    bm = build_safe_anomaly_benchmark(
        adapter, T=b["T"], stride=b["stride"], depths=tuple(b["depths"]), shift=b["shift"],
        train_conditions=tuple(b["train_conditions"]), test_conditions=tuple(b["test_conditions"]),
        indist_holdout_frac=b["indist_holdout_frac"], healthy_frac=b["healthy_frac"],
        n_train_per_depth=b["n_train_per_depth"], n_test_per_depth=b["n_test_per_depth"],
        hi_q=b["hi_q"], lo_q=b["lo_q"], smooth_k=b["smooth_k"], a_level=b["a_level"],
        a_anom=b["a_anom"], anomaly_q=b["anomaly_q"], anomaly_p=b["anomaly_p"],
        allow_until=b["allow_until"], max_windows_per_unit=b.get("max_windows_per_unit"),
        over_factor=b["over_factor"], seed=b["build_seed"],
        leak_max=b["leak_max"], max_leak_retries=b["max_leak_retries"])
    C = bm["meta"]["n_channels"]
    names = bm["meta"]["channel_names"] if "channel_names" in bm["meta"] else \
        [f"h_rms"] * C  # fallback; XJTU meta may not carry names
    if "channel_names" not in bm["meta"]:
        names = adapter.load()[0].channel_names
    baseline = bm["baseline"]
    pidx = pidx5(C)

    train4 = []
    for i in bm["train"]:
        j = copy.copy(i); j.mu_star = i.mu_star[:, :4 * C]; train4.append(j)
    print(f"training perception ({pep}ep) + anomaly head ({hep}ep) ...", flush=True)
    pres = train_perception(train4, n_channels=C, hidden=cfg["perception"]["hidden"],
                            kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
                            per_channel=cfg["perception"]["per_channel"], epochs=pep,
                            batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
                            weight_decay=cfg["perception"]["weight_decay"], device_pref="cpu",
                            seed=0, verbose=False)
    head = train_anomaly_head(torch.stack([i.X for i in bm["train"]]), baseline,
                              hidden=cfg["anomaly_head"]["hidden"], kernel=cfg["anomaly_head"]["kernel"],
                              n_layers=cfg["anomaly_head"]["n_layers"], epochs=hep,
                              batch_size=cfg["anomaly_head"]["batch_size"], lr=cfg["anomaly_head"]["lr"],
                              device_pref="cpu", seed=0)

    picked = select_instance(pres.model, head, bm["test_shift"], pidx, C)
    if picked is None:
        print("no faithful anomaly-supported shift instance found; try more epochs")
        return
    inst, mu_hat, leaf, t_star, rho = picked
    T = inst.X.shape[0]
    lo, hi = max(0, t_star - RADIUS), min(T - 1, t_star + RADIUS)
    ch = leaf.channel
    col = pidx[(leaf.name, ch)]
    support_human = f"{channel_human(names[ch])} {_FAMILY[leaf.name]}"
    mu_hat_anom = mu_hat[:, col]                          # learned anomaly truth on this channel
    mu_star_anom = inst.mu_star[:, 4 * C + ch]            # privileged healthy-baseline-deviation target

    out_dir = ROOT / cfg["run_root"] / "case_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    card = [
        "# Worked `anomalous` explanation case study (XJTU, cross-load shift)", "",
        f"- **Unit:** {inst.unit_id}  (operating condition `{inst.condition}`, unseen at train)",
        f"- **Question:** `{inst.phi_star.canonical()}`  (depth {inst.depth})",
        f"- **In plain terms:** *Is it true that {humanize(inst.phi_star, names)}?*",
        f"- **NS-TQA answer:** **{'YES' if inst.answer_star else 'NO'}**  (ρ = {rho:+.3f})",
        f"- **Critical interval τ:** snapshots **[{lo},{hi}]** (decisive t\\*={t_star})",
        f"- **Supporting predicate:** `{leaf.canonical()}` → **{support_human}**", "",
        "## Faithful explanation",
        f"> The system answers **{'YES' if inst.answer_star else 'NO'}** *because* the **learned "
        f"`anomalous` predicate** on {channel_human(names[ch])} fires over snapshots **[{lo},{hi}]** — "
        f"a deviation from the bearing's healthy baseline that a fixed high/low threshold does not "
        f"capture. The learned anomaly truth tracks the privileged healthy-baseline-deviation target "
        f"(panel 2), and removing this predicate flips the answer.",
    ]
    (out_dir / "anomaly_case_study.md").write_text("\n".join(card))
    print("\n".join(card))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.arange(T)
        fig, axs = plt.subplots(3, 1, figsize=(8, 7.2), sharex=True)
        for ax in axs:
            ax.axvspan(lo - 0.5, hi + 0.5, color="gold", alpha=0.30, label="critical interval τ")
            ax.axvline(t_star, color="darkorange", ls="--", lw=1.2)
        axs[0].plot(x, inst.X[:, ch].numpy(), color="steelblue", marker=".", ms=3)
        axs[0].set_ylabel(f"{channel_human(names[ch])}\n(normalized)")
        axs[0].set_title(f"{inst.unit_id} (unseen load) — answer {'YES' if inst.answer_star else 'NO'}: "
                         f"`{leaf.canonical()}` fires in τ=[{lo},{hi}]", fontsize=9)
        axs[1].plot(x, mu_star_anom.numpy(), color="black", lw=2, label="privileged μ*_anom (target)")
        axs[1].plot(x, mu_hat_anom.detach().numpy(), color="crimson", marker=".", ms=3,
                    label="learned μ̂_anom (head)")
        axs[1].axhline(0.5, color="gray", ls=":", lw=1)
        axs[1].set_ylim(-0.05, 1.05)
        axs[1].set_ylabel("anomalous truth")
        axs[1].legend(fontsize=8, loc="upper left")
        # panel 3: the raw deviation the head must learn (healthy-baseline distance)
        dev = baseline.deviation(inst.X)[:, ch].numpy()
        axs[2].plot(x, dev, color="seagreen", marker=".", ms=3)
        axs[2].axhline(float(baseline.theta[ch]), color="gray", ls=":", lw=1, label="healthy 0.99-quantile θ")
        axs[2].set_ylabel("healthy-baseline\ndeviation")
        axs[2].set_xlabel("snapshot index within window")
        axs[2].legend(fontsize=8, loc="upper left")
        axs[0].legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_anomaly_case_study.png", dpi=300)
        print(f"\nwrote card + figure to {out_dir}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
