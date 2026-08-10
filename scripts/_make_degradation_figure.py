"""Generate degradation curves (accuracy vs severity) from runs/degradation/degradation.json.

Runs LOCALLY (the cluster torch container has no matplotlib). One panel per
(dataset, regime, family); lines = methods; y = mean accuracy, shaded ±std over
seeds (STL-only is deterministic, no band).

Usage:  python scripts/_make_degradation_figure.py [runs/degradation/degradation.json]
"""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# colourblind-friendly, consistent across panels
STYLE = {
    "STL-only":   ("#d55e00", "-",  "o"),   # vermillion
    "NS-TQA":     ("#0072b2", "-",  "s"),   # blue
    "NS-TQA-aug": ("#009e73", "-",  "^"),   # green
    "best-e2e":   ("#666666", "--", "x"),   # grey dashed
}


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    return (sum(xs) / len(xs), st.pstdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)


def best_e2e_series(cell, baselines):
    ms = [mean_std(cell[b])[0] for b in baselines if b in cell]
    ms = [m for m in ms if m == m]
    return max(ms) if ms else float("nan")


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/degradation/degradation.json"
    data = json.load(open(src))
    sweep, seeds = data["sweep"], data["seeds"]
    baselines = [b for b in ("lstm", "transformer", "tcn") if b in data["methods"]]
    families = list(sweep.keys())

    panels = []  # (dataset, regime)
    for dname, D in data["results"].items():
        regimes = sorted({k.split("|")[0] for k in D["res"]})
        for r in regimes:
            panels.append((dname, r, D))

    nrows, ncols = len(panels), len(families)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                             squeeze=False)
    for ri, (dname, regime, D) in enumerate(panels):
        res = D["res"]
        for ci, fam in enumerate(families):
            ax = axes[ri][ci]
            labels = [lv["label"] for lv in sweep[fam]]
            x = list(range(len(labels)))
            methods = ["STL-only", "NS-TQA", "NS-TQA-aug", "best-e2e"]
            for m in methods:
                ys, es = [], []
                for lab in labels:
                    key = f"{regime}|{fam}|{lab}"
                    cell = res.get(key, {})
                    if m == "best-e2e":
                        ys.append(best_e2e_series(cell, baselines)); es.append(0.0)
                    else:
                        mu, sd = mean_std(cell.get(m, []))
                        ys.append(mu); es.append(sd)
                c, ls, mk = STYLE[m]
                ax.plot(x, ys, ls, color=c, marker=mk, ms=5, lw=1.8, label=m)
                if any(e > 0 for e in es):
                    lo = [y - e for y, e in zip(ys, es)]
                    hi = [y + e for y, e in zip(ys, es)]
                    ax.fill_between(x, lo, hi, color=c, alpha=0.15, linewidth=0)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylim(0.45, 1.0)
            ax.grid(True, alpha=0.25)
            if ri == 0:
                ax.set_title(fam, fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{dname} / {regime}\naccuracy", fontsize=9)
            ax.set_xlabel("severity", fontsize=8)
            if ri == 0 and ci == ncols - 1:
                ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.suptitle(f"Degraded-sensing robustness (WO-1B) — {len(seeds)} seeds", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = src.parent / "degradation_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
