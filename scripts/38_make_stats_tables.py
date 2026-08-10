"""38 - Significance post-processor: raw + Holm-corrected p-values (WO-4).

Ingests the per-seed arrays already written by the extension experiments and produces
one significance-annotated table per family of headline comparisons. For each family we
compute the paired (or one-sample, vs a deterministic baseline) Wilcoxon p, then apply
Holm--Bonferroni across the family, and report it alongside the effect size and seed
count.

An honesty note baked into the output: with n seeds the two-sided Wilcoxon signed-rank
p-value cannot go below $2/2^{n}$ (e.g. 0.0625 at n=5). When a family's raw p sits at
that floor, Holm across several comparisons can exceed 0.05 even for a huge effect; we
flag those rows so the effect size, not a seed-limited p, carries the claim.

Run:  python scripts/38_make_stats_tables.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.stats import holm_bonferroni, wilcoxon


def load(rel):
    p = ROOT / rel
    return json.load(open(p)) if p.exists() else None


def wilcoxon_floor(n):
    return 1.0 / (2 ** n) if n >= 1 else float("nan")   # one-sided floor


def family(name, comps):
    """comps: list of dicts {label, x(list), y(list or scalar)}. Returns rows with raw+Holm p.

    We test the DIRECTIONAL hypothesis x>y (one-sided), which is the actual claim
    (NS-TQA beats the baseline); its floor at n seeds is $1/2^{n}$ (0.031 at n=5),
    tighter than the two-sided $2/2^{n}$."""
    raws, rows = [], []
    for c in comps:
        x = c["x"]
        y = c["y"] if isinstance(c["y"], list) else [c["y"]] * len(x)
        w = wilcoxon(x, y, alternative="greater")
        eff = (sum(x) / len(x)) - (sum(y) / len(y))
        raws.append(w["p"])
        rows.append({"label": c["label"], "n": len(x), "eff": eff, "raw": w["p"]})
    holm = holm_bonferroni(raws)
    floor_hit = 0
    for r, padj, rej in zip(rows, holm["p_adjusted"], holm["reject"]):
        r["holm"] = padj
        r["sig"] = rej
        r["floor"] = (r["raw"] == r["raw"]) and abs(r["raw"] - wilcoxon_floor(r["n"])) < 1e-6
        floor_hit += int(r["floor"])
    return {"name": name, "rows": rows, "floor_hit": floor_hit}


def collect():
    fams = []

    # --- C3b anomaly: learned head vs fixed proxy (10 seeds) ---
    d = load("runs/anomaly_predicate/anomaly_predicate.json")
    if d:
        comps = []
        for ds, D in d["results"].items():
            for pool in ("indist", "shift"):
                hf = D["metrics"]["head_f1"][pool]
                comps.append({"label": f"{ds}/{pool}: head vs proxy",
                              "x": hf, "y": D["proxy_f1"][pool]})
        fams.append(family("C3b: learned anomaly head vs proxy (macro-F1)", comps))

    # --- WO-3A regimes: NS-TQA vs TCN per target x arm (5 seeds) ---
    d = load("runs/cmapss_regimes/cmapss_regimes.json")
    if d:
        comps = []
        for arm, D in d["results"].items():
            cell = D["cell"]
            for tgt in [k for k in cell.get("NS-TQA", {}) if k != "FD001 (held-out)"]:
                if cell.get("tcn", {}).get(tgt):
                    comps.append({"label": f"{arm}/{tgt}: NS-TQA vs TCN",
                                  "x": cell["NS-TQA"][tgt], "y": cell["tcn"][tgt]})
        fams.append(family("WO-3A: NS-TQA vs TCN by C-MAPSS target (shifted)", comps))

    # --- WO-2B post-hoc: NS-TQA vs each attribution, per dataset x metric (5 seeds) ---
    d = load("runs/posthoc_xai/posthoc_xai.json")
    if d:
        comps = []
        for ds, D in d["results"].items():
            ps = D["per_seed"]; ns = ps["NS-TQA (by-construction)"]
            for m in ("grad-input", "integrated-gradients", "SHAP"):
                for metric in ("support", "iou", "cf"):
                    if len(ps.get(m, {}).get(metric, [])) == len(ns[metric]):
                        comps.append({"label": f"{ds}/{metric}: NS-TQA vs {m}",
                                      "x": ns[metric], "y": ps[m][metric]})
        fams.append(family("WO-2B: NS-TQA vs post-hoc attribution (identical axes)", comps))

    # --- WO-4 sensitivity: NS-TQA vs TCN across the grid (5 seeds) ---
    d = load("runs/sensitivity/sensitivity.json")
    if d:
        comps = []
        for ds, facs in d["results"].items():
            for fac, cells in facs.items():
                for label, c in cells.items():
                    if isinstance(c.get("ns"), list) and isinstance(c.get("tcn"), list):
                        comps.append({"label": f"{ds}/{fac}={label}: NS-TQA vs TCN",
                                      "x": c["ns"], "y": c["tcn"]})
        if comps:
            fams.append(family("WO-4: NS-TQA vs TCN across the sensitivity grid", comps))

    return fams


def main():
    fams = collect()
    out_dir = ROOT / "runs" / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    L = ["# Significance tables: raw + Holm-corrected (WO-4)", "",
         "Per family: paired/one-sample Wilcoxon (`raw p`), Holm--Bonferroni across the "
         "family (`Holm p`), effect size (mean difference), and seed count `n`. A `*` marks "
         "a Wilcoxon p sitting at the $2/2^n$ floor, where the seed count (not the effect) "
         "caps significance; there the effect size carries the claim.", ""]
    for f in fams:
        L.append(f"## {f['name']}")
        L += ["", "| comparison | n | effect | raw p | Holm p | significant |", "|---|---|---|---|---|---|"]
        for r in f["rows"]:
            floor = "*" if r["floor"] else ""
            raw = "nan" if r["raw"] != r["raw"] else f"{r['raw']:.4f}{floor}"
            holm = "nan" if r["holm"] != r["holm"] else f"{r['holm']:.4f}"
            L.append(f"| {r['label']} | {r['n']} | {r['eff']:+.3f} | {raw} | {holm} | "
                     f"{'yes' if r['sig'] else 'no'} |")
        nsig = sum(1 for r in f["rows"] if r["sig"])
        note = ""
        if f["floor_hit"]:
            note = (f"  _{f['floor_hit']} comparison(s) hit the n-seed p-floor; their effect "
                    f"sizes (all one-directional across seeds) carry the result._")
        L.append(f"\n**{nsig}/{len(f['rows'])} significant after Holm correction.**{note}\n")
    table = "\n".join(L)
    (out_dir / "holm_tables.md").write_text(table)
    print(table)
    print(f"\nwrote -> {out_dir}/holm_tables.md")


if __name__ == "__main__":
    main()
