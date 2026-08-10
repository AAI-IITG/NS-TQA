"""Generate notebooks/ns_tqa_cross_dataset_explorer.ipynb (correct JSON escaping)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "ns_tqa_cross_dataset_explorer.ipynb"


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


CELLS = []

CELLS.append(md(
"""# NS-TQA — Cross-Dataset Interactive Explorer

Pick a **dataset** and a **question**, and see the faithful answer path end to end:
the STL program in plain English, the **real signal** with its **critical evidence
interval $\\tau$**, the supporting named-sensor predicate, the learned perception
$\\hat\\mu$ vs the privileged grounding $\\mu^\\star$, the robustness $\\rho(t)$, and
the unit's **RUL curve** showing where in its life the question looks.

Datasets: **C-MAPSS** (turbofan sensors), **XJTU-SY**, **FEMTO/PRONOSTIA**, **NASA IMS**
(run-to-failure bearings). Perception is trained on first use per dataset (cached).

> Requires the dataset caches built by the adapters (XJTU/FEMTO/IMS HI pickles, C-MAPSS raw)."""))

CELLS.append(code(
"""# --- setup: find the repo root so the notebook works from any launch directory ---
import sys, os
from pathlib import Path
def _find_root():
    for cand in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (cand / "src").is_dir() and (cand / "data").is_dir():
            return cand
    return Path.cwd().resolve()
ROOT = _find_root()
os.chdir(ROOT)                            # relative data paths (data/raw/...) now resolve
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
print("repo root:", ROOT)
%matplotlib inline
import numpy as np, torch
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

from benchmark.adapters_cmapss import CMAPSSAdapter
from benchmark.adapters_femto import FEMTOAdapter
from benchmark.adapters_ims import IMSAdapter
from benchmark.adapters_xjtu import XJTUAdapter
from benchmark.realdata import build_real_benchmark
from executor.grammar import Always, And, Eventually, Not, Or, Predicate, Until
from models.nstqa_learned import LearnedNSTQA
from perception.learned import train_perception
from utils.faithfulness import governing_leaf
plt.rcParams["savefig.dpi"] = 300        # any saved figure is >=300 dpi
print("imports OK  •  torch", torch.__version__)"""))

CELLS.append(code(
'''# --- datasets, naming, and English rendering of programs ---
_FAMILY = {"high": "is high", "low": "is low", "rising": "is rising", "falling": "is falling"}
_VAX = {"h": "Horizontal", "v": "Vertical"}
_VFE = {"rms": "RMS", "kurtosis": "kurtosis", "peak": "peak amplitude", "crest": "crest factor",
        "fft_b0": "low-band energy", "fft_b1": "mid-band energy", "fft_b2": "high-band energy"}

def vib_namer(n):
    ax, _, fe = n.partition("_"); return f"{_VAX.get(ax, ax)} {_VFE.get(fe, fe)}"
def cmapss_namer(n):
    return ("operating setting " if n.startswith("op") else "sensor ") + n

def humanize(node, names, namer):
    if isinstance(node, Predicate): return f"{namer(names[node.channel])} {_FAMILY[node.name]}"
    if isinstance(node, Not):  return f"not ({humanize(node.child, names, namer)})"
    if isinstance(node, And):  return f"{humanize(node.left, names, namer)} AND {humanize(node.right, names, namer)}"
    if isinstance(node, Or):   return f"{humanize(node.left, names, namer)} OR {humanize(node.right, names, namer)}"
    if isinstance(node, Eventually): return f"at some point in [{node.a},{node.b}], {humanize(node.child, names, namer)}"
    if isinstance(node, Always):     return f"throughout [{node.a},{node.b}], {humanize(node.child, names, namer)}"
    if isinstance(node, Until):
        return f"({humanize(node.left, names, namer)}) until ({humanize(node.right, names, namer)}) in [{node.a},{node.b}]"

SPECS = {
    "cmapss": dict(label="C-MAPSS (turbofan sensors)",
                   make=lambda: CMAPSSAdapter(root="data/raw/CMAPSS", subsets=("FD001","FD002","FD004"),
                                              rul_cap=125, flat_std_thresh=1e-6, min_qspan=0.05, hi_q=0.85, lo_q=0.15),
                   train=("FD001",), test=("FD002","FD004"), namer=cmapss_namer, T=48, stride=4,
                   xlabel="cycle", excl="op"),
    "xjtu":   dict(label="XJTU-SY bearings", make=lambda: XJTUAdapter(), train=("35Hz12kN",),
                   test=("37.5Hz11kN","40Hz10kN"), namer=vib_namer, T=32, stride=2, xlabel="snapshot", excl=None),
    "femto":  dict(label="FEMTO/PRONOSTIA bearings", make=lambda: FEMTOAdapter(), train=("1",), test=("2","3"),
                   namer=vib_namer, T=32, stride=2, xlabel="snapshot", excl=None),
    "ims":    dict(label="NASA IMS bearings", make=lambda: IMSAdapter(), train=("ims1",), test=("ims2","ims3"),
                   namer=vib_namer, T=32, stride=2, xlabel="snapshot", excl=None),
}
print("datasets:", list(SPECS))'''))

CELLS.append(code(
'''# --- build benchmark + train perception (cached per dataset) and curate questions ---
EPOCHS = 35          # raise for sharper perception; lower for speed
_CACHE = {}

def get_state(key):
    if key in _CACHE:
        return _CACHE[key]
    sp = SPECS[key]; adp = sp["make"]()
    bm = build_real_benchmark(adp, T=sp["T"], stride=sp["stride"], depths=(1,2,3,4),
            shift="condition", train_conditions=sp["train"], test_conditions=sp["test"],
            indist_holdout_frac=0.3, n_train_per_depth=180, n_test_per_depth=120,
            hi_q=0.85, lo_q=0.15, smooth_k=5, a_level=4.0, max_windows_per_unit=200,
            over_factor=120, seed=0)
    C = bm["meta"]["n_channels"]; names = bm["meta"]["channel_names"]
    pres = train_perception(bm["train"], n_channels=C, hidden=32, kernel=5, n_layers=2, per_channel=True,
            epochs=EPOCHS, batch_size=64, lr=1e-3, weight_decay=0.0, seed=0, verbose=False)
    nst = LearnedNSTQA(pres.model, n_channels=C)
    st = dict(bm=bm, nst=nst, names=names, C=C, spec=sp,
              series={s.unit_id: s for s in adp.load()})
    _CACHE[key] = st
    return st

def curate(state, n=16):
    nst, names, sp = state["nst"], state["names"], state["spec"]
    excl = {i for i, nm in enumerate(names) if sp["excl"] and nm.startswith(sp["excl"])}
    pool = state["bm"]["test_shift"] + state["bm"]["test_indist"]
    cands = []
    for inst in pool:
        if inst.mu_star is None: continue
        mu = nst.perceive(inst.X)
        hl, ht = governing_leaf(mu, inst.phi_star, nst.pidx, 0)
        if hl is None or hl.channel in excl: continue
        col = nst.pidx[(hl.name, hl.channel)]; disc = float(mu[:, col].std())
        rho = float(nst.answer(inst.X, inst.phi_star)["rho"])
        pred, gold = rho > 0, bool(inst.answer_star)
        gl, _ = governing_leaf(inst.mu_star, inst.phi_star, nst.pidx, 0)
        faithful = gl is not None and (hl.name, hl.channel) == (gl.name, gl.channel)
        score = 2*faithful + 1.6*disc + (pred == gold) + min(abs(rho), 1.0) + (disc > 0.1)
        cands.append(dict(inst=inst, leaf=hl, t=ht, rho=rho, pred=pred, gold=gold,
                          correct=pred == gold, faithful=faithful, disc=disc, score=score,
                          q=humanize(inst.phi_star, names, sp["namer"])))
    cands = [c for c in cands if c["disc"] > 0.06]
    cands.sort(key=lambda c: -c["score"])
    out, seen = [], set()
    for d in (1, 2, 3, 4):
        for ans in (True, False):
            for c in cands:
                if c["inst"].depth == d and c["gold"] == ans and id(c) not in seen:
                    out.append(c); seen.add(id(c)); break
    return (out + [c for c in cands if id(c) not in seen])[:n]
print("helpers ready (EPOCHS=%d)" % EPOCHS)'''))

CELLS.append(code(
r'''# --- the comprehensive explanation figure (signal + critical window + perception + rho + RUL) ---
def plot_explanation(state, c):
    nst, names, sp = state["nst"], state["names"], state["spec"]
    inst, leaf, t_star = c["inst"], c["leaf"], c["t"]
    T = inst.X.shape[0]; lo, hi = max(0, t_star - 2), min(T - 1, t_star + 2)
    col = nst.pidx[(leaf.name, leaf.channel)]
    expl = nst.explain(inst.X, inst.phi_star)
    sig = inst.X[:, leaf.channel].numpy(); truth = nst.perceive(inst.X)[:, col].numpy()
    mus = inst.mu_star[:, col].numpy(); rho_sig = expl["rho_signal"].numpy(); x = np.arange(T)
    support = f"{sp['namer'](names[leaf.channel])} {_FAMILY[leaf.name]}"
    EX, PE, DA, SH = "#C8841E", "#2C7A7B", "#6B46C1", "#B0413E"

    fig = plt.figure(figsize=(9.4, 7.6)); gs = fig.add_gridspec(4, 1, height_ratios=[1.9, 1, 1, 1.35], hspace=.62)
    verdict = "correct" if c["correct"] else "WRONG"
    fig.suptitle(f"[{sp['label']}]  Q: Is it true that {c['q']}?\n"
                 f"NS-TQA answer: {'YES' if inst.answer_star else 'NO'}  (rho={c['rho']:+.2f})   "
                 f"depth {inst.depth}   -   NS-TQA {verdict} (pred {'YES' if c['pred'] else 'NO'} / "
                 f"gold {'YES' if c['gold'] else 'NO'})",
                 fontsize=10.5, x=.02, ha="left", y=1.005)

    a0 = fig.add_subplot(gs[0]); a0.axvspan(lo-.5, hi+.5, color=EX, alpha=.18, label=r"critical interval $\tau$")
    a0.axvline(t_star, color=EX, ls="--"); a0.plot(x, sig, color=DA, marker=".")
    a0.set_ylabel(f"{sp['namer'](names[leaf.channel])}\n(normalized)"); a0.legend(fontsize=8, loc="upper left")
    a0.annotate(f"because {support}\nover {sp['xlabel']}s [{lo},{hi}]", xy=(t_star, sig[t_star]),
                xytext=(.63, .12), textcoords="axes fraction",
                bbox=dict(boxstyle="round", fc="#FFF6E6", ec=EX), arrowprops=dict(arrowstyle="->", color=EX), fontsize=9)

    a1 = fig.add_subplot(gs[1], sharex=a0); a1.axvspan(lo-.5, hi+.5, color=EX, alpha=.18)
    a1.plot(x, truth, color=PE, marker=".", label=r"learned $\hat\mu$")
    a1.plot(x, mus, color="0.6", ls="--", label=r"privileged $\mu^\star$")
    a1.axhline(.5, color="k", ls=":", lw=.8); a1.set_ylim(-.05, 1.05); a1.legend(fontsize=7, ncol=2, loc="lower left")
    a1.set_ylabel(f"perceived\n{leaf.canonical()}")

    a2 = fig.add_subplot(gs[2], sharex=a0); a2.axvspan(lo-.5, hi+.5, color=EX, alpha=.18)
    a2.plot(x, rho_sig, color=SH, marker="."); a2.axhline(0, color="k", ls=":", lw=.8)
    a2.set_ylabel("robustness\n" r"$\rho(t)$"); a2.set_xlabel(f"{sp['xlabel']} index within the question window")

    a3 = fig.add_subplot(gs[3]); s = state["series"].get(inst.unit_id); start = inst.provenance.get("start", 0)
    if s is not None and "rul" in s.meta:
        rul = s.meta["rul"].numpy(); a3.plot(np.arange(len(rul)), rul, color="#444", lw=1.4)
        a3.axvspan(start, start+T, color=PE, alpha=.22, label="question window")
        a3.axvspan(start+lo, start+hi, color=EX, alpha=.5, label=r"critical $\tau$")
        a3.set_ylabel("RUL\n(life left)"); a3.set_xlabel(f"{sp['xlabel']} index over the unit's full run-to-failure life")
        a3.legend(fontsize=7, loc="upper right")
        a3.set_title(f"unit {inst.unit_id}  (condition {inst.condition}, unseen at train) - "
                     f"the RUL curve and where this question looks", fontsize=8.5)
    return fig'''))

CELLS.append(code(
'''# --- interactive UI: pick a dataset and a question ---
ds_dd = widgets.Dropdown(options=[(SPECS[k]["label"], k) for k in SPECS], value="xjtu", description="Dataset:")
q_dd = widgets.Dropdown(options=[], description="Question:", layout=widgets.Layout(width="92%"),
                        style={"description_width": "70px"})
status = widgets.HTML()
out = widgets.Output()
_S = {}

def _show(idx):
    with out:
        clear_output(wait=True)
        if not _S.get("qs") or idx is None:
            print("no questions loaded"); return
        fig = plot_explanation(_S["st"], _S["qs"][idx])
        display(fig); plt.close(fig)

def _load(key):
    import traceback
    status.value = f"<b style='color:#C8841E'>Building + training perception on {SPECS[key]['label']} … (first time per dataset, ~20–40s)</b>"
    out.clear_output()
    try:
        st = get_state(key); qs = curate(st, 16)
        _S["st"], _S["qs"] = st, qs
        if not qs:
            status.value = f"<b style='color:#B0413E'>{SPECS[key]['label']}: no curated questions (try raising EPOCHS).</b>"
            return
        acc = float(np.mean([q["correct"] for q in qs]))
        q_dd.options = [(f"[d{q['inst'].depth} · {'YES' if q['gold'] else 'NO'} · {'OK' if q['correct'] else 'X'}]  {q['q'][:96]}", i)
                        for i, q in enumerate(qs)]
        q_dd.value = 0
        status.value = (f"<b>{SPECS[key]['label']}</b>: {len(qs)} curated questions "
                        f"(NS-TQA correct on {acc*100:.0f}% of them).  Channels: {st['C']}.")
        _show(0)
    except Exception:
        status.value = f"<b style='color:#B0413E'>Error loading {SPECS[key]['label']} — see output below.</b>"
        with out:
            clear_output(wait=True); traceback.print_exc()

ds_dd.observe(lambda ch: _load(ch["new"]), names="value")
q_dd.observe(lambda ch: _show(ch["new"]) if ch["new"] is not None else None, names="value")

display(widgets.VBox([widgets.HBox([ds_dd]), q_dd, status, out]))
_load("xjtu")'''))

CELLS.append(md(
"""---
### How to read each figure
- **Top:** the real (normalized) signal of the *supporting* channel; the gold band is the
  **critical evidence interval $\\tau$** the executor used; the annotation states the answer's reason.
- **2nd:** the perceived predicate truth — **learned $\\hat\\mu$** (teal) vs the **privileged $\\mu^\\star$** (dashed). Their gap is perception error.
- **3rd:** the robustness $\\rho(t)$; the sign of $\\rho$ at the anchor is the yes/no answer.
- **Bottom:** the unit's **RUL curve** (life remaining) over its whole run-to-failure life, with the
  **question window** (teal) and **critical interval** (gold) marked — so you see *where in life* the question looks.

The header shows whether **NS-TQA** got the question right (✓/✗ vs the privileged label). Increase
`EPOCHS` in the helpers cell for sharper perception; switch datasets with the dropdown (each trains once and is cached)."""))

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({OUT.stat().st_size} bytes, {len(CELLS)} cells)")
# validate it re-loads
json.load(open(OUT)); print("JSON valid")
