"""23 - Cross-rig transfer of the LEARNED ANOMALY predicate (Phase H.3', the
principled neuro-necessity / C3b experiment).

WHY this and not script 19's threshold QA: the high/low/rising/falling families are
per-channel quantile groundings on *standardized* windows, so every rig's quantile
boundary maps to ~+/-1 sigma. A fixed-threshold STL-only is therefore near-oracle
and BEATS learned NS-TQA cross-rig (script 19, 9/9 cells). That table proves the
EXECUTOR + vocabulary transfer (C4), not that the *learned* perception is necessary.

The ``anomalous`` predicate has a NON-threshold target: deviation from a per-channel
HEALTHY baseline (early life). A fixed "high OR low" proxy has no notion of a healthy
reference, so it cannot reproduce it (in-rig: head F1 ~0.75 vs proxy ~0.57). This
script asks whether the LEARNED head transfers that concept ACROSS RIGS better than
the fixed proxy -- the honest C3b-at-cross-rig test.

Protocol (per train rig A -> test rig B), label-free SELF-normalized per rig so the
only variable is the concept (not raw scale):
  * fit HealthyBaseline_R on each rig R's early-life (healthy) self-normed windows;
  * train AnomalyHead on rig A's full-life windows (target = baseline_A);
  * GOLD anomaly on rig B = baseline_B.target > 0.5 (rig B's OWN healthy reference);
  * LEARNED pred = head_A(rig-B windows) > 0.5;  PROXY pred = max(high,low) > 0.5;
  * report binary macro-F1(pred, gold) over (channels x time) for both, 3x3.

C3b is supported iff the learned head beats the proxy OFF-DIAGONAL (transfer cells).

Run:  python scripts/23_run_anomaly_crossrig.py [--config configs/cross_rig.yaml] [--quick]
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

from benchmark.adapters import window_series
from benchmark.adapters_femto import FEMTOAdapter
from benchmark.adapters_ims import IMSAdapter
from benchmark.adapters_multirig import MultiRigAdapter
from benchmark.adapters_xjtu import XJTUAdapter
from models.stl_only import hand_rule_calibrator
from perception.anomaly import (AnomalyHead, HealthyBaseline, healthy_windows,
                                predict_anomaly, train_anomaly_head)
from perception.grounding import ground


def mean_std(xs):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return (float("nan"), 0.0)
    return (sum(xs) / len(xs), stats.pstdev(xs) if len(xs) > 1 else 0.0)


def build_multirig(cfg) -> MultiRigAdapter:
    a = cfg["adapters"]
    nb = a["n_bands"]
    children = []
    for rig in cfg["rigs"]:
        if rig == "xjtu":
            children.append(XJTUAdapter(root=ROOT / a["xjtu_root"], n_bands=nb))
        elif rig == "femto":
            children.append(FEMTOAdapter(root=ROOT / a["femto_root"], n_bands=nb))
        elif rig == "ims":
            tests = tuple(a.get("ims_tests") or ()) or None
            children.append(IMSAdapter(root=ROOT / a["ims_root"], n_bands=nb, tests=tests))
        else:
            raise ValueError(f"unknown rig {rig!r}")
    return MultiRigAdapter(children)


def rig_full_windows(series, rig, T, stride, max_per_unit):
    """Raw full-life windows [N,T,C] for one rig (all its units)."""
    out = []
    for s in series:
        if s.condition != rig:
            continue
        for w in window_series(s, T, stride, max_per_unit):
            out.append(w.X)
    return torch.stack(out) if out else torch.empty(0)


def self_norm_stats(stack):
    """Per-channel mean/std from a [N,T,C] stack (label-free)."""
    flat = stack.reshape(-1, stack.shape[-1])
    return flat.mean(0), flat.std(0).clamp_min(1e-6)


def macro_f1(pred: torch.Tensor, gold: torch.Tensor) -> float:
    """Binary macro-F1 (mean of positive- and negative-class F1) over all entries."""
    f1s = []
    for cls in (True, False):
        p = (pred == cls)
        g = (gold == cls)
        tp = int((p & g).sum())
        fp = int((p & ~g).sum())
        fn = int((~p & g).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / 2


def proxy_anomaly(windows_n: torch.Tensor, C: int, a_level: float, smooth_k: int) -> torch.Tensor:
    """Fixed hand-rule 'anomalous' ~ max(high, low) on self-normed windows -> [N,T,C] in [0,1]."""
    cal = hand_rule_calibrator(C, hi=1.0, lo=-1.0, slope_scale=1.0, a_level=a_level, smooth_k=smooth_k)
    out = []
    for w in windows_n:
        mu4, _ = ground(w, cal)                       # [T, 4C], family-major
        out.append(torch.maximum(mu4[:, :C], mu4[:, C:2 * C]))
    return torch.stack(out)                            # [N,T,C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "cross_rig.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    b = cfg["build"]
    an = cfg.get("anomaly", {})
    healthy_frac = an.get("healthy_frac", 0.25)
    q = an.get("q", 0.99)
    a_anom = an.get("a_anom", 4.0)
    smooth_k = b["smooth_k"]
    a_level = b["a_level"]
    T, stride = b["T"], b["stride"]
    maxper = b.get("max_windows_per_unit") or 80
    epochs = an.get("epochs", 60)
    seeds = list(cfg["experiment"]["seeds"])
    if args.quick:
        seeds = seeds[:1]
        epochs = 15

    adapter = build_multirig(cfg)
    print(f"loading rigs {cfg['rigs']} ...", flush=True)
    series = adapter.load()
    present = adapter.conditions(series)
    rigs = [r for r in cfg["rigs"] if r in present]
    C = series[0].C
    print(f"  channels={C}; rigs={rigs}; T={T} stride={stride} maxper={maxper} epochs={epochs}")

    # ---- per-rig: self-norm stats, normalized full windows, healthy baseline ----
    full_n, base = {}, {}
    for r in rigs:
        full_raw = rig_full_windows(series, r, T, stride, maxper)
        healthy_raw = healthy_windows([s for s in series if s.condition == r],
                                      healthy_frac, T, stride, max_per_unit=maxper)
        m, s = self_norm_stats(full_raw)
        full_n[r] = (full_raw - m) / s
        healthy_n = (healthy_raw - m) / s
        base[r] = HealthyBaseline.fit(healthy_n, q=q, a_anom=a_anom, smooth_k=smooth_k)
        print(f"  {r}: full {tuple(full_n[r].shape)}, healthy {tuple(healthy_n.shape)}")

    # gold anomaly per rig (rig's OWN healthy baseline), and the fixed proxy prediction
    gold = {r: torch.stack([base[r].target(w) for w in full_n[r]]) > 0.5 for r in rigs}
    proxy_pred = {r: proxy_anomaly(full_n[r], C, a_level, smooth_k) > 0.5 for r in rigs}
    proxy_f1 = {r: macro_f1(proxy_pred[r], gold[r]) for r in rigs}      # same every train rig
    pos_rate = {r: float(gold[r].float().mean()) for r in rigs}
    print("  gold anomaly positive-rate per rig: " +
          ", ".join(f"{r}:{pos_rate[r]:.2f}" for r in rigs))

    # ---- learned head: train on rig A, eval F1 on every rig B ----
    head_f1 = {tr: {te: [] for te in rigs} for tr in rigs}
    for tr in rigs:
        for seed in seeds:
            print(f"  train anomaly head on {tr} (seed {seed}) ...", flush=True)
            head = train_anomaly_head(full_n[tr], base[tr], hidden=cfg["perception"]["hidden"],
                                      kernel=cfg["perception"]["kernel"],
                                      n_layers=cfg["perception"]["n_layers"], epochs=epochs,
                                      device_pref=cfg.get("device", "cpu"), seed=seed)
            for te in rigs:
                pred = torch.stack([predict_anomaly(head, w) for w in full_n[te]]) > 0.5
                head_f1[tr][te].append(macro_f1(pred, gold[te]))

    # ---- write matrices + neuro-necessity readout ----
    out_dir = ROOT / cfg["run_root"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def f1cell(tr, te):
        m, _ = mean_std(head_f1[tr][te])
        return "—" if m != m else f"{m:.3f}"

    head_lines = ["### Learned anomaly head — macro-F1 (train rows -> test cols)", "",
                  "| train ＼ test | " + " | ".join(rigs) + " |", "|---" * (len(rigs) + 1) + "|"]
    for tr in rigs:
        head_lines.append(f"| **{tr}** | " + " | ".join(f1cell(tr, te) for te in rigs) + " |")

    proxy_lines = ["### Fixed hand-rule proxy (max(high,low)) — macro-F1 (no train; same per col)", "",
                   "| test rig | " + " | ".join(rigs) + " |", "|---" * (len(rigs) + 1) + "|",
                   "| proxy | " + " | ".join(f"{proxy_f1[r]:.3f}" for r in rigs) + " |"]

    # off-diagonal (transfer) means
    def off_head():
        v = [mean_std(head_f1[tr][te])[0] for tr in rigs for te in rigs if tr != te]
        v = [x for x in v if x == x]
        return sum(v) / len(v) if v else float("nan")

    def diag_head():
        v = [mean_std(head_f1[tr][tr])[0] for tr in rigs]
        v = [x for x in v if x == x]
        return sum(v) / len(v) if v else float("nan")

    off_proxy = sum(proxy_f1[r] for r in rigs) / len(rigs)  # proxy has no train rig -> per-rig mean
    h_off, h_dia = off_head(), diag_head()
    margin = h_off - off_proxy
    verdict = ("learned anomaly head TRANSFERS the concept cross-rig and beats the fixed proxy "
               "-> C3b supported at the answer/predicate level cross-rig"
               if margin > 0.02 else
               "learned head does NOT beat the fixed proxy off-diagonal -> C3b does not transfer "
               "cross-rig; rest C3b on synthetic + in-rig anomaly (held-out) and reframe")
    readout = "\n".join([
        "### Neuro-necessity readout (C3b, anomaly predicate cross-rig)", "",
        "| metric | learned head | fixed proxy | head − proxy |",
        "|---|---|---|---|",
        f"| diagonal (same-rig) mean | {h_dia:.3f} | {off_proxy:.3f} | {h_dia - off_proxy:+.3f} |",
        f"| off-diagonal (transfer) mean | {h_off:.3f} | {off_proxy:.3f} | {margin:+.3f} |",
        "", f"**Verdict:** {verdict}.",
    ])

    table = "\n\n".join([f"# Cross-rig ANOMALY-predicate transfer ({len(seeds)} seeds): "
                         f"{' / '.join(rigs)}",
                         "_Self-normalized per rig (label-free); gold = each rig's own healthy "
                         "baseline. The fixed proxy has no healthy reference, so a learned head that "
                         "keeps its lead off-diagonal is genuine learned-perception necessity (C3b)._",
                         "\n".join(head_lines), "\n".join(proxy_lines), readout])
    (out_dir / "anomaly_crossrig.md").write_text(table)
    (out_dir / "anomaly_crossrig.json").write_text(json.dumps({
        "rigs": rigs, "seeds": seeds,
        "head_f1": {tr: {te: list(mean_std(head_f1[tr][te])) for te in rigs} for tr in rigs},
        "proxy_f1": proxy_f1, "gold_positive_rate": pos_rate,
    }, indent=2, default=str))
    print("\n" + table)
    print(f"\nwrote anomaly cross-rig matrices to {out_dir}")


if __name__ == "__main__":
    main()
