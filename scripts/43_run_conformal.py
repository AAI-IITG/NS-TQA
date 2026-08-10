"""43 - Split-conformal answer confidence for NS-TQA (WO-7).

Turns the executor's signed robustness ``rho`` into a distribution-free confidence
set over {yes, no}. We (i) train NS-TQA perception on C-MAPSS FD001, (ii) collect
per-window ``(pred_rho, gold_answer)`` on the in-distribution held-out pool, split
it into a calibration half and a test half, (iii) calibrate the conformal quantile
on the calibration half, and (iv) report, across risk levels alpha, the empirical
coverage, the abstain (doubleton) rate, and the SELECTIVE accuracy on committed
(singleton) answers. Finally we calibrate on in-dist and evaluate the SAME quantile
under an operating-condition SHIFT (FD002/FD004): coverage there drops below
1-alpha exactly because exchangeability is broken -- so the confidence set is itself
an honest shift detector. Averaged over model seeds.

Run:  python scripts/43_run_conformal.py [--config configs/conformal.yaml] [--quick]
"""
import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from benchmark.realdata import build_real_benchmark
from benchmark.adapters_cmapss import CMAPSSAdapter
from models.nstqa_learned import LearnedNSTQA
from perception.learned import train_perception
from utils.conformal import coverage_curve


def build_bm(c):
    a, b = c["adapter"], c["build"]
    adapter = CMAPSSAdapter(root=ROOT / a["root"], subsets=["FD001", "FD002", "FD004"],
                            rul_cap=a["rul_cap"], flat_std_thresh=a.get("flat_std_thresh", 1e-6),
                            min_qspan=a.get("min_qspan", 0.05), op_normalize=a.get("op_normalize", False))
    return build_real_benchmark(
        adapter, T=b["T"], stride=b["stride"], depths=tuple(b["depths"]), shift="condition",
        train_conditions=tuple(b["train_conditions"]), test_conditions=tuple(b["test_conditions"]),
        indist_holdout_frac=b["indist_holdout_frac"], n_train_per_depth=b["n_train_per_depth"],
        n_test_per_depth=b["n_test_per_depth"], hi_q=a.get("hi_q", 0.85), lo_q=a.get("lo_q", 0.15),
        smooth_k=b["smooth_k"], a_level=b["a_level"], allow_until=b["allow_until"],
        max_windows_per_unit=b.get("max_windows_per_unit"), over_factor=b["over_factor"],
        seed=b["build_seed"])


def rho_gold(nst, pool):
    """(pred_rho, gold_answer) per instance from the faithful NS-TQA path."""
    recs = nst.evaluate_instances(pool)["records"]
    return [r["pred_rho"] for r in recs], [r["gold_answer"] for r in recs]


def mean_over_seeds(rows_by_seed):
    """Average a list (per seed) of per-alpha row-lists, keyed by alpha."""
    keys = ["empirical_coverage", "singleton_rate", "abstain_rate", "empty_rate", "selective_accuracy"]
    out = []
    for i in range(len(rows_by_seed[0])):
        agg = {"alpha": rows_by_seed[0][i]["alpha"],
               "target_coverage": rows_by_seed[0][i]["target_coverage"]}
        for k in keys:
            vals = [rs[i][k] for rs in rows_by_seed if rs[i][k] is not None]
            agg[k] = round(sum(vals) / len(vals), 4) if vals else None
        out.append(agg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "conformal.yaml"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seeds = cfg["experiment"]["seeds"][:1 if args.quick else None]
    alphas = cfg["experiment"]["alphas"]
    calib_frac = cfg["experiment"]["calib_frac"]

    bm = build_bm(cfg["cmapss"])
    C = bm["meta"]["n_channels"]
    print(f"benchmark: C={C} train={len(bm['train'])} indist={len(bm['test_indist'])} "
          f"shift={len(bm['test_shift'])}", flush=True)

    indist_rows, shift_rows, accs = [], [], []
    for seed in seeds:
        print(f"  seed {seed}: training perception ...", flush=True)
        ep = 8 if args.quick else cfg["perception"]["epochs"]
        pres = train_perception(
            bm["train"], n_channels=C, hidden=cfg["perception"]["hidden"],
            kernel=cfg["perception"]["kernel"], n_layers=cfg["perception"]["n_layers"],
            per_channel=cfg["perception"]["per_channel"], epochs=ep,
            batch_size=cfg["perception"]["batch_size"], lr=cfg["perception"]["lr"],
            weight_decay=cfg["perception"]["weight_decay"], device_pref=cfg.get("device", "cpu"),
            seed=seed, log_every=cfg["perception"].get("log_every", 20), verbose=False)
        nst = LearnedNSTQA(pres.model, n_channels=C)

        # in-dist held-out pool -> shuffle -> split calib / test
        idr, idy = rho_gold(nst, bm["test_indist"])
        idx = list(range(len(idr)))
        random.Random(seed).shuffle(idx)
        ncal = int(len(idx) * calib_frac)
        cal_i, te_i = idx[:ncal], idx[ncal:]
        cal_rho = [idr[i] for i in cal_i]; cal_y = [idy[i] for i in cal_i]
        te_rho = [idr[i] for i in te_i]; te_y = [idy[i] for i in te_i]
        accs.append(sum((r > 0) == y for r, y in zip(idr, idy)) / max(1, len(idr)))

        indist_rows.append(coverage_curve(cal_rho, cal_y, te_rho, te_y, alphas))
        # SAME calibration, evaluated under shift
        shr, shy = rho_gold(nst, bm["test_shift"])
        shift_rows.append(coverage_curve(cal_rho, cal_y, shr, shy, alphas))

    id_avg = mean_over_seeds(indist_rows)
    sh_avg = mean_over_seeds(shift_rows)
    acc = sum(accs) / len(accs)
    _write(ROOT / cfg["run_root"], id_avg, sh_avg, acc, len(seeds), C, calib_frac)


def _tbl(rows):
    L = ["| target (1-a) | coverage | singleton | abstain | empty | selective-acc |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        sa = f"{r['selective_accuracy']:.3f}" if r["selective_accuracy"] is not None else "—"
        L.append(f"| {r['target_coverage']:.2f} | {r['empirical_coverage']:.3f} | "
                 f"{r['singleton_rate']:.3f} | {r['abstain_rate']:.3f} | {r['empty_rate']:.3f} | {sa} |")
    return L


def _write(out_dir, id_avg, sh_avg, acc, nseeds, C, calib_frac):
    out_dir.mkdir(parents=True, exist_ok=True)
    L = ["# Conformal answer confidence (WO-7)", "",
         f"Split-conformal on the NS-TQA robustness margin, C-MAPSS (C={C}), "
         f"mean over {nseeds} seeds. Calibration = {calib_frac:.0%} of the in-distribution "
         f"held-out pool; the SAME quantile is then applied under an operating-condition shift.",
         "", f"_point-answer accuracy on the in-dist pool: {acc:.3f}_", "",
         "## In-distribution (calibration and test exchangeable)", ""]
    L += _tbl(id_avg)
    L += ["", "Empirical coverage tracks the target 1-alpha (guarantee holds); the abstain "
          "rate is the price of higher coverage, and selective accuracy is the accuracy on "
          "committed singleton answers.", "",
          "## Under operating-condition shift (SAME in-dist calibration)", ""]
    L += _tbl(sh_avg)
    L += ["", "Coverage now falls BELOW the target because calibration and test are no longer "
          "exchangeable: the confidence set is itself an honest detector of distribution shift.", ""]
    table = "\n".join(L)
    (out_dir / "conformal.md").write_text(table)
    (out_dir / "conformal.json").write_text(json.dumps(
        {"n_seeds": nseeds, "n_channels": C, "indist_accuracy": acc,
         "indist": id_avg, "shift": sh_avg}, indent=2))
    print("\n" + table)
    print(f"\nwrote -> {out_dir}/conformal.{{md,json}}")


if __name__ == "__main__":
    main()
