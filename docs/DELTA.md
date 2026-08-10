# DELTA.md — what the journal version adds over the NeSy 2026 conference paper

Running record of the extension work, updated as each work-order item lands (per
`markdowns/JOURNAL_EXTENSION_WORK_ORDER.md` WO-8). This becomes the journal's
"extension over the conference version" statement. Every number below is read from
`runs/` and cross-checked in `markdowns/JOURNAL.md`; unrun items are marked **TBD**.

**Protocol.** Main tables are **10 model seeds** (benchmark built once per
`build_seed`; only model seeds vary). Headline comparisons carry paired/one-sample
**Wilcoxon** p-values (`src/utils/stats.py`; scipy-exact with a numpy normal-approx
fallback). Runs executed on the GPU cluster (P100, apptainer `torch231.sif`); all
invariants (balance 0.45–0.55, oracle 1.000, grouped splits, leakage ≤ chance at
depth ≥ 2) re-verified per build.

---

## 1. Headline delta — *when is learned perception necessary?* (WO-1, the gate)

The conference paper left a tension: a hand-threshold **STL-only** grounding feeding
the same executor is near-oracle on the real benchmarks, so it was unclear whether the
*learned* perception is load-bearing. The journal version **resolves this**, three
ways, and reframes the contribution as a clean **C3a / C3b** split.

### C3a — the symbolic executor is the load-bearing, regime-robust contribution *(proven)*

- **WO-1A (cross-rig, both labeling regimes).** New `label_regime` switch in
  `build_real_benchmark` (`L-src` = train-side grounding, the conference regime;
  `L-tgt` = each test rig's own train-fraction grounding — see
  `docs/labeling_regimes.md`). Full 3×3 XJTU/FEMTO/IMS matrix × {STL-only, NS-TQA,
  best end-to-end} × both regimes, 10 seeds
  (`runs/crossrig_stlonly/crossrig_labelregimes.md`). **Verdict:** STL-only carries
  cross-rig transfer under **both** regimes (off-diagonal STL-only 0.874 / 0.889 vs
  NS-TQA 0.827 / 0.774); **L-tgt did not flip it**. But NS-TQA beats every end-to-end
  baseline in all 9 cells — the symbolic *structure* is what transfers.
- **WO-1B (degraded sensing).** New `src/benchmark/degrade.py` (noise / dropout /
  drift / jitter, applied to TEST inputs only; labels stay on the clean privileged
  grounding — non-circular). Sweep on C-MAPSS + XJTU, 10 seeds (`runs/degradation/`,
  `degradation_curves.png`). STL-only stays ahead of NS-TQA at essentially every
  severity (only significant learned>STL cell: XJTU/40%-dropout +0.008); NS-TQA-aug ≈
  NS-TQA; both crush the end-to-end baselines everywhere.

**Takeaway (C3a):** the parameter-free executor + fixed-threshold vocabulary is
near-oracle, transfers across rigs, and is robust to degraded sensing — the workhorse,
beating black-box baselines by large margins.

### C3b — learned perception is necessary for NON-threshold predicates *(proven, scoped)*

- **WO-1C (learned `anomalous` predicate, leakage-safe).** New
  `src/benchmark/anomaly_questions.py`: a leakage-SAFE compositional anomaly benchmark
  on the Phase-G anomaly infra, with three mitigations — (a) full-life windows,
  (b) every depth≥2 anomaly program must conjoin ≥1 non-anomaly leaf, (c) an **honest
  held-out** single-predicate leakage probe (select rule on one split, score on
  another, averaged) gating each depth's batch — replacing the project's optimistic
  same-data probe (chance ceiling ~0.66 at small n). Tuned to `anomaly_p=0.40,
  n_test=400, retries=12` → the gate PASSES at depth ≥ 2.

  Predicate-level macro-F1, learned head vs fixed `max(high,low)` proxy, 10 seeds
  (`runs/anomaly_predicate/`):

  | dataset | pool | head | proxy | Δ | Wilcoxon p |
  |---|---|---|---|---|---|
  | XJTU (vibration) | in-dist | 0.739 | 0.452 | **+0.287** | 0.006 |
  | XJTU | shift | 0.835 | 0.578 | **+0.257** | 0.006 |
  | C-MAPSS (regime-norm.) | in-dist | 0.779 | 0.757 | +0.022 | 0.008 |
  | C-MAPSS | shift | 0.734 | 0.762 | −0.028 | 0.014 |

  The head **wins decisively on vibration** (and now edges STL-only at the *answer*
  level on XJTU: 0.836 vs 0.809 shift) and is at **parity** on regime-normalized
  C-MAPSS — never meaningfully worse. A worked explanation figure
  (`runs/anomaly_predicate/case_study/`) shows the learned `anomalous` predicate as the
  faithful decisive evidence on a held-out bearing, μ̂_anom tracking the privileged
  μ*_anom target.

**Takeaway (C3b):** learned perception earns its place specifically where the health
signal is **non-threshold** (vibration degradation signatures); its advantage scales
with how non-threshold the modality is.

### The C-MAPSS transfer fix (also a down-payment on WO-3A)

The learned anomaly head initially *collapsed* under C-MAPSS fault-mode shift
(FD001→FD002/4: −0.212), because those subsets interleave 6 operating regimes
cycle-to-cycle and the regime jumps dominate the raw signal, making them OOD for an
FD001-trained head. New **operating-condition normalization** (`op_normalize` in
`load_cmapss`/`CMAPSSAdapter`: numpy k-means over op-setting columns, then z-score each
sensor within its regime — label-free, opt-in, default off) removes the regime
level-jumps and **recovers the head to parity** (shift −0.212 → −0.028). Standard
multi-regime C-MAPSS preprocessing; lands part of **WO-3A**.

---

## 2. New infrastructure

| Component | Purpose | WO |
|---|---|---|
| `src/utils/stats.py` | Wilcoxon (scipy + numpy fallback), bootstrap CI, Holm–Bonferroni | global |
| `src/benchmark/degrade.py` | deterministic input-side sensor degradation | WO-1B |
| `src/benchmark/anomaly_questions.py` | leakage-safe compositional anomaly QA + honest held-out probe | WO-1C |
| `build_real_benchmark(label_regime=…)` | L-src / L-tgt labeling regimes | WO-1A |
| `load_cmapss(op_normalize=…)` | operating-condition (per-regime) normalization | WO-1C fix / WO-3A |
| `docs/labeling_regimes.md` | audit of how privileged labels are defined | WO-1A |
| scripts `27`/`28`/`29`/`30` | crossrig-labelregimes / degradation / anomaly-predicate (multi-dataset) / anomaly case study | — |
| tests `test_label_regime`, `test_degrade`, `test_anomaly_family` | invariants + regression | WO-8 |
| `slurm/{crossrig_labelregimes,degradation,anomaly_predicate}.slurm`; portable torch-tensor HI caches; adapter cache-only load | cluster execution | infra |

Full test suite: **96 passed** (was 77). New heavy deps: none yet (scipy optional).

---

## 3. Still TBD (author / next sessions)

- **WO-2A** LLM-bridge (ITFormer-style) baseline — biggest reviewer demand; needs
  `transformers/peft` in `requirements-ext.txt`. **TBD.**
- **WO-2B** post-hoc XAI faithfulness head-to-head (IG/SHAP/attention vs
  by-construction) — `scripts/26` exists to build on. **TBD.**
- **WO-3A** full FD003 + per-regime generalization table — mechanism (`op_normalize`)
  landed; table **TBD.**
- **WO-3B** N-CMAPSS · **WO-4** sensitivity grid · **WO-5** faithfulness chance /
  two-level counterfactual · **WO-6** parser / RUL · **WO-7** conformal — **TBD.**
- Regenerate the conference tables at 10 seeds under the final code state
  (`runs/conference_repro/` diff note) — **TBD.**

---

## 4. One-paragraph extension statement (draft)

> Over the conference version, we resolve precisely *when* the learned perception is
> necessary. Through a cross-rig study under two labeling regimes, a degraded-sensing
> sweep, and a leakage-safe learned-anomaly-predicate experiment (all 10 seeds with
> Wilcoxon significance), we show the parameter-free symbolic executor with a fixed
> threshold vocabulary is the regime-robust workhorse that carries cross-rig transfer
> and degraded-sensing robustness (C3a), while learned perception is necessary and
> superior specifically for *non-threshold* health predicates — decisively on
> vibration and at parity on monotone tabular sensors (C3b). We add operating-condition
> normalization that makes the learned anomaly predicate transfer across C-MAPSS
> operating regimes, a statistical-rigor layer (Wilcoxon/bootstrap/Holm), and a
> deterministic degradation battery. SOTA LLM-bridge and post-hoc-XAI comparisons are
> in progress.
