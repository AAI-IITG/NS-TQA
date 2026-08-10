# Labeling regimes — how privileged answer labels are defined (WO-1A audit)

This document answers one question that the journal review (`PRESENT_SITUATION.md` §2a)
made load-bearing: **whose grounding defines the privileged answer labels on the
shifted/test-rig data?** The answer decides whether the cross-rig / cross-condition
tables measure *executor + predicate-vocabulary transfer* (a C4 claim) or *learned-
perception necessity* (C3b). It is stated here so the paper can cite the regime
explicitly rather than leaving it implicit in the code.

Read alongside `AGENTS.md` §2 (non-circular labels; train-only fitting) and
`src/benchmark/realdata.py`.

---

## 1. The label pipeline (as built)

Every real benchmark is produced by `build_real_benchmark(adapter, ...)`
(`src/benchmark/realdata.py`). The privileged answer for a QA instance is

```
answer_star = executor(phi_star, mu_star) > 0        # hard_logic.evaluate
mu_star     = ground(normalizer.transform(X), calibrator)   # perception.grounding.ground
```

Two objects define `mu_star`, and therefore the label:

1. **`normalizer`** — a `ChannelNormalizer` (per-channel z-score). It maps the raw
   modeling-ready window `X` to the model's input view.
2. **`calibrator`** — a `Calibrator` holding per-channel `hi`/`lo` quantile
   thresholds and a slope scale; `ground()` turns the normalized window into the
   privileged predicate truths `mu_star ∈ [T, 4C]` (`high/low/rising/falling`).

**Where they are fit (verified in code):**

| Object | Fit on | Code |
|---|---|---|
| `normalizer` | **TRAIN windows only** | `realdata.py:202` (`ChannelNormalizer().fit(train_windows)`) |
| `calibrator` | **train-normalized TRAIN windows only** | `realdata.py:203–204` (`Calibrator.fit(train_norm, ...)`) |

**Where they are applied:** to *every* pool — `train`, `test_indist`, and
`test_shift` — inside `_build_balanced`'s `ground_window` (`realdata.py:64–66`):
the same train-fit `normalizer` and `calibrator` compute `mu_star` for the shifted
test windows too. There is no separate test-side grounding.

This is correct and non-circular for the model (the model never sees the
calibrator/thresholds; it must infer `mu_hat` and is scored on held-out units and a
shifted regime — `AGENTS.md` §2.2). The subtlety is **only** about the label:
on the shifted/other-rig data the label is defined by the *train-side* grounding.

---

## 2. The two regimes

We name the two possible ways to ground the shifted/test-rig labels:

### L-src — *labels from the source (train-side) grounding*
> Test labels are the **train-side grounding applied to test signals.** The
> `normalizer` and `calibrator` are fit on train units only and used to define
> `mu_star` (hence `answer_star`) on every test window, including the shifted /
> other-rig ones.

This is **the regime the conference paper (NeSy 2026, draft v1) uses**, and the
current default of `build_real_benchmark`. Consequence, already measured
(`PRESENT_SITUATION.md` §7, 2026-06-14): because the labels are per-channel
*quantile* thresholds on *z-normalized* windows, every rig's decision boundary sits
near +1σ / −1σ, so a **fixed-threshold STL-only** grounding reproduces the labeler
almost exactly — STL-only is near-oracle and beats learned NS-TQA cross-rig (9/9
cells). Under L-src the cross-rig table is therefore an **executor + shared-
vocabulary transfer** result (**C4**), *not* evidence that the *learned* perception
is necessary (C3b).

### L-tgt — *labels from the target (test-rig-fit) grounding*
> Test labels are defined by a grounding **fit on the test rig/condition's own
> training-fraction units** — never on the evaluated windows. Each test condition
> gets its own `normalizer_tgt` + `calibrator_tgt`, fit on a held-out label-fitting
> fraction of that condition's units; the evaluated windows of that condition are
> labeled by *their own rig's* grounding.

Under L-tgt the label boundary is set in the test rig's *own* statistics, which a
fixed train-rig threshold rule need **not** reproduce. The hypothesis WO-1A tests:
**STL-only (train-rig thresholds) should degrade cross-rig under L-tgt, while learned
perception — which adapts per channel/regime — transfers the concept.** If it does,
L-tgt is the regime that isolates learned-perception necessity (C3b) at the answer
level; if STL-only still tracks the L-tgt labels, the honest conclusion is that the
symbolic executor carries cross-rig transfer regardless (the numbers decide — no
spin, per WO-1A acceptance).

**Non-circularity preserved under L-tgt.** The label-fitting units are disjoint from
the evaluated windows (a within-condition unit split), so no evaluated window's own
signal defines its label. The model's *input* `X` is still normalized by the
**train-side** `normalizer` in both regimes — the distribution shift the model must
overcome is unchanged; L-tgt moves *only* the label-defining grounding, so the
model-side comparison stays apples-to-apples. `mu_star` stored on shifted instances
(used as the oracle and the perception-F1 / faithfulness reference) is, under L-tgt,
the test-rig grounding — which is the more honest privileged target for those units.

---

## 3. Per-benchmark summary (which regime each existing result used)

| Benchmark | Builder call | Regime as run | Claim it currently supports |
|---|---|---|---|
| C-MAPSS FD001→FD002/4 (`scripts/12`) | `shift="condition"` | **L-src** | cross-condition generalization + necessity-on-real |
| XJTU cross-load (`scripts/14`) | `shift="condition"` | **L-src** | cross-load generalization + necessity-on-real |
| Cross-rig XJTU/FEMTO/IMS (`scripts/19`) | `shift="condition"`, condition = rig | **L-src** | C4 (executor + vocabulary transfer); **not** C3b |
| Synthetic planted-spurious | `spurious.py` (planted truths, not grounding-defined) | n/a — labels are planted class truths + noise | necessity + composition (C3b holds here: STL-only 0.777 < NS-TQA 0.884) |

**Conference-paper regime, stated for the paper:** all *real-data* tables in the
NeSy 2026 draft are **L-src**. The only place learned-perception necessity is shown
at the answer level is the *synthetic* benchmark (planted labels, not a grounding),
where a fixed threshold cannot recover the planted truth.

---

## 4. What WO-1A adds

1. A `label_regime ∈ {"L-src", "L-tgt"}` switch on `build_real_benchmark`
   (default `"L-src"` = current behavior, documented above).
2. The full 3×3 cross-rig matrix (FEMTO/IMS/XJTU) × {STL-only, NS-TQA, best
   end-to-end} × {L-src, L-tgt}, 10 model seeds (STL-only deterministic → 1 seed),
   written to `runs/crossrig_stlonly/crossrig_labelregimes.md`.
3. The extracted claim goes in that table's caption, whichever way it falls —
   either "STL-only transfers ⇒ executor carries cross-rig too" or "STL-only fails
   cross-rig under L-tgt while learned perception transfers the concept."
