# NS-TQA: Neuro-Symbolic Temporal Question Answering for Explainable PHM

Faithful, shortcut-resistant temporal question answering over multivariate machine-health
signals. A neural **perception** module grounds a raw signal window into human-named health
predicates; a deterministic **Signal Temporal Logic (STL) executor** then answers compositional
temporal questions over those predicates. Because the answer is *computed* by executing the
program, the explanation — the supporting predicate, the critical evidence interval, and the
robustness margin — is a **by-product of the same computation**, not a post-hoc rationalization.

```
raw signal window  X ∈ ℝ^{T×N}
        │  f_θ   (learned perception)
        ▼
symbolic state  μ ∈ [0,1]^{T×P}   (predicates: high / low / rising / falling / anomalous)
        │  E(φ, μ)   (deterministic hard-STL executor — no learnable parameters)
        ▼
answer a  +  robustness ρ  +  critical interval τ  +  supporting predicate
```

A correct NS-TQA answer therefore entails both an adequate perceived state **and** a correct
program executed over it — so the symbolic layer is load-bearing, and end-to-end neural models
(which can exploit shortcuts) are included only as ablations/baselines.

This is the code for our NeSy 2026 paper and its journal extension.

---

## Installation

Python ≥ 3.10.

```bash
git clone https://github.com/AAI-IITG/NS-TQA.git
cd NS-TQA
pip install -r requirements.txt          # lean core (torch, numpy, pyyaml, matplotlib, tqdm, pytest)
pip install -r requirements-ext.txt      # optional: scipy, transformers, peft, shap, scikit-learn
                                         # (only needed for the LLM-bridge, post-hoc XAI, and
                                         #  learned-STL-tree experiments; the core runs without them)
```

`pyproject.toml` puts `src/` on the path for `pytest`; scripts also self-bootstrap `src/`, so run
everything from the repository root.

Sanity check (no data needed):

```bash
python -m pytest tests/ -q
```

---

## Repository layout

| Path | Contents |
|---|---|
| `src/executor/` | STL AST/grammar, hard (exact) + soft (differentiable) robustness semantics, `parse_program` |
| `src/perception/` | learned per-channel grounding, calibrator, feature front-ends, anomaly head |
| `src/benchmark/` | dataset adapters (C-MAPSS, XJTU, FEMTO/PRONOSTIA, IMS), non-circular QA benchmark builders, synthetic + spurious generators |
| `src/models/` | faithful `LearnedNSTQA`, STL-only baseline, end-to-end baselines, LLM-bridge |
| `src/explain/` | by-construction explanation extraction + post-hoc attribution reduction (grad×input, IG, SHAP) |
| `src/parser/` | grammar-verified NL→program parser (proof-of-concept) |
| `src/utils/` | statistics (Wilcoxon/Holm), faithfulness metrics, conformal prediction |
| `scripts/` | numbered, runnable experiment entry points (see below) |
| `configs/` | YAML configs for each experiment |
| `tests/` | unit tests for the executor, perception, benchmark, and metrics |

---

## Data

The datasets are **not** bundled (they are large and separately licensed). Download the four
public PHM datasets and place them under `data/raw/` as described in
[`data/README.md`](data/README.md). In brief:

| Dataset | Used for | Expected path under `data/raw/` |
|---|---|---|
| NASA C-MAPSS (turbofan) | engine QA, shift factorization, RUL, conformal | `CMAPSS/` |
| XJTU-SY (bearings) | bearing QA, cross-load shift, cross-rig | `XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/` |
| FEMTO / PRONOSTIA (IEEE PHM 2012) | cross-rig transfer | `IEEE_PRONOSTIA_PHM/dataset/` |
| NASA IMS (bearings) | cross-rig transfer | `NASA_IMS_Bearings/` |

The synthetic-necessity experiments need **no** downloaded data.

---

## Quickstart

**1) Synthetic necessity (no data required)** — shows that end-to-end baselines collapse to a
planted shortcut under distribution shift while NS-TQA is invariant by construction:

```bash
python scripts/11_run_necessity_multiseed.py --config configs/necessity.yaml
```

**2) Real-data generalization under natural shift** (after setting up C-MAPSS + XJTU):

```bash
python scripts/12_run_cmapss_necessity.py  --config configs/cmapss_necessity.yaml
python scripts/14_run_xjtu_necessity.py    --config configs/xjtu_necessity.yaml
```

Results (tables + JSON) are written under `runs/<experiment>/`.

---

## Reproducing the paper experiments

Each script reads a config in `configs/` and writes to `runs/`. Grouped by theme:

| Theme | Scripts | Config(s) |
|---|---|---|
| Synthetic + real necessity (shortcut resistance) | `10`, `11`, `12`, `14`, `25` | `necessity.yaml`, `cmapss_necessity.yaml`, `xjtu_necessity.yaml` |
| Headline generalization (10 seeds + Holm significance) | `41` | `generalization10.yaml` |
| Cross-rig transfer + label-regime attribution | `19_run_cross_rig`, `27` | `cross_rig.yaml`, `crossrig_labelregimes.yaml` |
| Learned anomaly (non-threshold) predicate | `29`, `23`, `30` | `anomaly_predicate.yaml` |
| Faithfulness vs. post-hoc XAI (SHAP/IG/grad×input) | `15`, `26`, `32`, `33`, `39` | `faithfulness.yaml`, `posthoc_xai.yaml` |
| SOTA LLM-bridge baseline (Qwen 1.5B / 7B) | `24`, `31` | `llm_bridge_instruct.yaml`, `llm_bridge_instruct_7b.yaml` |
| Shift factorization (op-regime vs. fault mode) | `35` | `cmapss_regimes.yaml` |
| Prognostic RUL-grounded questions | `44`, `19_run_rul_questions` | `rul.yaml` |
| Conformal answer confidence | `43` | `conformal.yaml` |
| Degraded-sensing robustness | `28` | `degradation.yaml` |
| NL→program parser (grammar firewall, proof-of-concept) | `42` | — |
| Sensitivity / executor profiling / stats / figures | `36`, `37`, `38`, `22` | `sensitivity.yaml` |

Most scripts accept `--quick` for a fast smoke run and `--config <path>` to override the default.
The heavier experiments were run on a GPU; each script runs standalone from the repository root.

---

## Design notes

- The executor is deterministic and **parameter-free**; inference uses hard STL semantics
  (soft/differentiable semantics are used only to backpropagate perception gradients).
- Questions are structured STL programs; a natural-language front-end is an optional, grammar-
  *verified* add-on (`src/parser/`) that can never place a malformed program on the answer path.
- The benchmark is **non-circular**: answer labels come from a privileged grounding fit on
  training data only, and the model is always scored on held-out units or shifted conditions.

---

## Citation

If you use this code, please cite the NeSy 2026 paper (journal version in preparation):

```bibtex
@inproceedings{das2026nstqa,
  title     = {Neuro-Symbolic Temporal Question Answering for Explainable PHM},
  author    = {Das, Nirban and Dutta Baruah, Rashmi},
  booktitle = {Neurosymbolic Learning and Reasoning (NeSy)},
  year      = {2026}
}
```

