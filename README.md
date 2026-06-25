# Imputer × Forecaster Benchmark on FreshRetailNet-50K

Systematic benchmark of the interaction between stockout-imputation strategies and
demand forecasters on perishable retail data (FreshRetailNet-50K, 50 000 series).

The repository underpins an applied research paper currently in preparation. All
finding numerics, tables and claims are consolidated in
[`PAPER_FINDINGS.md`](./PAPER_FINDINGS.md); this README documents the codebase
structure and the experimental protocol.

---

## Problem statement

In retail demand forecasting on perishable goods, observed sales `S_obs(t)` are
censored by stockouts:

```
S_obs(t) = min(D(t), I(t))
```

Standard forecasters trained on `S_obs` learn to predict zero during stockouts,
producing a systematic underestimation of latent demand `D(t)`. The community
addresses this with a *two-stage* pipeline: first impute the censored values,
then forecast from the completed signal.

**Question.** Does the choice of imputer materially affect forecasting accuracy,
and if so under which conditions?

The benchmark answers this with five Research Questions structured as
existence → mechanism → identification → conditions → boundary
(see [`PAPER_FINDINGS.md`](./PAPER_FINDINGS.md) §1–3).

---

## Dataset

[FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
(Liu et al., 2025, arXiv:2505.16319): 50 000 store × product series, 90 train
days + 7 test days at hourly granularity, 18 Chinese cities, 863 perishable
SKUs.

Operational hours 6–22 (17 hours/day). Temporal split:

- train: days 1–83
- val: days 84–90 (early stopping, HP selection)
- test: days 91–97 (final evaluation)

The dataset itself is not committed (see `.gitignore`).

---

## Experimental design

**Matrix.** 14 imputers × 8 forecaster families → 113 active cells
(some forecaster × imputer combinations were not feasible; see Section 3 of
`PAPER_FINDINGS.md` for exclusions).

| Family | Members |
|---|---|
| Naive imputers | Global mean, conditional mean, global median, conditional median |
| Classical TS imputers | Forward fill, seasonal naive, linear interp |
| ML / DL imputers | LGB imputer, DLinear, SAITS, iTransformer, TimesNet, ImputeFormer, CSDI |
| Naive forecasters | Global Mean, DoW Mean, **MA (K=56)** |
| ML forecasters | LGB (M5 lags) |
| Deep learning | MLP (M5 lags), TFT (Temporal Fusion Transformer) |
| Foundation models | Chronos-bolt, TimesFM 2.5 |

The MA window `K=56` is selected on validation under the per-series median
WAPE criterion with `min_hours=34`, the same objective used by Optuna HPO for
the ML/DL forecasters (see *Coherence note* at the top of `PAPER_FINDINGS.md`).

**Loss uniformity.** All ML/DL forecasters are trained with MAE loss to align
training objective and evaluation metric (WAPE = volume-weighted MAE).

**HPO.** TPE + MedianPruner (Optuna), objective = `val_WAPE_med` per-series
on in-stock hours (min 34 hours). Best hyperparameters serialized in
`pipeline/results/hpo_*_best.json`.

**Evaluation.** WAPE and WPE on in-stock hours of the test horizon, both
pooled and as per-series median over the 49 939 series that admit a finite
WAPE (61 / 50 000 series have zero total in-stock sales over the 7 test days
and are dropped from the paired analysis).

**Statistical framework.** Friedman χ² + Kendall’s W + Nemenyi CD post-hoc
(Demšar 2006, JMLR). Effect size via Cliff’s δ (descriptive, not as decision
rule). Stratification: Q1–Q4 of the per-series volume distribution.

---

## Repository layout

```
.
├── README.md                ← this file
├── CLAUDE.md                ← engineering log + project history (working notes)
├── PAPER_FINDINGS.md        ← findings of record for the paper
├── STUDY_REPORT.md          ← internal report (PINN-Retail legacy + pivot)
├── build_presentation.py    ← assembles the .pptx presentation
├── data/                    ← raw FreshRetailNet parquets (gitignored)
├── pipeline/
│   ├── 01–06_*.py           ← Phase A: baseline forecasters (no imputation)
│   ├── 04–18_*.py           ← Phase B1: imputers
│   ├── 06–32_*.py           ← Phase B2: forecaster × imputer cells
│   ├── 25_tft_full_training.py
│   ├── 26_chronos_bolt_*    ← Chronos-bolt zero-shot scoring
│   ├── 30–33_timesfm_*      ← TimesFM cells
│   ├── 35–39_*.py           ← matrix aggregation, Pareto frontiers, RQ4 crossover
│   ├── 41–49_*.py           ← Friedman + Kendall W + Nemenyi CD analyses, RQ1–RQ4 stratified
│   ├── ma_k56_retrain.py    ← MA forecaster re-train under HPO-coherent K
│   ├── ma_k_reselect_median.py ← MA grid search under both criteria
│   ├── aggregate_robustness.py ← seed sensitivity + recursive vs frozen lags
│   ├── build_nv_ref.py      ← reference completed demand y*(s,d) for newsvendor eval
│   ├── results/             ← per-series parquets, aggregate parquets (gitignored)
│   └── figures/             ← PNG outputs (gitignored)
├── notebooks/
├── notebooks_622/           ← exploration restricted to operational hours 6-22
├── notebooks_final/         ← consolidated baselines (PINN-Retail era)
├── src/                     ← reusable modules (data, models, losses, metrics)
└── 4_chronos2_forecasting.py
```

History note. An earlier line of work (sections 1–5 of `CLAUDE.md`) explored a
physics-informed neural network (PINN-Retail) for end-to-end censored-demand
forecasting. The current paper pivoted to the benchmark above; the PINN code
remains under `notebooks_final/` and `src/` for completeness.

---

## Reproducibility

The full re-run is staged in `pipeline/` and numbered. The most relevant
endpoints are:

1. **Imputation.** Scripts `04`, `05`, `14`, `16`, `18`, `27`, `28`, `31` (one
   per imputer family). Each writes `data/completed_sales_622/<imputer>.parquet`
   and a per-series recovery WAPE.
2. **Forecasting cells.** Scripts `06–32` (one per forecaster family or single
   cell). Each writes `pipeline/results/<imputer>__<forecaster>_test_per_series.parquet`.
3. **Matrix aggregation.** `35_pareto_analysis_hpo.py` consolidates the 113
   cells into `hpo_matrix_pareto.parquet` and the global Pareto frontier.
4. **Statistical analyses.** `45/46/47/48/49_*.py` for Friedman + Kendall W +
   Nemenyi CD at global / per-quartile / per-forecaster scope. `42_*.py` for
   the recovery → forecasting Spearman / pairwise concordance analyses.
5. **Figures.** Slide-ready 16:9 PNGs are produced by `35f`, `49b`, `42c2b`,
   etc., directly from the aggregate parquets.

Python environments used during development are kept under `freshnet/`,
`freshnet_timesfm/`, `chronos_env/` (gitignored). Key versions: Python 3.9,
PyTorch 2.x, PyTorch Lightning, PyPOTS (DLinear, SAITS, iTransformer, TimesNet,
ImputeFormer, CSDI), LightGBM, statsmodels, scikit-posthocs.

---

## Status

- **Matrix**: 113 active cells.
- **Framework**: Friedman + Kendall’s W + Nemenyi CD.
- **Global best**: `itransformer__MLP_M5` (Kendall W = 0.459 moderate, CD = 0.903, equivalence set 2 cells).
- **Best per quartile**: `lgb__MLP_M5` (Q1, Q2), `itransformer__MLP_M5` (Q3, Q4).
- **Global Pareto**: 25 / 113 cells optimal; per-quartile ranges from 27 (Q1) to 9 (Q4).
- **Recovery → forecasting**: naive aggregators large effect; foundation models
  small/medium; ML+lag negligible (lag features absorb the imputer signal).

All findings consolidated in `PAPER_FINDINGS.md`; that file is the source of
truth for the paper draft.

---

## License & citation

Internal research code, no public license. If you build on this work, please
cite the FreshRetailNet-50K dataset paper (Liu et al., 2025) and contact the
maintainer regarding citation of this benchmark.
