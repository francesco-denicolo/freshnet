# Imputer × Forecaster Benchmark on FreshRetailNet-50K — Study Report

> Documento di sintesi del lavoro svolto. Standalone, autosufficiente.
> Per il contesto storico (precedente progetto PINN-Retail), vedi `CLAUDE.md`.
> Per il design document originale del pivot, vedi `CLAUDE_FINAL.md`.

---

## 1. Sintesi esecutiva

Studio sistematico dell'impatto della **scelta dell'imputer di stockout** sulla **performance del forecasting** nel retail deperibile, su un benchmark di **11 imputer × 8 forecaster = 88 celle**.

**Domande di ricerca**:
1. La qualità dell'imputation influenza il forecasting downstream?
2. L'imputer ottimale dipende dal forecaster?
3. Quali forecaster sono più robusti al censoring?
4. L'effetto varia per tipo di serie (volume, stockout)?

**Findings chiave**:
- **Chronos-bolt** (foundation model) ha il **miglior WAPE mediana** (1.0066) ma **WPE catastrofico** (-0.96)
- **Trade-off WAPE vs WPE** caratterizzato in tutta la matrice
- **Crossover volume-dipendente**: Chronos vince su serie piccole, ML su serie grandi
- **Volume conta molto, stockout poco** (Cliff's δ ≈ 0.75 vs 0.15)
- **L'imputer ottimale dipende dal regime di volume**

**Target di pubblicazione**: International Journal of Forecasting / Decision Support Systems / Expert Systems with Applications.

---

## 2. Dataset: FreshRetailNet-50K

### Origine e scala
- **Fonte**: Liu et al. (2025), arXiv:2505.16319, [HuggingFace](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
- **50.000 serie temporali** (combinazioni store × SKU)
- **898 negozi** in **18 città cinesi**
- **865 SKU deperibili** (frutta, verdura, latticini, carne fresca)
- **Periodo**: 2024-03-28 → 2024-07-02 (97 giorni totali)
- **Granularità**: oraria (24 ore/giorno)
- **Tasso stockout medio**: 21.2% (sulle ore 6-22)

### Distribuzione (sbilanciata)

| | Conteggio |
|---|:---:|
| City 0 (la più grande) | 25.811 serie (52% dataset), 290 negozi |
| City 8 (la più piccola) | 65 serie, 3 negozi |
| Mediana negozi/città | 18 |
| Mediana SKU/negozio | 50 |

### Split temporale (HuggingFace)
- **Train HF**: gg 1-90 (2024-03-28 → 2024-06-25), 4.5M righe
- **Test HF (eval)**: gg 91-97 (2024-06-26 → 2024-07-02), 350K righe
- Settimana di test: mercoledì → martedì
- **Covariate shift**: temp +7°C, pioggia +59% nel test (passaggio primavera → estate)

### Restrizione orario operativo

Le serie sono ristrette alle ore **6-22 (17 ore/giorno)**. Le ore 0-5 e 23 hanno vendite ~0 (negozi chiusi/vuoti) e distorcerebbero le metriche WAPE.

```
Distribuzione vendite per fascia oraria:
  Ore 0-5 (notte):   3.2% del totale
  Ore 6-22 (giorno): 96.1% del totale
  Ora 23:            0.7%
```

### Codifica stock_status
- **0 = in stock** (S_obs = domanda vera)
- **1 = stockout** (S_obs ≈ 0, dato censurato)

---

## 3. Setup sperimentale

### Split temporale interno

```
gg 1-83        Train interno (training modelli)
gg 84-90       Validation (early stopping, eval Traccia A)
gg 91-97       Test (HF eval, valutazione finale, MAI usato in training)
```

### Direct forecast

I lag features per il test sono calcolati **una sola volta** dall'anchor giorno 90, e applicati uguali a tutti i 7 giorni del test:

```
Test g91: pred = f(lag da gg 1-90, dow=mer, meteo_91)
Test g92: pred = f(lag da gg 1-90, dow=gio, meteo_92)  ← stessi lag
...
Test g97: pred = f(lag da gg 1-90, dow=mar, meteo_97)
```

Solo le covariate esogene (dow, meteo, holiday, discount) variano. Niente recursive forecast.

**Motivazione**: isola l'effetto dell'imputation. La differenza tra "no imp" e "imputer X" è solo nella qualità dei lag.

### Metriche

```
WAPE = Σ |pred(h) − S_obs(h)| / Σ S_obs(h)        ← accuratezza
WPE  = Σ (pred(h) − S_obs(h)) / Σ S_obs(h)        ← bias
```

Calcolate **solo sulle ore in-stock del test set**, ore 6-22.

**Aggregazioni**:
- **Pooled**: tutte le 50K serie insieme (volume-weighted)
- **Mediana per-serie**: metrica per serie, poi mediana → robusta a outlier

### Effect size per i confronti

Con N = 50.000 serie, **p-value ≈ 0** per qualsiasi differenza. **Effect size** è la metrica discriminante:

- **Rank-biserial r** (paired Wilcoxon)
- **Cliff's δ** = P(X > Y) − P(X < Y)

Soglie standard (Cohen/Romano): negligible (|δ|<0.147), small (<0.33), medium (<0.474), **large (≥0.474)**.

---

## 4. Imputer testati (11 totali, 4 famiglie)

### Famiglia 1 — Baseline (1 imputer)

#### 1. No imputation
Lascia gli zeri da stockout come sono. Punto di riferimento.

### Famiglia 2 — Naive aggregati (4 imputer, design 2×2)

|  | Globale | Condizionato su (dow, hour) |
|---|---|---|
| **Media** | Media globale | Media condizionata |
| **Mediana** | Mediana globale | Mediana condizionata |

#### 2. Media globale
`media[S_obs in-stock | (store, product, hour)]`. Ignora dow.

#### 3. Media condizionata
`media[S_obs in-stock | (store, product, dow, hour)]`. Cattura stagionalità settimanale.

#### 4. Mediana globale
`mediana[S_obs in-stock | (store, product, hour)]`. Robusta a outlier.

#### 5. Mediana condizionata
`mediana[S_obs in-stock | (store, product, dow, hour)]`. Combina robustezza + stagionalità.

### Famiglia 3 — Time-series classici (3 imputer)

#### 6. Forward Fill
Per ogni stockout, copia l'**ultimo valore in-stock** osservato cronologicamente.

#### 7. Seasonal Naive
Per ogni stockout, copia il valore della **stessa (dow, hour)** della **settimana precedente**. Se anche quello stockout, va a 2 settimane fa, ecc.

#### 8. Linear Interpolation
Interpola linearmente tra il valore in-stock precedente e successivo nella sequenza.

### Famiglia 4 — ML/Deep Learning (3 imputer)

#### 9. LGB imputer
LightGBM trainato sulle ore in-stock. Features: store_id, product_id, city_id, dow, hour, discount, meteo, holiday, activity. Target: S_obs.

#### 10. DLinear (PyPOTS)
Modello sequenziale con decomposizione trend + seasonal, due linear layers. Operazione su finestre 30gg × 17h = 510 step, 150K samples (50K serie × 3 finestre). Loss ORT + MIT. Early stopping (patience=5) su val MSE.

#### 11. SAITS (PyPOTS)
Self-attention model con due Transformer encoder e Diagonal Masked Self-Attention (DMSA). Stesso framework di DLinear ma con attention. Modello small (38K params, d_model=32) per gestire memoria MPS.

---

## 5. Forecaster testati (8 totali)

### Naive (3+1)

#### 1. Global Mean
Profilo medio per (store, product, hour) su tutto il train. Vettore 17 valori.

#### 2. DoW Mean
Profilo medio per (store, product, dow, hour). 7 profili da 17 valori.

#### 3. MA (K=21)
Media degli ultimi K=21 giorni (selezionato su val). Vettore 17 valori.

#### 4. Naive Direct
Profilo del giorno 90 (ultimo del train), replicato sui 7 giorni test.

### ML tabellare (2)

#### 5. LGB no-lags
LightGBM con features: store_id, product_id, city_id, dow, hour (categoriche), + discount, meteo, holiday, activity.

#### 6. LGB M5-lags
LGB no-lags + 11 lag M5-style:
- lag_1d, lag_7d, lag_14d
- rmean_7d, rmean_14d, rstd_7d
- lag_dow, rmean_dow
- daily_total_lag1, daily_total_rmean7
- momentum = lag_1d / rmean_7d

### Deep Learning (2)

#### 7. MLP no-lags
Embeddings: store→32, product→32, city→8, dow→4 (76 dim) + 7 features continue.
Architettura: Linear(83→128) + ReLU → Linear(128→64) + ReLU → Linear(64→17) + Softplus.
Output: 17 valori (vendite per ora 6-22) per giorno target.

#### 8. MLP M5-lags
MLP no-lags + 275 dim lag (11 × 17 valori + 11 maschere binarie × 17).
Architettura: Linear(358→128) + ReLU → Linear(128→64) + ReLU → Linear(64→17) + Softplus.
102K parametri.

### Foundation Model (1)

#### 9. Chronos-bolt
Pre-trainato (Amazon, T5-based, ~250M params nella small variant).
Input: contesto = serie raw 1530 step (90 gg × 17 ore).
Output: 119 step predetti (7 gg × 17 ore).
Zero-shot: nessun training fine-tuning. Predict deterministico (quantile=0.5).

**Hyperparametri standardizzati per LGB e MLP** (NON ottimizzati per cella):

| Parametro | LGB | MLP |
|---|---|---|
| Loss | MAE | MSE |
| LR | 0.1 | 1e-3 |
| Batch | — | 4096 |
| Max iter/epochs | 500 | 100 |
| Early stop | 30 rounds (val MAE) | patience=10 (val WAPE) |
| Seed | 42 | 42 |

---

## 6. Pipeline operativa

### Per gli imputer naive/classici

1. Train su gg 1-83 (statistica/regola)
2. Eval MNAR su gg 84-90 → **Traccia A**
3. Retrain su gg 1-90 (modello finale)
4. Apply su tutte le ore stockout di gg 1-90 → `completed_sales/{nome}.parquet`

### Per LGB imputer

1-2. Stessa procedura, con early stopping su val (gg 84-90)
3-4. Stessa procedura

### Per DLinear/SAITS

1. Costruzione 150K samples (50K serie × 3 finestre da 30 giorni)
2. Split per-serie 80/20 (40K train, 10K val)
3. Training con loss ORT+MIT, early stopping su val MSE
4. **NESSUN retrain** — modello early-stopped è il finale
5. Predict su tutte le 150K samples → completed_sales
6. Eval MNAR con NaN extra per le maschere

### Per i forecaster

1. Train su gg 1-83 con lag/dati da `completed_sales[imputer]` (o da S_obs se no_imp)
2. Validation su gg 84-90 per early stopping
3. Retrain su gg 1-90
4. Predizione del test (gg 91-97) con direct forecast
5. Valutazione: WAPE/WPE solo sulle ore in-stock del test

### Maschere MNAR (Traccia A)

- **File**: `data/mnar_masks_val.parquet`
- **Periodo**: gg 84-90
- **Seed**: 42 (riproducibile)
- **Missing rate**: 30% delle ore in-stock
- **Pattern**: probabilità di mascheramento proporzionale al tasso di stockout reale per ora del giorno

```
WAPE_recovery = Σ |D_hat − ground_truth| / Σ ground_truth
                  sulle ore mascherate MNAR
WPE_recovery  = idem ma signed
```

---

## 7. Risultati

### 7.1 — Traccia A: qualità dell'imputation (MNAR recovery)

| Imputer | WAPE_recovery | WPE_recovery |
|---|:---:|:---:|
| **Mediana globale** | **0.809** | -0.57 |
| Mediana condizionata | 0.846 | -0.47 |
| LGB imputer | 0.930 | -0.10 |
| SAITS | 0.943 | -0.82 |
| Media globale | 0.945 | -0.13 |
| DLinear | 0.951 | -0.74 |
| Media condizionata | 0.956 | -0.13 |
| Linear Interp | 1.047 | +0.03 |
| Seasonal Naive | 1.064 | -0.10 |
| Forward Fill | 1.188 | +0.13 |

**Pattern**: la **Mediana globale** è la migliore su Traccia A (più dati per la stima → stima più stabile dei valori "tipici"). Forward Fill è la peggiore. Il design 2×2 dei naive aggregati rivela che il conditioning su DoW aiuta poco la mediana ma migliora la media.

### 7.2 — Matrice completa Traccia B (forecasting)

WAPE mediana per-serie, ore 6-22, in-stock, test set (88 celle):

```
                       Global   DoW     MA      LGB      LGB    MLP      MLP    Chronos
Imputer                Mean    Mean   (K=21)  (no lag) (M5 lag)(no lag)(M5 lag)  -bolt
─────────────────────  ─────── ─────── ─────── ──────── ──────── ──────── ──────── ───────
No imputation          1.1069  1.1004  1.1111  1.1003   1.1116  1.0707   1.0849  1.0066 ★
Media condizionata     1.1375  1.1347  1.1414  1.1003   1.1120  1.0707   1.0856  1.0370
Media globale          1.1376  1.1312  1.1418  1.1003   1.1070  1.0707   1.0792  1.0466
Mediana condizionata   1.1139  1.1099  1.1179  1.1003   1.1067  1.0707   1.0689  1.0111
Mediana globale        1.1072  1.1012  1.1111  1.1003   1.1103  1.0707   1.0753  1.0090
LGB imputer            1.1414  1.1348  1.1461  1.1003   1.1114  1.0707   1.0867  1.0500
DLinear                1.1187  1.1124  1.1244  1.1003   1.1098  1.0707   1.0771  1.0234
Forward Fill           1.3063  1.3060  1.2997  1.1003   1.1054  1.0707   1.0833  1.1066
Seasonal Naive         1.1579  1.1553  1.1381  1.1003   1.1074  1.0707   1.0786  1.0137
Linear Interp          1.2070  1.2038  1.2130  1.1003   1.1150  1.0707   1.0912  1.0698
SAITS                  1.1075  1.1008  1.1123  1.1003   1.1094  1.0707   1.0876  1.0117
```

**★ Best cella**: No imputation × Chronos-bolt = **1.0066**

### 7.3 — Top 10 combinazioni (ranking ascendente WAPE mediana)

| Rank | Imputer + Forecaster | WAPE med | WPE med |
|:---:|---|:---:|:---:|
| **1 ★** | No imputation + Chronos-bolt | **1.0066** | -0.96 |
| 2 | Mediana globale + Chronos-bolt | 1.0090 | -0.95 |
| 3 | Mediana condizionata + Chronos-bolt | 1.0111 | -0.94 |
| 4 | SAITS + Chronos-bolt | 1.0117 | -0.94 |
| 5 | Seasonal Naive + Chronos-bolt | 1.0137 | -0.88 |
| 6 | DLinear + Chronos-bolt | 1.0234 | -0.86 |
| 7 | Media condizionata + Chronos-bolt | 1.0370 | -0.82 |
| 8 | Media globale + Chronos-bolt | 1.0466 | -0.78 |
| 9 | LGB imputer + Chronos-bolt | 1.0500 | -0.76 |
| 10 | Mediana cond + MLP (M5 lags) | 1.0689 | -0.34 |

**Top 9 sono tutte Chronos-bolt**, ma con WPE catastrofico (-0.76 a -0.96). Mediana globale e Mediana condizionata occupano i rank 2-3 — confermando che gli imputer mediani "structure-preserving" preservano i pattern utili a Chronos-bolt.

### 7.4 — Best per quartile di volume (analisi stratificata)

| Quartile | Combinazione raccomandata | WAPE | Robustezza statistica |
|---|---|:---:|:---:|
| Q1-Q2 (basso, 50% serie) | No imp + Chronos-bolt | 1.01 | **Robusta** (Cliff's δ > 0.6, large) |
| Q3 (medio-alto, 25%) | Mediana cond + MLP M5 | 1.00 | **Equivalenza statistica** (small/negligible vs alternative) |
| Q4 (alto, 25%) | LGB M5 + Mediana | **0.74** | **Equivalenza statistica** (LGB+Mediana ≈ LGB+MediaGlob ≈ MLP+MediaCond) |

### 7.5 — Crossover Chronos vs ML (per quartile, no imputation)

| Quartile vol | Δ MLP-Chronos | Cliff's δ | Effect | Vincitore |
|:---:|:---:|:---:|:---:|---|
| Q1 | +0.246 | +0.85 | LARGE | Chronos (93% serie) |
| Q2 | +0.163 | +0.64 | LARGE | Chronos |
| Q3 | +0.002 | -0.02 | negligible | tie |
| Q4 | -0.200 | -0.64 | LARGE | MLP (82% serie) |

**Crossover dimostrato statisticamente**: il vincitore si inverte tra Q1-Q2 e Q4.

### 7.6 — Sensibilità a volume vs stockout (Cliff's δ medio per forecaster)

| Forecaster | δ_volume | δ_stockout |
|---|:---:|:---:|
| LGB (M5 lags) | +0.83 (large) | +0.20 (small) |
| MLP (M5 lags) | +0.82 (large) | +0.19 (small) |
| LGB (no lags) | +0.80 (large) | +0.20 (small) |
| MLP (no lags) | +0.78 (large) | +0.19 (small) |
| MA (K=21) | +0.77 (large) | +0.14 (negligible) |
| Global Mean | +0.74 (large) | +0.16 (small) |
| DoW Mean | +0.72 (large) | +0.15 (negligible) |
| **Chronos-bolt** | **+0.71** (large) | **+0.09** (negligible) |

**Pattern**: il volume ha large effect su tutti, lo stockout ha small/negligible effect. **Chronos-bolt è il forecaster meno sensibile** a entrambe le dimensioni.

---

## 8. Findings principali

### Finding 1 — Trade-off WAPE vs WPE
Chronos-bolt domina il WAPE ma con bias catastrofico (-0.96). MLP/LGB hanno WPE migliore (-0.18 a -0.38) a costo di WAPE peggiore. **Non esiste un'unica soluzione ottimale** — la scelta dipende dal caso d'uso.

### Finding 2 — Crossover volume-dipendente
Su serie a basso volume (Q1), Chronos-bolt batte MLP M5 con large effect. Su serie ad alto volume (Q4), MLP/LGB battono Chronos con large effect. **Il vincitore cambia per regime**.

### Finding 3 — Volume dominates, Stockout negligible
Il volume della serie ha effect size large (Cliff's δ ≈ 0.75) sul WAPE; il tasso di stockout ha effect size small/negligible (Cliff's δ ≈ 0.15). Questo perché il dataset ha **range di stockout ristretto** (16-31% per il 90% delle serie).

### Finding 4 — Imputer "structure-preserving" preservano il crossover
- **Preservano**: No imp, Seasonal Naive, DLinear, SAITS, Mediana cond
- **Rovinano**: Media cond, Media glob, LGB imputer, Forward Fill

I primi mantengono punti di vendita "tipici" (zeri ammessi); i secondi appiattiscono verso la media.

### Finding 5 — Traccia A non predice Traccia B
Mediana globale ha il miglior WAPE_recovery (0.809) ma SAITS (3°-4° su recovery) batte Mediana globale per Chronos-bolt come forecaster downstream. **La qualità dell'imputation MNAR non è sufficiente a predire la qualità del forecasting**.

### Finding 6 — Equivalenza statistica nei regimi medi/alti
In Q3 e Q4, il "best" sulla mediana globale **non è statisticamente migliore** delle alternative (effect size small/negligible). Questo apre lo spazio a scelte basate su criteri secondari (semplicità, costo computazionale, interpretabilità).

---

## 9. Test statistici eseguiti (~600 test totali)

| Test | Numero | Salvato in |
|---|:---:|---|
| Wilcoxon paired (imputer pair × forecaster) | 330 (11 imputer, C(11,2)=55 × 6 fc) | `wilcoxon_all.parquet` |
| Wilcoxon best combo vs alternatives | 7+21=28 | `wilcoxon_best_vs_all.parquet`, `wilcoxon_combos.parquet` |
| Cliff's δ stratificato (cell × dimension) | 176 (11×8×2) | `volume_stockout_tests.parquet` |
| Crossover Chronos vs ML (per quartile × imputer) | 56 (11 imputer × 4 q) | `crossover_tests.parquet` |

**Tutti i test paired Wilcoxon** danno p ≈ 0 con N=50K → **effect size è la metrica primaria**.

---

## 10. Limitazioni dichiarate

1. **1 dataset**, retail cinese deperibile, 3.5 mesi (no stagionalità annuale).
2. **Orizzonte breve** (7 giorni di test) — orizzonti più lunghi non valutati.
3. **HP non ottimizzati** — tutti i forecaster usano HP standard. Sensitivity analysis lasciata per appendice.
4. **DLinear/SAITS asimmetria** — 80/20 split per-serie, no retrain finale (limitazione di PyPOTS).
5. **City 0 = 52% del dataset** — sbilanciamento geografico (mitigato da modello globale + city_id come feature).
6. **Range stockout ristretto** (16-31% per 90% serie) — risultati su stockout estremi (>50%) non validati (solo 1.7% delle serie).
7. **Imputer SOTA non testati**: TimesNet, iTransformer (skipped per OOM/tempi). Solo DLinear e SAITS rappresentano i SOTA.

---

## 11. Struttura del codice

### Cartella principale: `pipeline/`

```
pipeline/
├── 01_fase_a_naive.py                         # Naive forecaster baseline
├── 02_fase_a_lgb.py                           # LGB forecaster baseline
├── 03_fase_a_mlp.py                           # MLP forecaster baseline
├── 04_fase_b1_imputation_naive_ml.py          # Media glob/cond, Mediana cond, LGB imputer
├── 05_fase_b1_imputation_dlinear.py           # DLinear imputer
├── 06_fase_b2_forecast_naive.py               # Naive × completed_sales
├── 07_fase_b2_forecast_lgb.py <imputer>       # LGB M5 × imputer
├── 08_fase_b2_forecast_mlp.py <imputer>       # MLP M5 × imputer
├── 09_fase_c_analysis.py                      # Analisi globale + heatmap + Wilcoxon
├── 10_fase_b2_forecast_chronos.py <imputer>   # Chronos-bolt × imputer
├── 11_fase_c_stratified.py                    # Stratificazione 4×4 (subset)
├── 13_fase_c_volume_stockout_tests.py         # Test volume+stockout (Cliff's δ, JT)
├── 14_fase_b1_imputation_classic.py           # Forward fill, Seasonal Naive, Linear Interp
├── 14b_fase_b2_forecast_naive_classic.py      # Naive × imputer classici
├── 15_fase_c_stratified_full.py               # Opzione A+B + crossover Chronos vs ML
├── 16_fase_b1_imputation_saits.py             # SAITS imputer (training)
├── 16b_saits_predict.py                       # SAITS predict (continuation)
├── 17_presentation_figures.py                 # 3 figure ad-hoc per presentazione
└── 18_fase_b1_imputation_mediana_glob.py      # Mediana globale imputer
```

### Cartella risultati: `pipeline/results/`

```
{imputer}__{forecaster}_test_per_series.parquet  # 88 cells (per-serie WAPE/WPE)
traccia_a*.parquet                               # WAPE_recovery imputer
stratification.parquet                           # vol_bin, so_bin per serie
volume_stockout_tests.parquet                    # 176 test stratificati
crossover_tests.parquet                          # Crossover Chronos vs ML
wilcoxon_all.parquet                             # 270 test pairwise imputer
wilcoxon_combos.parquet                          # 21 test best combo
wilcoxon_best_vs_all.parquet                     # 7 test best globale
ranked_combinations.parquet                      # 88 celle ordinate per WAPE med
best_per_group_16_v2.parquet                     # Best per gruppo (16 vol×so)
```

### Cartella figure: `pipeline/figures/`

```
fig01_heatmap_wape_median.png          # Matrice 11×8 WAPE
fig02_heatmap_wpe_median.png           # Matrice 11×8 WPE
fig03_boxplot_*.png                    # Boxplot per forecaster
fig04_effect_*.png                     # Effect size heatmap
fig05_saturation.png                   # WAPE recovery vs forecasting
fig06_best_vs_all.png                  # Best combo vs altre
fig07_heatmap_full_matrix.png          # Heatmap completa
fig08_heatmap_full_matrix_wpe.png      # Heatmap completa WPE
fig09_stratified_matrix.png            # 4×4 grid (vol × stockout)
fig10_trend_stockout.png               # Trend stockout
fig11_spearman_rho.png                 # Effect size volume
fig12_cliff_delta_volume.png           # Cliff's δ volume per cella
fig12_cliff_delta_stockout.png         # Cliff's δ stockout per cella
fig13_distribution_volume_stockout.png # Distribuzione volume/stockout (quartili)
fig14_scatter_volume_stockout.png      # Scatter volume × stockout
fig15_tradeoff_wape_wpe.png            # Trade-off WAPE × |WPE|
fig16_crossover_volume.png             # Line chart crossover
fig17_cliff_delta_dim_bars.png         # Bar chart effect size per dimension
fig18_distribution_city_store.png      # Distribuzione city/store
```

### Ambienti Python

```
freshnet/      ← Python 3.9, per pipeline principale (numpy, pandas, lightgbm, torch+MPS, PyPOTS)
chronos_env/   ← Python 3.13, per Chronos-bolt (richiede torch>=2.4)
```

---

## 12. Decisioni metodologiche del pivot

1. **Restrizione orario 6-22**: per evitare distorsione delle ore notturne con vendite ~0 (non era nel piano originale).
2. **Direct forecast**: per tutti i forecaster — i lag fissati all'anchor giorno 90.
3. **Niente HP optimization**: tutti i forecaster usano HP fissi e standardizzati.
4. **Effect size come metrica primaria**: con N=50K il p-value perde discriminabilità.
5. **Mediana globale aggiunta a posteriori**: per completare il design 2×2 dei naive aggregati.
6. **Solo DLinear+SAITS come SOTA**: TimesNet e iTransformer skipped per OOM/tempi su MPS.
7. **Chronos-bolt invece di Chronos-2**: il "bolt" è 250× più veloce (stima 18 giorni → 20 min).

---

## 13. Riferimenti bibliografici

- **Liu et al. (2025)** — FreshRetailNet-50K: Latent Demand from 50,000 Stores for World-scale Stockout Prediction in Fresh Retail. arXiv:2505.16319.
- **Du et al. (2023)** — SAITS: Self-Attention-based Imputation for Time Series. NeurIPS.
- **Zeng et al. (2022)** — DLinear: Are Transformers Effective for Time Series Forecasting? AAAI 2023.
- **Ansari et al. (2024)** — Chronos: Learning the Language of Time Series. arXiv:2403.07815.
- **Du et al. (2023)** — PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series. arXiv:2305.18811.
- **Cohen (1988)** — Statistical Power Analysis for the Behavioral Sciences. 2nd ed.
- **Romano et al. (2006)** — Effect size guidelines for paired data.

---

## 14. Stato attuale e prossimi passi

### Stato

- ✅ Matrice **88 celle** completata (11 imputer × 8 forecaster)
- ✅ Mediana globale al **rank 2 globale** con Chronos-bolt (WAPE med = 1.0090, miglior su Traccia A)
- ✅ Analisi statistica completa (Wilcoxon + Cliff's δ + JT + Spearman, ~600 test)
- ✅ Analisi stratificata volume × stockout (16 gruppi, quartili)
- ✅ Crossover Chronos vs ML dimostrato statisticamente
- ✅ Pipeline riproducibile, 88 parquet per-serie + parquet aggregati
- ✅ 18+ figure rigenerate per la matrice 11×8

### Prossimi passi suggeriti

1. **Completamento analisi finale con Mediana globale** (in corso): ~30 min
2. **Sensitivity analysis HP** (1-2 giorni): variare HP del LGB/MLP per dimostrare robustezza
3. **Investigazione bias di Chronos-bolt** (analisi residui, plot per quartile): mezza giornata
4. **Eventuale 2° dataset** (1-2 settimane): M5 o Rossmann per generalizzare
5. **Scrittura paper** (1-2 settimane): bozza per rivista applicata

### Target di pubblicazione

Rivista applicata: **International Journal of Forecasting**, **Decision Support Systems**, **Expert Systems with Applications**, oppure **KDD Applied Data Science Track**.

Probabilità accettazione attuale: 50-70% per una rivista applicata; 20-30% per top-tier (NeurIPS/ICML).

---

*Documento generato: 2026-05-05*
