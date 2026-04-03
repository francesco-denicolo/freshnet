# Impatto della Qualità dell'Imputation sul Demand Forecasting di Prodotti Deperibili

## Obiettivo del paper

Studiare sistematicamente come la qualità dell'imputation delle vendite censurate
da stockout influenza le performance dei modelli di demand forecasting.

**Domanda di ricerca principale:** dato un dataset di vendite retail con ~25% di ore
censurate da stockout, e dati diversi metodi di imputation (da naive a stato dell'arte),
come cambia la performance del forecasting downstream?

**Domande secondarie:**
- Esiste una soglia di qualità dell'imputation oltre la quale il forecasting non migliora?
- L'imputer migliore dipende dal forecaster, o ce n'è uno universalmente superiore?
- Quali forecaster sono più robusti al censoring (meno sensibili all'imputation)?
- L'effetto dell'imputation varia per tipo di serie (alto vs basso volume, alto vs basso stockout)?

**Target:** International Journal of Forecasting / Expert Systems with Applications /
KDD Applied Data Science Track.

---

## Dataset: FreshRetailNet-50K

- **Fonte**: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
- **Paper**: arxiv 2505.16319, codice: https://github.com/Dingdong-Inc/frn-50k-baseline
- **Train HuggingFace**: 4.500.000 righe = 50.000 serie × 90 giorni
- **Eval HuggingFace**: 350.000 righe = 50.000 serie × 7 giorni (subito dopo il train)
- **Totale**: 97 giorni di dati (90 train HF + 7 eval HF)
- **Granularità**: oraria (24 ore per giorno)
- **Copertura**: 898 negozi, 18 città cinesi, 863 SKU deperibili
- **Tasso di stockout**: ~25% delle ore sono in stockout

### Colonne del dataset

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| city_id | int | ID città |
| store_id | int | ID negozio |
| management_group_id | int | Gruppo gestione |
| first_category_id | int | Categoria livello 1 |
| second_category_id | int | Categoria livello 2 |
| third_category_id | int | Categoria livello 3 |
| product_id | int | ID prodotto |
| dt | str | Data giornaliera |
| sale_amount | float | Vendite giornaliere totali |
| hours_sale | array[24] | Vendite per ora (0-23) |
| stock_hour6_22_cnt | int | Ore di stockout tra 6:00 e 22:00 |
| hours_stock_status | array[24] | Stato stock binario: 0=in stock, 1=stockout |
| discount | float | Sconto promozionale |
| holiday_flag | int | Flag festività |
| activity_flag | int | Flag attività promozionale |
| precpt | float | Precipitazioni |
| avg_temperature | float | Temperatura media |
| avg_humidity | float | Umidità media |
| avg_wind_level | float | Livello vento medio |

### Codifica stock_status

- **0 = in stock**: prodotto disponibile, S_obs = domanda vera
- **1 = stockout**: prodotto esaurito, S_obs ≈ 0, dato censurato

---

## Train / Validation / Test Split

### Struttura

```
|------------ Train HuggingFace: 90 giorni ------------|-- Eval HF: 7 gg --|
|--- Train interno (gg 1-83) ---|--- Val (gg 84-90) ---|--- Test (eval) ---|
         83 giorni                    7 giorni               7 giorni
```

**Train: giorni 1-83** del train HF.
Per allenare i modelli (sia imputer sia forecaster) durante il tuning.

**Validation: giorni 84-90** del train HF.
Per early stopping, hyperparameter tuning, selezione del miglior imputer,
e valutazione della qualità dell'imputation (Traccia A con maschere MNAR).

**Test: eval set HF** (7 giorni, subito dopo il giorno 90).
Per la valutazione finale del forecasting (Traccia B).
Mai usato durante il training di nessun modello.

**Retraining: giorni 1-90** (tutti i 90 giorni del train HF).
Dopo il tuning, i modelli finali vengono riallenati su tutti i 90 giorni
prima di fare le previsioni sul test set.

### Principio fondamentale: separazione imputer / forecaster / test

```
IMPUTER:      opera SOLO su gg 1-90 (train HF).
              Non vede mai il test set. Non produce nulla per il test set.
              Output: completed_sales per i gg 1-90.

FORECASTER:   usa completed_sales (gg 1-90) come storico per calcolare i lag.
              Produce previsioni per il test set (eval HF).
              Viene valutato sulle ore in-stock del test set.

TEST SET:     toccato SOLO dal forecaster.
              Solo in output (previsioni) e in valutazione (confronto con S_obs).
```

### Approccio di forecasting: Direct Forecast

I forecaster usano **solo lag storici** (calcolati dai gg 1-90) per prevedere
il test set. **Nessun dato del test set entra mai come input del forecaster.**
Se il test set ha 7 giorni, il giorno 7 viene predetto con gli stessi lag
del giorno 1 (quelli dai gg 1-90), non con dati dei giorni 1-6 del test.

Questo garantisce:
- Nessuna contaminazione da stockout nel test set.
- Confronto pulito tra imputer (l'unica differenza è la qualità dei completed_sales storici).
- Valutazione indipendente dal comportamento del modello durante il test.

---

## Metriche di valutazione

### Regola unica: solo ore in-stock, ground truth = S_obs

La valutazione del forecasting avviene **SEMPRE e SOLO sulle ore in-stock**
(stock_status=0) sia a livello orario sia a livello giornaliero.
Il ground truth è **SEMPRE S_obs** (valore esatto, perché nelle ore in-stock S_obs = D).

Questo garantisce:
- Ground truth esatto (non approssimato da imputation).
- Coerenza tra livello orario e giornaliero.
- Nessuna dipendenza dall'imputer nella valutazione.

### Livello orario (metrica primaria del paper)

```
WAPE_orario = Σ |pred(h) - S_obs(h)| / Σ S_obs(h)
              per tutte le ore h in-stock del periodo di valutazione

WPE_orario  = Σ (pred(h) - S_obs(h)) / Σ S_obs(h)
              per tutte le ore h in-stock del periodo di valutazione
```

WPE negativo = sottostima (effetto tipico del censoring).
WPE positivo = sovrastima.
WPE ≈ 0 = nessun bias sistematico.

### Livello giornaliero (metrica secondaria, per confronto con il paper baseline)

Per ogni giorno d del periodo di valutazione:
```
pred_d = Σ pred(h)   per le ore h in-stock di quel giorno
gt_d   = Σ S_obs(h)  per le stesse ore h in-stock di quel giorno
```

Poi:
```
WAPE_daily = Σ_d |pred_d - gt_d| / Σ_d gt_d
WPE_daily  = Σ_d (pred_d - gt_d) / Σ_d gt_d
```

Nota: giorni diversi possono avere un numero diverso di ore in-stock.
Il WAPE come rapporto gestisce questo naturalmente.

### Metriche per l'imputation (Traccia A)

Calcolate sulle ore mascherate artificialmente (dove conosci il ground truth):
```
WAPE_recovery = Σ |D_hat(t) - ground_truth(t)| / Σ ground_truth(t)
WPE_recovery  = Σ (D_hat(t) - ground_truth(t)) / Σ ground_truth(t)
```

### Aggregazione

Per ogni metrica, riportiamo:
- **Pooled**: aggregando tutte le serie (volume-weighted).
- **Mediana per-serie**: calcolata per serie, poi mediana (peso uguale per ogni serie).

---

## Pipeline sperimentale

La pipeline ha 4 fasi sequenziali: EDA, Fase A, Fase B, Fase C.

---

### FASE 0 — Esplorazione e analisi dei dati (EDA)

**Obiettivo:** comprendere il dataset, identificare pattern, anomalie, e informare
le decisioni di modellazione.

**Analisi da condurre:**

*Struttura dei dati:*
- Parsing dei campi orari (hours_sale, hours_stock_status): verificare formato,
  coerenza con sale_amount e stock_hour6_22_cnt.
- Distribuzione delle serie per negozio, città, categoria merceologica.
- Copertura temporale: verificare che tutte le 50.000 serie coprano tutti i 90 giorni.
- Valori mancanti o anomali nelle covariate.

*Analisi del censoring:*
- Tasso di stockout per ora del giorno (profilo orario del censoring).
- Tasso di stockout per giorno della settimana.
- Tasso di stockout per categoria merceologica.
- Distribuzione della durata degli stockout (ore consecutive).
- Percentuale di giorni-serie con almeno un'ora di stockout.
- Percentuale di giorni-serie con stockout completo (24h).
- Coerenza vendite/stockout: verificare che S_obs ≈ 0 durante stockout.

*Pattern di domanda:*
- Profilo orario medio delle vendite (per ore in-stock).
- Differenze weekday vs weekend.
- Effetto delle promozioni (discount) sulle vendite.
- Effetto del meteo (temperatura, pioggia) sulle vendite.
- Effetto delle festività.
- Distribuzione delle vendite per serie (power law? long tail?).

*Analisi del test set:*
- Confronto distribuzioni train vs eval (covariate shift).
- Tasso di stockout nel test set vs train set.

*Analisi per la segmentazione:*
- Classificazione delle serie per volume (alto/medio/basso).
- Classificazione delle serie per tasso di stockout (alto/medio/basso).
- Cross-tabulation volume × tasso di stockout.

**Output:**
- Notebook con visualizzazioni.
- Statistiche descrittive documentate.
- Decisioni informate per le fasi successive (es. scelta dei lag, scelta delle feature).

---

### FASE A — Forecast senza imputation (baseline)

**Obiettivo:** stabilire le performance baseline su dati sporchi (con zeri da stockout).
Questi risultati formano la prima riga della matrice dei risultati.

#### Punto (A1): Baseline naive su dati sporchi

**Modelli:**
- Global Mean: media di S_obs(h) per (store, product, hour) su tutto il train.
- DoW Mean: media di S_obs(h) per (store, product, day_of_week, hour) su tutto il train.
- Naive Direct: profilo S_obs dell'ultimo giorno di train, applicato a tutti i giorni di test.
- MA (K giorni): media di S_obs(h) sugli ultimi K giorni. K selezionato su val.

**Input:** S_obs storiche (con zeri da stockout).
**Output:** previsione per ogni ora h=0..23 per ogni giorno del test set.
**Valutazione:** WAPE e WPE sulle ore in-stock del test set.
**Split:** profili calcolati su gg 1-90 → test eval HF. (MA seleziona K su gg 84-90.)

#### Punto (A2): ML e DL su dati sporchi

**Modelli:**
- LightGBM (con feature tabellari + lag M5-style da S_obs)
- MLP (con embedding + feature continue + lag M5-style da S_obs)
- Foundation models (es. Chronos-2, se applicabile)

**Input per ogni campione:**
- Embedding categoriali: store_id, product_id, city_id, category L1/L2/L3, dow
- Covariate continue: discount, temperatura, umidità, precipitazioni, vento, holiday, activity
- Lag features M5-style calcolati da **S_obs grezzo** (contaminato dagli zeri):
  lag_1d, lag_7d, lag_14d, lag_21d, rolling_mean_7d, rolling_mean_14d,
  rolling_mean_dow, rolling_std_7d, daily_agg_lag, momentum_1d_7d, ecc.

**Target:** S_obs(h) per h=0..23 del giorno target.
**Valutazione:** WAPE e WPE sulle ore in-stock del test set.
**Split:** train gg 1-83, val gg 84-90 (tuning), retrain gg 1-90, test eval HF.
**Lag per il test:** calcolati da S_obs dei gg 1-90. Direct forecast, nessun dato di test come input.

---

### FASE B — Imputation + Forecast (two-stage)

**Obiettivo:** imputare le ore censurate con diversi metodi, poi allenare i forecaster
sui dati puliti. Le righe successive della matrice dei risultati.

#### Punto (B1): Imputation

**Cosa fa l'imputation:**
Per ogni ora di ogni serie, produce:
```
completed_sales(t) = S_obs(t)   se stock_status(t) = 0  (in-stock, affidabile)
                     D_hat(t)   se stock_status(t) = 1  (stockout, valore imputato)
```

**Candidati:**

Famiglia 1 — Naive:
- Media condizionata: per ogni (store, product, dow, hour), media di S_obs
  sulle ore in-stock del train.
- Media globale: per ogni (store, product, hour), media su tutti i giorni in-stock.
- Mediana condizionata: come media condizionata ma con la mediana.

Famiglia 2 — ML:
- LGB imputer: LightGBM allenato sulle ore in-stock del train.
  Feature: store_id, product_id, ora, dow, discount, meteo, holiday, activity.
  Target: S_obs.

Famiglia 3 — Stato dell'arte (sequenziale, bidirezionale):
- TimesNet, SAITS, iTransformer, DLinear tramite PyPOTS.
  Operano su sequenze con NaN nelle ore di stockout.
  Allenati con strategia ORT+MIT (Observed Reconstruction + Masked Imputation Task).
  Codice: https://github.com/Dingdong-Inc/frn-50k-baseline

**Valutazione dell'imputation: maschere MNAR**

Non puoi valutare l'imputation sulle ore di stockout reale (ground truth ignoto).
Soluzione: maschera artificialmente ore in-stock del validation (dove conosci il valore vero),
chiedi al modello di ricostruirle, confronta con il ground truth.

Le maschere servono per valutare uniformemente tutte le famiglie di imputer
(naive, ML, stato dell'arte) con la stessa metrica sulle stesse ore.

Generazione maschere (una sola volta, mai rigenerarle):
```python
# File: data/mnar_masks_val.parquet
# Periodo: gg 84-90 (validation)
# Seed: 42
# Missing rate: 30% delle ore in-stock
# Pattern MNAR: probabilità di mascheramento proporzionale alla distribuzione
#   empirica degli stockout reali per ora del giorno
#   (più maschere ore pomeridiane/serali, meno ore notturne)
# Colonne: store_id, product_id, dt, hour, is_masked, ground_truth
```

**Procedura per OGNI imputer candidato:**

```
Passo 1.  Allena l'imputer sui gg 1-83.
Passo 2.  Applica alle ore mascherate dei gg 84-90 (maschere MNAR, seed=42).
Passo 3.  Calcola WAPE_recovery e WPE_recovery sulle ore mascherate.
          → Questi numeri formano la TABELLA TRACCIA A del paper.
Passo 4.  Riallena l'imputer su gg 1-90 (tutti i 90 giorni).
Passo 5.  Applica a TUTTE le ore di stockout dei gg 1-90.
Passo 6.  Salva: data/completed_sales/<nome_imputer>.parquet
```

La procedura viene eseguita per OGNI candidato, non solo per il vincitore.
Servono i completed_sales di ogni imputer per costruire la matrice completa.

**Dettagli per famiglia:**

Naive: calcola le statistiche su gg 1-83 (ore in-stock). Applica ai gg 84-90 mascherati.
  Ricalcola su gg 1-90. Applica a tutte le ore di stockout dei gg 1-90.
LGB: allena su ore in-stock dei gg 1-83. Predici le ore mascherate dei gg 84-90.
  Riallena su gg 1-90. Predici tutte le ore di stockout dei gg 1-90.
PyPOTS: allena su gg 1-83 con ORT+MIT interna. Applica ai gg 84-90 con ore
  mascherate come NaN aggiuntivi. Il modello riempie i NaN.
  Riallena su gg 1-90. Applica a tutte le ore di stockout dei gg 1-90.

NOTA per PyPOTS: durante il training il modello genera le SUE maschere interne
(per la MIT loss). Le NOSTRE maschere (seed=42) servono SOLO per la valutazione,
non per il training. Le due maschere sono indipendenti.

**Output:**
- Tabella Traccia A: WAPE_recovery e WPE_recovery per ogni imputer
- data/completed_sales/<nome>.parquet per ogni imputer (calcolati su gg 1-90)

#### Punto (B2): Forecast su dati puliti — per OGNI imputer

Per ogni imputer I e per ogni forecaster F, allena F sui completed_sales di I
e valuta sul test set.

**Baseline naive su dati puliti (per ogni imputer I):**
Identici al punto (A1) ma calcolati su completed_sales(I).
Profili calcolati su completed_sales(I) dei gg 1-90 → test eval HF.

**ML e DL su dati puliti (per ogni imputer I):**
Identici al punto (A2) ma i lag features sono calcolati da completed_sales(I).
Train gg 1-83 (lag da completed_sales(I)), val gg 84-90, retrain gg 1-90, test eval HF.
Lag per il test: calcolati da completed_sales(I) dei gg 1-90.
Direct forecast, nessun dato di test come input.

**Output:** una cella (WAPE, WPE) per ogni combinazione (imputer I, forecaster F).

---

### FASE C — Analisi dei risultati

**Obiettivo:** costruire la matrice imputation × forecasting e rispondere alle
domande di ricerca.

#### La matrice dei risultati

```
                      Forecaster 1   Forecaster 2   Forecaster 3   Forecaster 4
                      (LightGBM)     (MLP)          (Foundation)   (...)
                      WAPE | WPE     WAPE | WPE     WAPE | WPE     WAPE | WPE
                      
No imputation          ___   ___      ___   ___      ___   ___      ___   ___
(Fase A, dati sporchi)

Imputer 1              ___   ___      ___   ___      ___   ___      ___   ___
(Media condiz.)

Imputer 2              ___   ___      ___   ___      ___   ___      ___   ___
(LGB)

Imputer 3              ___   ___      ___   ___      ___   ___      ___   ___
(DLinear)

Imputer 4              ___   ___      ___   ___      ___   ___      ___   ___
(SAITS)

Imputer 5              ___   ___      ___   ___      ___   ___      ___   ___
(TimesNet)
```

La prima riga è la Fase A. Le righe successive sono la Fase B con diversi imputer.
Ogni cella contiene WAPE e WPE (sia pooled sia mediana per-serie).

A fianco della matrice, una colonna aggiuntiva con i risultati della Traccia A:
WAPE_recovery e WPE_recovery di ogni imputer (dal passo 3 della Fase B1).

#### Analisi da condurre

**Analisi 1 — Per colonna (effetto dell'imputation su un dato forecaster):**
Fissi il forecaster, varî l'imputer. Risponde a: "quanto migliora LightGBM
con l'imputation? E quale imputer è il migliore per LightGBM?"

**Analisi 2 — Per riga (effetto del forecaster a parità di imputation):**
Fissi l'imputer, varî il forecaster. Risponde a: "dati i dati ripuliti con
TimesNet, quale forecaster è il migliore?"

**Analisi 3 — Interazione imputer × forecaster:**
L'imputer migliore dipende dal forecaster, o c'è un imputer universalmente migliore?
Se la classifica degli imputer cambia per forecaster diversi, c'è interazione.

**Analisi 4 — Saturazione:**
Ordina gli imputer per WAPE_recovery crescente (dal peggiore al migliore
sulla Traccia A). Per ogni forecaster, grafica il WAPE del forecasting (asse Y)
vs la WAPE_recovery dell'imputer (asse X). Se la curva si appiattisce,
c'è un punto di saturazione.

**Analisi 5 — Robustezza al censoring:**
Per ogni forecaster, calcola ΔWAPE = WAPE(no imputation) - WAPE(miglior imputation).
Il forecaster con ΔWAPE più piccolo è il più robusto al censoring.

**Analisi 6 — Stratificazione per tipo di serie:**
Ripeti le analisi 1-5 separatamente per:
- Serie a basso stockout (<10%), medio (10-30%), alto (>30%)
- Serie a basso volume, medio, alto

**Analisi 7 — Test statistici:**
Per ogni confronto tra celle della matrice:
- Intervalli di confidenza bootstrap sulla differenza di WAPE/WPE.
- Test di Wilcoxon signed-rank (confronto paired per-serie).
- p-value per la significatività statistica.

---

## Ordine di esecuzione

### Fase 0 — EDA (Settimana 1)

```
Notebook: notebooks/01_eda.py
- Carica e parsa il dataset
- Statistiche descrittive complete
- Visualizzazioni dei pattern di vendita e stockout
- Analisi covariate
- Confronto train vs eval (covariate shift)
- Segmentazione serie per volume e tasso di stockout
Output: notebooks/figures/, statistiche documentate
```

### Fase 0b — Generazione maschere MNAR (dopo EDA)

```
Notebook: notebooks/02_generate_mnar_masks.py
Input: train HF (90 giorni)
Output: data/mnar_masks_val.parquet (gg 84-90, seed=42, 30% MNAR)
Nota: generare una sola volta, mai rigenerare.
```

### Fase A — Baseline senza imputation (Settimane 1-2)

```
Notebook: notebooks/03_baseline_naive.py
  → Global Mean, DoW Mean, Naive Direct, MA su dati sporchi
  → Valutazione su eval HF (ore in-stock)

Notebook: notebooks/04_baseline_ml.py
  → LightGBM, MLP su dati sporchi (con lag M5-style da S_obs)
  → Tuning su gg 1-83/84-90, retrain su 1-90, test su eval HF

Output: prima riga della matrice
```

### Fase B — Imputation + Forecast (Settimane 2-4)

```
Notebook: notebooks/05_imputation_naive_ml.py
  → Media condizionata, Media globale, Mediana, LGB imputer
  → Train su gg 1-83, valutazione su maschere MNAR val (gg 84-90)
  → Retrain su gg 1-90, produzione completed_sales per ciascuno

Notebook: notebooks/06_imputation_sota.py
  → TimesNet, SAITS, iTransformer, DLinear via PyPOTS
  → Train su gg 1-83, valutazione su maschere MNAR val (gg 84-90)
  → Retrain su gg 1-90, produzione completed_sales per ciascuno
  → Nota: richiede installazione PyPOTS e ambiente Python 3.8

Notebook: notebooks/07_traccia_a_results.py
  → Tabella Traccia A: WAPE_recovery e WPE_recovery di tutti gli imputer
  → Ranking degli imputer

Notebook: notebooks/08_forecast_all_combinations.py
  → Per ogni imputer I × ogni forecaster F:
    - Naive su completed_sales(I) → eval HF
    - LightGBM su completed_sales(I) → tuning/retrain/eval HF
    - MLP su completed_sales(I) → tuning/retrain/eval HF
    - Foundation su completed_sales(I) → eval HF (se applicabile)
  → Output: tutte le celle della matrice

Output: matrice completa imputer × forecaster
```

### Fase C — Analisi (Settimana 4-5)

```
Notebook: notebooks/09_analysis_matrix.py
  → Costruzione matrice completa
  → Analisi 1-7
  → Visualizzazioni: heatmap, curve di saturazione,
    boxplot per tipo di serie, grafici interazione

Notebook: notebooks/10_daily_aggregation.py
  → Aggregazione a livello giornaliero (solo ore in-stock)
  → Confronto con numeri del paper baseline

Output: tutte le figure e tabelle per il paper
```

---

## Struttura del progetto

```
project/
├── CLAUDE.md                          ← questo file
├── data/
│   ├── frn50k_train.parquet           ← train HF (90 giorni)
│   ├── frn50k_eval.parquet            ← eval HF (7 giorni)
│   ├── mnar_masks_val.parquet         ← maschere MNAR (gg 84-90, seed=42)
│   └── completed_sales/
│       ├── media_cond.parquet
│       ├── media_glob.parquet
│       ├── mediana_cond.parquet
│       ├── lgb.parquet
│       ├── timesnet.parquet
│       ├── saits.parquet
│       ├── itransformer.parquet
│       └── dlinear.parquet
├── baseline_paper/
│   ├── frn-50k-baseline/              ← clone repo Dingdong-Inc
│   └── notes_baseline.md              ← appunti sul codice
├── notebooks/
│   ├── 01_eda.py
│   ├── 02_generate_mnar_masks.py
│   ├── 03_baseline_naive.py
│   ├── 04_baseline_ml.py
│   ├── 05_imputation_naive_ml.py
│   ├── 06_imputation_sota.py
│   ├── 07_traccia_a_results.py
│   ├── 08_forecast_all_combinations.py
│   ├── 09_analysis_matrix.py
│   ├── 10_daily_aggregation.py
│   └── figures/
├── src/
│   ├── data/
│   │   ├── loading.py
│   │   └── mnar_masking.py
│   ├── imputation/
│   │   ├── naive_imputers.py
│   │   ├── lgb_imputer.py
│   │   └── pypots_wrapper.py
│   ├── forecasting/
│   │   ├── naive_forecasters.py
│   │   ├── lgb_forecaster.py
│   │   ├── mlp_forecaster.py
│   │   └── feature_engineering.py
│   └── evaluation/
│       ├── metrics.py
│       ├── statistical_tests.py
│       └── analysis.py
├── results/
│   ├── traccia_a.parquet
│   ├── matrix.parquet
│   └── per_series/
└── configs/
    └── default.yaml
```

---

## Riferimenti chiave

- FreshRetailNet-50K — Wang et al. (2025), arxiv 2505.16319
- TimesNet — Wu et al. (2023), ICLR
- SAITS — Du et al. (2023), Expert Systems with Applications
- iTransformer — Liu et al. (2024), ICLR
- DLinear — Zeng et al. (2023), AAAI
- PyPOTS — Du et al. (2023), toolkit per time series con dati mancanti
- LightGBM — Ke et al. (2017), NeurIPS
- Chronos-2 — Ansari et al. (2024), Amazon Science
