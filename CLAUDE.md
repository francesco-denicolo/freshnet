# PINN-Retail: Physics-Informed Neural Networks per Demand Forecasting di Prodotti Deperibili

## Obiettivo del progetto

Costruire un modello PINN (Physics-Informed Neural Network) per prevedere la domanda latente di prodotti deperibili nel retail, gestendo nativamente il censoring da stockout senza una fase di imputation separata.

**Proposta di ricerca B6** — Target: pubblicazione su NeurIPS/ICLR/JCP.

---

## Dataset: FreshRetailNet-50K

- **Fonte**: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
- **Dimensione**: 4.500.000 righe (train) + 350.000 (eval)
- **Periodo**: Marzo–Giugno 2024, 90 giorni
- **Granularità**: Oraria (vendite e stock status)
- **Copertura**: 898 negozi, 18 città cinesi, 863 SKU deperibili

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
| dt | str | Data (giornaliera) |
| sale_amount | float | Vendite giornaliere totali |
| hours_sale | str/list | Vendite per fascia oraria |
| stock_hour6_22_cnt | int | Ore di stockout tra 6:00 e 22:00 (conta degli 1 in hours_stock_status) |
| hours_stock_status | str/list | Stato stock binario per fascia oraria (0=in stock, 1=stockout) |
| discount | float | Sconto promozionale |
| holiday_flag | int | Flag festività |
| activity_flag | int | Flag attività promozionale |
| precpt | float | Precipitazioni |
| avg_temperature | float | Temperatura media |
| avg_humidity | float | Umidità media |
| avg_wind_level | float | Livello vento medio |

### IMPORTANTE: cosa NON c'è nel dataset

- **NON c'è il livello di inventario continuo I(t)**. C'è solo lo stato binario (in stock / stockout).
- **NON ci sono i rifornimenti R(t)**.
- **NON c'è lo scarto per deterioramento W(t)**.
- I campi `hours_sale` e `hours_stock_status` sono probabilmente stringhe JSON o liste da parsare.

---

## Formulazione matematica

### Il problema del censoring

Le vendite osservate NON sono la domanda vera:

```
S_obs(t) = min(D(t), I(t))
```

Quando c'è stockout, S_obs = 0 indipendentemente dalla domanda vera. I modelli standard imparano a prevedere zero durante gli stockout → sottostima sistematica → ciclo vizioso.

### Architettura del modello PINN-Retail

**Componenti:**
1. **Encoder** (MLP): processa le features dell'ora target. Architettura semplice perché il contributo è nella loss, non nell'encoder. Si può scalare a LSTM/Transformer dopo aver validato la loss.
2. **Testa domanda** (Linear + softplus): produce D*(t) > 0 (domanda latente)
3. **Testa inventario** (Linear + softplus): produce I*(t) ≥ 0 (inventario latente)

**DECISIONE: niente decoder deterioramento.** Il dataset non ha dati sullo scarto, quindi modellare il deterioramento sarebbe speculativo (variabile latente che dipende da altra variabile latente senza supervisione).

**DECISIONE: stock_status NON è un input del network.** Motivazioni:
1. A inference non lo conosciamo (stiamo prevedendo il futuro)
2. Se fosse input, il network imparerebbe status=1 → D≈0, replicando il censoring
3. stock_status va usato SOLO nella loss (per distinguere regime in-stock vs stockout)

**Output:** La rete produce due quantità tramite teste separate:
```
h(t)  = encoder(features(t))
D*(t) = softplus(head_D(h(t)))    — domanda latente (sempre > 0)
I*(t) = softplus(head_I(h(t)))    — inventario latente (sempre ≥ 0)
```

**Features di input** (NO stock_status):
- Temporali: ora del giorno, giorno della settimana, giorno del mese
- Categoriali: city_id, store_id, product_id, category L1/L2/L3
- Esogene: temperatura, umidità, precipitazioni, vento, discount, holiday, activity
- Storiche: lag di vendite osservate (da determinare con analisi autocorrelazione)
  NOTA: gli zeri da stockout nello storico sono segnale censurato, non domanda zero.
  Opzioni: (a) usare lag raw, (b) mascherare/imputare zeri da stockout nei lag.
  La scelta (a) vs (b) sarà validata sperimentalmente.

**Vendite predette** (codifica: 0=in stock, 1=stockout):
- Quando stock_status(t) = 0 (in-stock): S_pred(t) = D*(t) — domanda soddisfatta
- Quando stock_status(t) = 1 (stockout): S_pred(t) = 0 — vendite censurate

### Loss function — 3 termini

```
L_total = L_data + λ₁·L_boundary + λ₂·L_cons
```

La non-negatività di D* e I* è garantita da softplus (non serve L_nonneg esplicito).
Il collasso della domanda durante stockout è prevenuto dall'ARCHITETTURA (stock_status
non è un input → il modello non può distinguere ore in-stock da stockout → D* generalizza
automaticamente), non da un termine nella loss.

**Termine 1 — L_data (aderenza ai dati, SOLO ore in-stock):**
```
L_data = (1/|T_in|) · Σ_{t: status=0} (D*(t) − S_obs(t))²
dove T_in = {t : stock_status(t) = 0}    (ore in-stock)
```
Nessun downweighting α: le ore di stockout sono ESCLUSE, non downweightate.
Giustificazione EDA: durante in-stock S_obs = D (75.1% dei dati, segnale abbondante).
Durante stockout S_obs ≈ 0 nel 97.1% dei casi → L_data sarebbe inutile.

**Termine 2 — L_boundary (condizioni al contorno da stock_status):**
```
L_boundary = (1/|T_so|) · Σ_{t: status=1} I*(t)²
           + (1/|T_in|) · Σ_{t: status=0} ReLU(D*(t) − I*(t))²
dove T_so = {t : stock_status(t) = 1}    (ore di stockout)
```
Due sotto-vincoli:
- **Stockout → I* ≈ 0**: quando status=1, l'inventario deve essere esaurito.
- **In-stock → I* ≥ D***: quando status=0, l'inventario è sufficiente a soddisfare
  la domanda (altrimenti sarebbe stockout). Equivale a dire S_obs = D, non min(D,I).
Giustificazione EDA: il pattern di deplezione giornaliera (5% stockout ore 7 → 42% ore 22)
mostra che I* si azzera progressivamente. stock_status binario funge da supervisione per I*.

**Termine 3 — L_cons (conservazione dell'inventario, DISUGUAGLIANZA):**
```
L_cons = (1/(T−1)) · Σ_t ReLU(−[I*(t+1) − I*(t) + min(D*(t), I*(t))])²
```
Fisica: I(t+1) = I(t) − min(D(t), I(t)) + R(t), con R(t) ≥ 0.
Riarrangiando: I(t+1) − I(t) + min(D(t), I(t)) = R(t) ≥ 0.
Il termine penalizza SOLO quando R(t) implicito < 0 (fisicamente impossibile).
Perché disuguaglianza e non uguaglianza: la formulazione originale (r_cons² → 0)
imponeva R(t) = 0 ∀t. Ma l'EDA mostra che il rifornimento avviene (stockout si resetta
tra giorni). La disuguaglianza permette R(t) > 0 dove serve.

**DECISIONE: L_censor (Tobit) eliminato.** Poiché stock_status NON è un input del network,
il modello non può distinguere ore in-stock da ore di stockout a livello di features.
D*(t) dipende solo da (ora, giorno, meteo, categoria, storico) e generalizza automaticamente.
L_censor sarebbe ridondante e potrebbe introdurre bias positivo nelle ore a domanda
naturalmente bassa (es. 3am). L'unico rischio residuo è che lo storico vendite censurato
(zeri da stockout) influenzi le predizioni, ma questo va gestito nel preprocessing dei lag,
non nella loss.

### Ottimizzazione: Lagrangiano Aumentato (ALM)

I λ₁, λ₂ non sono fissi — sono variabili ottimizzate con ALM:

```
L_ALM = L_data + Σ_k [λ_k · V_k + (ρ_k/2) · V_k²]
dove k ∈ {boundary, cons} e V_k è la violazione del vincolo k
```

Training a 3 fasi per ogni iterazione:
1. Passo primale: aggiorna Θ (pesi rete) con Adam minimizzando L_ALM
2. Passo duale: λ_k ← max(0, λ_k + ρ_k · V_k)
3. Adattamento: se V_k non migliora, ρ_k ← γ · ρ_k

A convergenza, λ_k* = shadow prices (interpretazione economica).
- λ_boundary* misura il "costo" della violazione delle condizioni di stockout
- λ_cons* misura il "costo" della violazione della conservazione dell'inventario

---

## Piano sperimentale

### Livello 1 — Baseline naive (direct forecast, ignorano il censoring)
- Naive direct (profilo ultimo giorno → applicato a tutto l'orizzonte)
- MA direct (profilo media ultimi K giorni → applicato a tutto l'orizzonte)
- Global Mean (profilo medio su tutto il training)
- DoW Mean (profilo medio per giorno della settimana)
- LightGBM su S_obs
- MLP su S_obs

### Livello 2 — Two-stage (imputation + forecasting)
- Imputation con media condizionata + LightGBM
- Imputation con TimesNet/iTransformer + TFT (come nel paper baseline)

### Livello 3 — PINN-Retail (nostro contributo)
- End-to-end, nessuna imputation, vincoli fisici nella loss

### Metriche
- **WAPE** (Weighted Absolute Percentage Error) — accuratezza
- **WPE** (Weighted Percentage Error) — bias/sottostima
- **Violazione vincoli** (solo PINN) — consistenza fisica

### Windowing
- Input: finestra di H ore (es. H=168, una settimana)
- Target: FH ore successive (es. FH=24, un giorno)
- Stride: 1 (o maggiore per ridurre correlazione)
- Campioni: ~N−H−FH+1 per serie, ×50.000 serie per modello globale

---

## Stato attuale

**PASSO 1 (completato):** Analisi esplorativa del dataset.
- ✅ Dati scaricati da HuggingFace e salvati come parquet in data/
- ✅ Parsing campi orari: numpy array di 24 elementi (ore 0-23)
- ✅ Statistiche descrittive complete
- ✅ 13 visualizzazioni in notebooks/figures/
- ✅ Notebook completo: notebooks/01_eda.py

**PASSO 3 (completato):** Framework metriche.
- ✅ Framework metriche riusabile: src/evaluation/metrics.py
- ✅ compute_metrics(), format_metrics_table()
- ✅ Parquet per-serie salvati in notebooks/results/ per confronto con modelli successivi

Note: WAPE mediana per-serie > WAPE pooled perché il pooled è volume-weighted
(serie ad alto volume hanno WAPE più basso e pesano di più).

**PASSO 3a (completato):** Global Mean Profile baseline.
- ✅ Notebook: notebooks/04_baseline_global_mean.py
- ✅ Per ogni serie, predizione = media hours_sale (profilo fisso 24h)
  - Val: profilo calcolato su giorni 1-83 (solo train)
  - Test: profilo calcolato su giorni 1-90 (train+val)
- ✅ Risultati (Global Mean):

| Split | WAPE_overall | WAPE_instock | WAPE_stockout | WPE_overall |
|-------|-------------|-------------|--------------|------------|
| Train (gg 2-83) | 1.1259 | 0.9745 | 6.9214 | -0.0033 |
| Val (gg 84-90) | 1.0418 | 0.9311 | 5.9399 | -0.1793 |
| Test (eval) | 1.0444 | 0.9319 | 6.1081 | -0.1630 |

Mediana per-serie WAPE (test): 1.2590

**PASSO 3b (completato):** Day-of-Week Mean Profile baseline.
- ✅ Notebook: notebooks/05_baseline_dow_mean.py
- ✅ Per ogni serie, 7 profili medi (uno per giorno della settimana) → cattura stagionalità settimanale
  - Val: profili calcolati su giorni 1-83 (solo train, 11-12 gg per DoW)
  - Test: profili calcolati su giorni 1-90 (train+val, 12-13 gg per DoW)
- ✅ Risultati (DoW Mean):

| Split | WAPE_overall | WAPE_instock | WAPE_stockout | WPE_overall |
|-------|-------------|-------------|--------------|------------|
| Train (gg 2-83) | 1.0457 | 0.9044 | 6.4554 | -0.0017 |
| Val (gg 84-90) | 1.0397 | 0.9298 | 5.9041 | -0.1802 |
| Test (eval) | 1.0410 | 0.9291 | 6.0781 | -0.1638 |

Mediana per-serie WAPE (test): 1.2517

**PASSO 3c (completato):** Naive Direct Forecast.
- ✅ Notebook: notebooks/06_baseline_naive_direct.py
- ✅ Predizione: profilo dell'ultimo giorno prima dell'orizzonte, applicato a tutti i 7 giorni
  - Val: profilo = S_obs(giorno 83) → applicato a giorni 84-90
  - Test: profilo = S_obs(giorno 90) → applicato a giorni 91-97
- ✅ Nessuna osservazione del periodo di forecast viene usata

| Split | WAPE_overall | WAPE_instock | WPE_overall |
|-------|-------------|-------------|------------|
| Val | 1.1469 | 1.0351 | -0.1215 |
| Test | 1.1765 | 1.0605 | -0.0198 |

Mediana per-serie WAPE (test): 1.3505

**PASSO 3d (completato):** MA Direct Forecast (K selection su val).
- ✅ Notebook: notebooks/07_baseline_ma_direct.py
- ✅ K selezionato su val in modalità direct forecast (profilo fisso, no osservazioni val)
- ✅ Criteri discordanti: K=14 (pooled) vs K=83 (median, ≈ Global Mean). Usato K*=14.
- ✅ Test: profilo = media ultimi 14 giorni prima del test (giorni 77-90, include val)

| Split | WAPE_overall | WAPE_instock | WPE_overall |
|-------|-------------|-------------|------------|
| Val | 1.0160 | 0.9069 | -0.1208 |
| Test | 1.0210 | 0.9072 | -0.0553 |

Mediana per-serie WAPE (test): 1.2735

**PASSO 3e (completato):** MLP Baseline (variante A, no history).
- ✅ Notebook: notebooks/08_baseline_mlp.py (training + variant selection)
- ✅ Notebook: notebooks/08b_mlp_evaluate.py (valutazione test + confronto)
- ✅ Notebook: notebooks/08c_mlp_instock_plots.py (boxplot in-stock)
- ✅ Architettura: embeddings (store→32, product→32, city→8, dow→4) + continuous(7) = 83 dim input
  → Linear(128)+ReLU → Linear(64)+ReLU → Linear(24)+Softplus
- ✅ Training: MSE loss su tutte le ore, 30 epoche, Adam lr=1e-3, batch=4096, MPS (Apple Silicon)
- ✅ Solo variante A (no history) completata — varianti B-E troppo lente per sessione singola

| Split | WAPE_overall | WAPE_instock | WPE_overall |
|-------|-------------|-------------|------------|
| Val | 0.9785 | 0.8638 | -0.2423 |
| Test | 0.9766 | 0.8686 | -0.2933 |

Mediana per-serie WAPE (test): 1.1835
Mediana per-serie WAPE in-stock (test): 1.0815

**PASSO 3e-bis (completato):** MLP Baseline (variante F, M5-style lag features).
- ✅ Notebook: notebooks/08_baseline_mlp.py (aggiornato per variante F)
- ✅ 11 lag features M5-style × 24 ore = 264 valori + 11 binary masks = 275 dim lag
- ✅ NaN handling: 0 + binary mask (1=disponibile, 0=mancante). Scelto dall'utente.
- ✅ Normalizzazione lag: media/std dal training set
- ✅ Input totale: 76 (emb) + 7 (cont) + 275 (lag+mask) = 358 dim → 112K parametri
- ✅ Training: early stopping a epoca 8, val WAPE=0.9635, batch=4096

| Split | WAPE_overall | WAPE_instock | WPE_overall |
|-------|-------------|-------------|------------|
| Val | 0.9635 | 0.8629 | -0.2054 |
| Test | 0.9590 | 0.8588 | -0.1776 |

Mediana per-serie WAPE (test): 1.2029
Mediana per-serie WAPE in-stock (test): 1.0859

Note MLP A vs F:
- F migliora WAPE pooled (-1.8%: 0.977→0.959) ma peggiora mediana (+1.6%: 1.184→1.203)
- F riduce il bias del 39% (WPE -0.293→-0.178). Stesso pattern di LGB A→F.
- Conferma: lag features beneficiano le serie ad alto volume (pooled) ma non le piccole serie (mediana)

**PASSO 3f (completato):** LightGBM Baseline (variante A, no history).
- ✅ Notebook: notebooks/09_baseline_lgb.py
- ✅ Modello singolo per-ora: ogni riga = (store, product, giorno, ora) → 1 vendita
  Features categoriche native (no embedding): store_id, product_id, city_id, dow, hour
  Features continue: discount, avg_temperature, avg_humidity, precpt, avg_wind_level, holiday_flag, activity_flag
- ✅ Training: MAE loss, 98M righe, 298 boosting rounds (early stopping 20)
- ✅ HP non ottimizzati: num_leaves=31, lr=0.1, bagging=0.3, max_bin=127
- ✅ Variante A (no history) e variante F (M5-style, 11 lag features) completate
- ✅ Variante F: 11 features lag M5-style (raw lags, rolling means/std, DoW-specific, daily agg, momentum)
  NaN per lag mancanti (LightGBM gestisce nativamente). 497 boosting rounds, MAE val = 0.0498

Risultati variante A (no history):

| Split | WAPE_overall | WAPE_instock | WPE_overall |
|-------|-------------|-------------|------------|
| Val | 1.0310 | 0.9106 | -0.1302 |
| Test | 1.0340 | 0.9197 | -0.1691 |

Risultati variante F (M5-style, 11 lag features):

| Split | WAPE_overall | WAPE_instock | WPE_overall |
|-------|-------------|-------------|------------|
| Val | 0.9981 | 0.8838 | -0.1151 |
| Test | 0.9993 | 0.8827 | -0.0747 |

Feature importance F (top 5): rmean_7d(360K), rmean_14d(284K), rmean_dow(82K), hour(76K), store_id(47K)
Le rolling means dominano; i raw lags (lag_1d=17K) sono meno informativi.

**Confronto tutti i baseline direct forecast (test):**

| Modello | WAPE_in med | WPE_in med | WAPE_all med | WAPE_pooled |
|---------|:----------:|:----------:|:------------:|:-----------:|
| MLP (var A) | **1.0815** | -0.4045 | **1.1835** | 0.9766 |
| DoW Mean | 1.1176 | -0.2279 | 1.2517 | 1.0410 |
| LGB (var A) | 1.1221 | -0.2271 | 1.2644 | 1.0340 |
| Global Mean | 1.1243 | -0.2272 | 1.2590 | 1.0444 |
| LGB (var F) | 1.1293 | -0.1827 | 1.2747 | **0.9993** |
| MA (K=14) | 1.1341 | -0.1722 | 1.2735 | 1.0210 |
| Naive Direct | 1.2192 | -0.1520 | 1.3505 | 1.1765 |

Note:
- MLP è il miglior baseline su WAPE mediana ma ha il WPE più negativo (-0.40 in-stock)
- LGB var F è il miglior baseline su WAPE pooled (0.999) ma mediana per-serie è nella media
- Le lag features M5-style migliorano il pooled (volume-weighted) -3.4% ma non la mediana per-serie
- Trade-off WAPE vs WPE confermato: modelli più accurati tendono a sottostimare di più
- LGB var F ha il WPE meno negativo tra i modelli ML (-0.183 vs MLP -0.405)

**PASSO 4 (completato):** Two-Stage (Imputation + LightGBM Forecasting).
- ✅ Notebook: notebooks/10_twostage_lgb.py
- ✅ Stage 1: due metodi di imputation per ore di stockout
  - Conditional Mean: media per (store, product, dow, hour) su ore in-stock
  - LGB Imputation: LGB trainato su in-stock hours (12 base features, no lag)
- ✅ Stage 2: LGB con M5 lag features calcolati da completed_sales (decontaminati)
  - Target = completed Y(t) = S_obs se in-stock, D̂ se stockout
  - Early stopping su val MAE (completed Y)
- ✅ LGB Imputation selezionato come metodo migliore

Imputation diagnostica:
- Media D̂ imputata (LGB): 0.0409 vs media S_obs in-stock: 0.0547 (ratio 0.75)
- momentum_1d_7d NaN: 18.6% (vs 42% nel single-stage) — meno NaN grazie ai valori imputati

Stage 2 selection (val):

| Metodo | WAPE_in pool | WAPE_in med | Iter |
|--------|:---:|:---:|:---:|
| Conditional Mean | 0.9057 | 1.143 | 77 |
| LGB Imputation | 0.8945 | 1.138 | 493 |

Risultati Two-Stage LGB (best = LGB Imputation):

| Split | WAPE_overall | WAPE_instock | WAPE_stockout | WPE_overall |
|-------|-------------|-------------|--------------|------------|
| Val | 1.0611 | 0.8945 | 8.4369 | +0.0553 |
| Test | 1.0567 | 0.8952 | 8.3207 | +0.0752 |

Mediana per-serie (test):
- WAPE_instock: 1.1573, WPE_instock: -0.0727, WAPE_overall: 1.3450

Feature importance Stage 2 (top 5): rmean_7d(365K), rmean_dow(318K), rmean_14d(186K), hour(105K), lag_1d(77K)
rmean_dow sale dal 3° al 2° posto vs single-stage — lag DoW beneficiano della de-censoring.

**PASSO 4b (completato):** Two-Stage MLP (Imputation + MLP Forecasting).
- ✅ Notebook: notebooks/11_twostage_mlp.py
- ✅ Stage 1: LGB Imputation (stesso approccio di notebook 10, 497 iter, MAE 0.0585)
  - Media D̂ imputata: 0.0409 vs S_obs in-stock: 0.0547 (ratio 0.75)
- ✅ Stage 2: MLP con M5-style lag features da completed_sales
  - Target = completed Y(t), early stopping su val WAPE vs completed_Y
  - Architettura: [128, 64] + Softplus, 112K parametri, input 358 dim
  - Early stopping a epoca 4, val WAPE=0.774 (vs completed_Y)
- ✅ Dataset construction vettorizzato: 63s (vs ~20min con loop naive)
  - cumsum per rolling mean/std, searchsorted per DoW features, pre-allocation

Risultati Two-Stage MLP (vs S_obs originale):

| Split | WAPE_overall | WAPE_instock | WAPE_stockout | WPE_overall |
|-------|-------------|-------------|--------------|------------|
| Val | 1.0270 | 0.8746 | 7.7731 | -0.0293 |
| Test | 1.0141 | 0.8721 | 7.4008 | -0.0367 |

Mediana per-serie (test):
- WAPE_instock: 1.1146, WPE_instock: -0.1883, WAPE_overall: 1.2765

**Confronto tutti i modelli (test, mediana per-serie in-stock):**

| Modello | WAPE_in med | WPE_in med | WAPE_all med | WAPE_in pool |
|---------|:----------:|:----------:|:------------:|:------------:|
| MLP (var A) | **1.0815** | -0.4045 | **1.1835** | 0.8686 |
| MLP (var F) | 1.0859 | -0.3235 | 1.2029 | 0.8588 |
| 2-Stage MLP | 1.1146 | -0.1883 | 1.2765 | **0.8721** |
| DoW Mean | 1.1176 | -0.2279 | 1.2517 | 0.9291 |
| LGB (var A) | 1.1186 | -0.2365 | 1.2586 | 0.9197 |
| Global Mean | 1.1243 | -0.2272 | 1.2590 | 0.9319 |
| LGB (var F) | 1.1268 | -0.2013 | 1.2747 | 0.8827 |
| MA (K=14) | 1.1341 | -0.1722 | 1.2735 | 0.9072 |
| 2-Stage LGB | 1.1573 | **-0.0727** | 1.3450 | 0.8952 |
| Naive Direct | 1.2192 | -0.1520 | 1.3505 | 1.0605 |

Note:
- **Two-Stage MLP** si posiziona al 3° posto su WAPE mediana (1.115) dopo MLP-A (1.082) e MLP-F (1.086)
- Riduce il bias del 53% vs MLP-A (WPE -0.19 vs -0.40) — a metà strada tra single-stage e two-stage LGB
- WAPE pooled (0.872) comparabile a MLP-A (0.869) — il two-stage non peggiora il pooled
- Two-Stage MLP > Two-Stage LGB su WAPE (1.115 vs 1.157) ma peggio su WPE (-0.19 vs -0.07)
- L'MLP cattura meglio i pattern di domanda ma sotto-imputa rispetto al LGB (il modello "si fida meno" delle lag decontaminate)
- Trade-off confermato: il two-stage sposta il cursore bias↔varianza. MLP single-stage minimizza WAPE, two-stage riduce il bias
- Il PINN dovrà dimostrare: ridurre il bias (come two-stage) senza sacrificare WAPE (come single-stage)

**PASSO 5 (completato):** PINN-Retail (end-to-end, physics-informed loss).
- ✅ Notebook: notebooks/12_pinn.py
- ✅ Architettura: stesso backbone MLP-F (358 dim input) + due teste (D* domanda, I* inventario)
  - Encoder: [128, 64] + ReLU condiviso, poi head_D(64→24)+Softplus e head_I(64→24)+Softplus
  - 113,916 parametri (vs 112K MLP — aggiunta solo head_I: 1,560 params)
  - stock_status NON è input — usato solo nella loss
  - Lag features M5-style da S_obs (non completati, end-to-end senza imputation)
- ✅ Loss: L_data (MSE in-stock only) + L_boundary (I*≈0 stockout, I*≥D* instock) + L_cons (R(h)≥0 within-day)
- ✅ Training ALM: warmup 3 epoche L_data only, poi max 15 iter ALM × 3 epoche interne
  - Early stopping a ALM iter 8 (best iter 3, epoch 12)
  - V_c (conservazione) ≈ 0 dall'inizio — la disuguaglianza R(h)≥0 è naturalmente soddisfatta
  - V_b (boundary) scende da 0.026 → 0.001 ma rho cresce troppo → WAPE degrada dopo iter 3
  - Shadow prices: λ_boundary=0.068, λ_conservation≈0
  - Training time: 3915s (~65 min, 27 epoche totali)

Risultati PINN-Retail (test):

| Split | WAPE_overall | WAPE_instock | WAPE_stockout | WPE_overall |
|-------|-------------|-------------|--------------|------------|
| Val | 0.9078 | 0.8403 | 3.8938 | -0.3572 |
| Test | 0.9043 | 0.8357 | 3.9872 | -0.3376 |

Mediana per-serie (test):
- WAPE_instock: 1.0404, WPE_instock: -0.4743, WAPE_overall: 1.1052

Constraint metrics (test):
- V_boundary: 0.016 (I* stockout mean=0.012, gap instock mean=0.005)
- V_conservation: 0.00001 (quasi perfetto)
- D* mean instock: 0.038, D* mean stockout: 0.017

**Confronto tutti i modelli (test, mediana per-serie in-stock):**

| Modello | WAPE_in med | WPE_in med | WAPE_all med | WAPE_in pool |
|---------|:----------:|:----------:|:------------:|:------------:|
| **PINN-Retail** | **1.0404** | -0.4743 | **1.1052** | **0.8357** |
| MLP (var A) | 1.0815 | -0.4045 | 1.1835 | 0.8686 |
| MLP (var F) | 1.0859 | -0.3235 | 1.2029 | 0.8588 |
| 2-Stage MLP | 1.1146 | -0.1883 | 1.2765 | 0.8721 |
| DoW Mean | 1.1176 | -0.2279 | 1.2517 | 0.9291 |
| LGB (var A) | 1.1186 | -0.2365 | 1.2586 | 0.9197 |
| Global Mean | 1.1243 | -0.2272 | 1.2590 | 0.9319 |
| LGB (var F) | 1.1268 | -0.2013 | 1.2747 | 0.8827 |
| MA (K=14) | 1.1341 | -0.1722 | 1.2735 | 0.9072 |
| 2-Stage LGB | 1.1573 | **-0.0727** | 1.3450 | 0.8952 |
| Naive Direct | 1.2192 | -0.1520 | 1.3505 | 1.0605 |

Note:
- **PINN è il miglior modello su WAPE** (accuratezza) su tutte le metriche:
  - WAPE_in pooled: 0.836 (-2.7% vs MLP-F, -3.8% vs MLP-A)
  - WAPE_in mediana: 1.040 (-3.8% vs MLP-A)
  - WAPE_all mediana: 1.105 (-6.6% vs MLP-A)
- **Ma il bias (WPE) è il peggiore**: -0.474 (peggio di MLP-A -0.405)
- I vincoli fisici (L_boundary, L_cons) agiscono come **regolarizzatore efficace** → migliorano la generalizzazione
- Il bias non migliora perché la causa è nei **lag features contaminati** (stockout zeros), non nella loss
  - L_boundary vincola I* ma non decontamina i lag
  - Il vincolo I*≥D* (instock) può spingere D* verso il basso → più sottostima
- WAPE_stockout basso (3.99 vs 7-8 per two-stage) → D* è più conservativo durante stockout
- La conservazione V_c≈0 è triviale: l'inequality R(h)≥0 è automaticamente soddisfatta dal softplus
- Il trade-off fondamentale persiste: decontaminare i lag (two-stage) riduce bias ma peggiora WAPE

**PROSSIMI PASSI:**
- Passo 6: Confronto sistematico e analisi approfondita

---

## Decisioni prese

1. **No deterioramento**: eliminato W_pred e L_decay. Il dataset non ha dati di scarto.
2. **Due teste (D*, I*)**: il network predice domanda latente e inventario latente tramite teste separate con softplus. L'inventario è vincolato dallo stock_status binario via L_boundary.
3. **3 termini nella loss**: L_data + L_boundary + L_cons. Non-negatività implicita via softplus.
4. **Modello globale**: un unico modello su tutte le 50.000 serie (non locale per serie).
5. **stock_status NON è input**: usato solo nella loss per distinguere regime in-stock/stockout. A inference non è disponibile e contaminerebbe la predizione. Questa decisione architetturale è ciò che previene il collasso della domanda durante stockout.
6. **L_data solo in-stock**: eliminato il downweighting α. Le ore di stockout sono escluse da L_data (EDA: S_obs ≈ 0 nel 97.1% dei casi di stockout, inutile fittarle).
7. **L_cons come disuguaglianza**: permette R(t) ≥ 0, non forza R(t) = 0 (EDA: il rifornimento avviene tra giorni).
8. **L_censor Tobit eliminato**: ridondante dato che stock_status non è input. Il collasso è prevenuto dall'architettura, non dalla loss. L_censor introdurrebbe bias positivo nelle ore a domanda naturalmente bassa.
9. **Encoder MLP (non Transformer)**: il contributo è nella loss, non nell'encoder. MLP semplice per validare la loss, scalabile a architetture più complesse successivamente.

## Scoperte dall'EDA (Passo 1)

1. **Codifica stock_status**: la documentazione ufficiale HuggingFace conferma **0=in stock, 1=stockout** (il CLAUDE.md originale aveva la codifica invertita — ora corretto). `stock_hour6_22_cnt` conta le ore di stockout (gli 1) tra 6 e 22.
2. **Tasso stockout reale**: 24.9% delle ore-slot sono in stockout (stock_status=1). Il 60% delle righe-giorno ha almeno 1h di stockout. Il 40% non ha alcun stockout. Il 3.8% ha stockout completo (24h).
3. **Vendite e stockout coerenti**: media vendite/ora in-stock=0.054 (28.6% ore con vendite>0), media vendite/ora stockout=0.004 (2.9% ore con vendite>0). Il censoring funziona come atteso: durante lo stockout le vendite sono quasi zero.
4. **Campi orari**: numpy arrays di 24 elementi. sum(hours_sale) ≈ sale_amount (coerente). stock_hour6_22_cnt corrisponde solo al 58% con sum(stock[6:23]) — possibile metodo di calcolo diverso nel dataset originale.
5. **Pattern temporali**: stockout minimo nelle ore notturne (~6% ore 0-5, prodotto disponibile ma nessuno compra), massimo nel tardo pomeriggio-sera. Vendite bimodali (picchi ore 8-10 e 15-17). Weekend +25% vendite.
6. **Variabili esogene**: discount < 0.7 → vendite +59%, holiday → +27%, meteo effetto debole.
7. **Eval set**: 7 giorni subito dopo train, covariate shift (temp +7°C, pioggia +59%).

---

## Riferimenti chiave

- Raissi et al. (2019) — PINNs fondativi, J. Computational Physics
- Cranmer et al. (2020) — Lagrangian Neural Networks, NeurIPS
- Shin et al. (2020) — Convergenza PINNs
- Bertsekas (1982) — ALM convergenza
- Ghare & Schrader (1963) — ODE inventario deperibile
- FreshRetailNet-50K paper (2025) — arxiv 2505.16319
- Covert & Philip (1973) — Weibull deterioration
- Xiao et al. (2024) — Generalized LNNs

---

## Struttura progetto prevista

```
pinn-retail/
├── CLAUDE.md              ← questo file
├── data/
│   ├── frn50k_train.parquet
│   └── frn50k_eval.parquet
├── notebooks/
│   ├── 01_eda.py          ← analisi esplorativa
│   ├── 04_baseline_global_mean.py    ← global mean profile baseline (direct)
│   ├── 05_baseline_dow_mean.py       ← day-of-week mean profile baseline (direct)
│   ├── 06_baseline_naive_direct.py   ← naive direct forecast baseline
│   ├── 07_baseline_ma_direct.py      ← MA direct forecast baseline (K=14)
│   ├── 08_baseline_mlp.py            ← MLP baseline (training + variant selection)
│   ├── 08b_mlp_evaluate.py           ← MLP evaluation on test + confronto
│   ├── 08c_mlp_instock_plots.py      ← boxplot in-stock tutti i modelli
│   ├── 09_baseline_lgb.py            ← LightGBM baseline (direct forecast)
│   ├── 10_twostage_lgb.py            ← Two-stage: imputation + LightGBM
│   ├── 11_twostage_mlp.py            ← Two-stage: imputation + MLP
│   └── 12_pinn.py                    ← PINN-Retail (end-to-end, physics-informed)
├── src/
│   ├── data/
│   │   ├── dataset.py     ← loading e parsing
│   │   └── windowing.py   ← creazione finestre train/val/test
│   ├── models/
│   │   ├── baselines.py   ← seasonal naive, media mobile, LightGBM
│   │   ├── twostage.py    ← imputation + forecasting
│   │   └── pinn_retail.py ← il modello PINN
│   ├── losses/
│   │   └── pinn_loss.py   ← L_data, L_boundary, L_cons, ALM
│   ├── training/
│   │   └── trainer.py     ← loop di training con ALM
│   └── evaluation/
│       └── metrics.py     ← WAPE, WPE, violazione vincoli
├── configs/
│   └── default.yaml       ← iperparametri
└── main.py                ← entry point
```

---
---

# PIVOT (2026-04 / 2026-05): Benchmark Imputer × Forecaster su FreshRetailNet-50K

> **NOTA**: il progetto ha cambiato direzione. Dal PINN-Retail (sopra) si è passati a un **benchmark sistematico** delle combinazioni imputer × forecaster sulle stesse serie. Il PINN-Retail resta documentato sopra come contesto storico. Il lavoro **attivo** è descritto qui sotto.

## Obiettivo del pivot

Studiare l'impatto della scelta dell'imputer di stockout sul forecasting di domanda nel retail deperibile, su un benchmark sistematico di **11 imputer × 9 forecaster = 99 celle (pianificato), 66 celle disponibili attualmente**. Target: pubblicazione su rivista applicata (es. International Journal of Forecasting, Decision Support Systems, Expert Systems with Applications).

## Setup

- **Dataset**: FreshRetailNet-50K (Liu et al., 2025, arXiv:2505.16319). 50.000 serie × 90 giorni di train + 7 giorni di test (2024-03-28 → 2024-07-02). Distribuzione molto sbilanciata: City 0 = 25.811 serie (52%), City 8 = 65 serie.
- **Restrizione orario**: ore 6-22 (17 ore/giorno operative). Le ore notturne 0-5 e 23 hanno vendite ~0 e sono escluse.
- **Split temporale**:
  - Train interno: gg 1-83 (training modelli)
  - Validation: gg 84-90 (early stopping, selezione HP)
  - Test: gg 91-97 (HF eval, valutazione finale)
- **Direct forecast**: lag features fissati all'anchor giorno 90, applicati a tutti i 7 giorni di test. Niente recursive forecast.
- **Valutazione**: WAPE e WPE solo sulle ore in-stock del test.

## Matrice 11 × 8

### 11 imputer (4 famiglie)

| Famiglia | Imputer |
|---|---|
| Baseline | No imputation |
| Naive aggregati | Media globale, Media condizionata, **Mediana globale**, Mediana condizionata |
| Time-series classici | Forward Fill, Seasonal Naive, Linear Interpolation |
| ML/Deep Learning | LGB imputer, DLinear, SAITS |

Mediana globale aggiunta successivamente per completare il design 2×2 (mean/median × global/conditional).

### 8 forecaster

| Famiglia | Forecaster |
|---|---|
| Naive | Global Mean, DoW Mean, MA (K=21), Naive Direct |
| ML tabellare | LGB (no lags), LGB (M5 lags) |
| Deep Learning | MLP (no lags), MLP (M5 lags), TFT (Temporal Fusion Transformer) |
| Foundation Model | Chronos-bolt (small) |

Lag M5-style: 11 lag features (lag_1d, lag_7d, lag_14d, rmean_7d, rmean_14d, rstd_7d, lag_dow, rmean_dow, daily_total_lag1, daily_total_rmean7, momentum_1d_7d).

## Risultati principali

### Top 5 combinazioni globali (WAPE mediana ascendente)

| Rank | Imputer + Forecaster | WAPE med | WPE med |
|:---:|---|:---:|:---:|
| 1 ★ | No imputation + Chronos-bolt | **1.0066** | -0.96 |
| 2 | Mediana globale + Chronos-bolt | 1.0090 | -0.95 |
| 3 | Mediana condizionata + Chronos-bolt | 1.0111 | -0.94 |
| 4 | SAITS + Chronos-bolt | 1.0117 | -0.94 |
| 5 | Seasonal Naive + Chronos-bolt | 1.0137 | -0.88 |

### Best per quartile di volume (analisi stratificata)

| Quartile | Combinazione raccomandata | WAPE | Robustezza statistica |
|---|---|:---:|:---:|
| Q1-Q2 (basso, 50% serie) | No imp + Chronos-bolt | 1.01 | **Robusta** (large effect, Cliff's δ > 0.6) |
| Q3 (medio-alto, 25%) | Mediana + MLP M5 | 1.00 | **Equivalenza statistica** (small/negligible vs alternative) |
| Q4 (alto, 25%) | LGB M5 + Mediana | **0.74** | **Equivalenza statistica** (Mediana ≈ MediaGlob ≈ MediaCond) |

### Findings principali

1. **Trade-off WAPE vs WPE**: Chronos-bolt domina il WAPE ma con WPE catastrofico (-0.96). MLP/LGB hanno WPE migliore (-0.18 a -0.38) a costo di WAPE peggiore.

2. **Crossover volume-dipendente**: su serie piccole (Q1) Chronos-bolt batte MLP (Cliff's δ = +0.85, 93% wins). Su serie grandi (Q4) MLP batte Chronos (δ = -0.64, 82% wins).

3. **Volume vs Stockout — effetti diversi**:
   - Volume: large effect su tutti (Cliff's δ ≈ 0.71-0.83)
   - Stockout: small/negligible (Cliff's δ ≈ 0.09-0.20)
   - Chronos-bolt è il forecaster meno sensibile a entrambe le dimensioni

4. **Imputer "structure-preserving" preservano il crossover** (No imp, Seasonal Naive, DLinear, SAITS, Mediana cond). Imputer "media-based" lo rovinano (Media cond, Media glob, LGB imputer, Forward Fill).

5. **Traccia A non predice Traccia B**: Mediana cond ha il miglior WAPE_recovery (0.846) ma SAITS (4° su recovery) batte Mediana per Chronos-bolt come forecaster downstream.

### Test statistici eseguiti (~600 totali con 11 imputer)

| Test | A cosa serve | Numero |
|---|---|:---:|
| Wilcoxon paired (imputer pair × forecaster) | significatività imputer | 270 |
| Wilcoxon best combo vs alternatives | best globale | 28 |
| Cliff's δ stratificato (cell × dimension) | sensibilità volume/stockout | 160 |
| Crossover Chronos vs ML (per quartile) | crossover paired | 48 |

Con N=50.000 serie, p-value ≈ 0 in quasi tutti i test. **Effect size (Cliff's δ, rank-biserial r) è la metrica discriminante**.

## Struttura del codice (cartella `pipeline/`)

```
pipeline/
├── 01_fase_a_naive.py                         # Baseline naive forecaster
├── 02_fase_a_lgb.py                           # LGB forecaster
├── 03_fase_a_mlp.py                           # MLP forecaster
├── 04_fase_b1_imputation_naive_ml.py          # 4 imputer (media glob/cond, mediana cond, LGB)
├── 05_fase_b1_imputation_dlinear.py           # DLinear imputer (PyPOTS)
├── 06_fase_b2_forecast_naive.py               # Naive × completed_sales
├── 07_fase_b2_forecast_lgb.py <imputer>       # LGB M5 × imputer
├── 08_fase_b2_forecast_mlp.py <imputer>       # MLP M5 × imputer
├── 09_fase_c_analysis.py                      # Analisi globale + heatmap + Wilcoxon
├── 10_fase_b2_forecast_chronos.py <imputer>   # Chronos-bolt × imputer
├── 11_fase_c_stratified.py                    # Stratificazione 4×4 (subset)
├── 12_fase_c_volume_tests.py                  # Test volume (deprecato, sostituito da 13)
├── 13_fase_c_volume_stockout_tests.py         # Test volume+stockout (Cliff's δ, JT)
├── 14_fase_b1_imputation_classic.py           # Forward fill, Seasonal Naive, Linear Interp
├── 14b_fase_b2_forecast_naive_classic.py      # Naive × imputer classici
├── 15_fase_c_stratified_full.py               # Opzione A+B + crossover Chronos vs ML
├── 16_fase_b1_imputation_saits.py             # SAITS imputer (training)
├── 16b_saits_predict.py                       # SAITS predict (continuation)
├── 17_presentation_figures.py                 # 3 figure ad-hoc per presentazione
└── 18_fase_b1_imputation_mediana_glob.py      # Mediana globale imputer
```

## Figure prodotte (cartella `pipeline/figures/`)

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

## File risultati chiave (cartella `pipeline/results/`)

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

## Decisioni metodologiche del pivot

1. **Direct forecast** per tutti i forecaster — i lag sono fissati all'anchor giorno 90.
2. **Niente HP optimization** — tutti i forecaster usano HP fissi e standardizzati.
3. **Split temporale** per i naive/ML, **split per-serie 80/20** per DLinear/SAITS (limitazione di PyPOTS).
4. **Mediana globale aggiunta a posteriori** per completare il design 2×2 dei naive aggregati.
5. **Effect size (Cliff's δ) è la metrica primaria**, non il p-value (che è ~0 con N=50K).

## Limitazioni dichiarate

- 1 dataset, retail cinese deperibile (3.5 mesi)
- Orizzonte breve (7 giorni)
- HP non ottimizzati (uguali per tutte le celle)
- DLinear/SAITS non retrained su tutte le 50K serie (asimmetria con i naive/ML)
- City 0 contiene 52% del dataset (sbilanciamento geografico)

## Stato attuale (data: 2026-05-12)

- ✅ Matrice **88 celle completata** (11 imputer × 8 forecaster)
- ✅ **TFT aggiunto come 9° forecaster su TUTTI gli 11 imputer (50K serie)** → matrice 72 celle disponibili totali (11×7 attivi)
- ✅ **TFT × SAITS è il rank 1 globale** (WAPE med = 0.9850) con MSE
- ✅ **Re-train completato: loss uniformata a MAE per tutti i forecaster ML (24 celle)** — vedi sezione "Loss uniformity" più sotto
- ✅ Pattern emergente: con MAE la **scelta dell'imputer collassa** (spread WAPE_med 0.014 vs 0.105 con MSE)
- ✅ Analisi statistica completa (Wilcoxon + Cliff's δ + JT + Spearman, ~600 test)
- ✅ Analisi stratificata volume × stockout (4×4 quartili, 16 gruppi)
- ⏳ **Prossimi passi pianificati** (in ordine):
  - **Fase 1**: training iTransformer + TimesNet + 5 forecaster cad (~16h compute)
  - **Fase 2**: training CSDI + ImputeFormer + 5 forecaster cad (~22h compute)
  - **Aggregazione finale**: matrice ~100 celle (15 imputer × 9 forecaster)
  - **HP sensitivity**: pacchetto "essenziale" (MLP architecture, HP paper-aligned, bootstrap CI, ~12h)
- ❌ **Caveat aperti**: bug `25_tft_full_training.py` (Best val_loss riporta last invece di min), sensitivity max_epochs=20 su linear_interp (cap raggiunto)

### Risultati TFT × 11 imputer (50K serie, test gg 91-97, in-stock filter)

Ordinati per WAPE mediana ascendente:

| Imputer | WAPE pool | WAPE med | WPE pool | WPE med | Rank globale |
|---|:---:|:---:|:---:|:---:|:---:|
| **saits** ★ | 0.8797 | **0.9850** | -0.6920 | -0.7925 | **1** |
| **seasonal_naive** 🆕 | 0.8340 | 0.9888 | -0.6035 | -0.8019 | 2 |
| no_imp | 0.8787 | 0.9954 | -0.7184 | -0.8395 | 3 |
| dlinear | 0.8921 | 1.0017 | -0.6959 | -0.7766 | 4 |
| forward_fill 🆕 | 0.8657 | 1.0132 | -0.4719 | -0.6050 | 9 |
| mediana_glob | 1.0993 | 1.0204 | -0.5431 | -0.6454 | 11 |
| mediana_cond | 1.0972 | 1.0206 | -0.5429 | -0.6379 | 12 |
| linear_interp 🆕 | 0.8822 | 1.0214 | -0.5228 | -0.5708 | 13 |
| media_cond 🆕 | 1.1208 | 1.0284 | -0.4641 | -0.5207 | 15 |
| media_glob 🆕 | 1.1170 | 1.0513 | -0.4511 | -0.4999 | 19 |
| lgb 🆕 | 1.1224 | 1.0612 | -0.4305 | -0.4683 | 20 |

🆕 = imputer aggiunti nel completamento (Maggio 2026-05-08 → 05-11).

### Top 5 globale (con tutti i TFT inclusi)

| Rank | Cella | WAPE med | WPE med |
|:---:|---|:---:|:---:|
| **1** | **saits + TFT** ★ | **0.9850** | -0.7925 |
| 2 | seasonal_naive + TFT 🆕 | 0.9888 | -0.8019 |
| 3 | no_imp + TFT | 0.9954 | -0.8395 |
| 4 | dlinear + TFT | 1.0016 | -0.7766 |
| 5 | no_imp + Chronos-bolt | 1.0067 | -0.9601 |

**TFT domina i top 4 con 4 imputer diversi**. Tra rank 1-10, 7/10 celle sono TFT-based.

### Best per quartile aggiornato (matrice 72 celle, post completamento TFT)

| Quartile | Cella best | WAPE med | Cambio? | Note |
|:---:|---|:---:|:---:|---|
| Q1 (basso vol) | **seasonal_naive + TFT** 🆕 | 1.0004 | sì | Cambio rispetto al best precedente (no_imp+TFT) |
| Q2 | **seasonal_naive + TFT** 🆕 | 1.0006 | sì | Cambio rispetto al best precedente (saits+TFT) |
| Q3 (medio-alto) | saits + TFT | 0.9773 | no | Invariato dal best precedente |
| Q4 (alto vol) | mediana_cond + LGB_M5 | 0.7434 | no | LGB_M5 mantiene il dominio (TFT perde con δ=-0.64) |

**Scoperta chiave**: con seasonal_naive (imputer aggiunto) TFT **conquista anche Q1 e Q2** che prima dominava ma con altri imputer. seasonal_naive replica la stagionalità settimanale, particolarmente utile su serie sparse.

### Modelli statisticamente equivalenti al best (|Cliff's δ| < 0.147 = negligible)

> **⚠ SUPERATO** — Sezione storica basata su Cliff δ + soglia Romano applicata come decision rule
> (vecchia metodologia, pre-HPO, pre-MAE re-train, pre-Friedman). Il framework attuale del paper
> è Friedman + Kendall W + Nemenyi CD — vedere sezione "RQ4 — Best cell + equivalence set" più sotto.

**Globalmente** (best = saits + TFT, WAPE_med = 0.9850) — **1 cella equivalente**:
- no_imp + TFT (δ = +0.093)
- Già seasonal_naive + TFT (δ = +0.255 small) → equivalenza solo con no_imp+TFT.

**Q1** (best = seasonal_naive + TFT, WAPE_med = 1.0004) — **0 equivalenti**:
- Tutte le altre celle hanno δ ≥ 0.147 (small effect)
- Più vicino: no_imp+TFT (δ ≈ +0.15 small/borderline)

**Q2** (best = seasonal_naive + TFT, WAPE_med = 1.0006) — **1 equivalente**:
- saits + TFT (δ = +0.139)

**Q3** (best = saits + TFT, WAPE_med = 0.9773) — **13 equivalenti**:
- seasonal_naive + TFT (δ = -0.13)
- mediana_cond + TFT, mediana_glob + TFT (δ ≈ +0.01-0.03)
- media_glob + TFT (δ = +0.11), linear_interp + TFT (δ = +0.12)
- 8 celle MLP_M5 con vari imputer (δ +0.09 a +0.14)

**Q4** (best = mediana_cond + LGB_M5, WAPE_med = 0.7434) — **14 equivalenti** (regime "saturato"):
- 9 celle MLP_M5 con varie imputer
- 5 celle LGB_M5 con varie imputer
- **Nessuna cella TFT entra in equivalenza** — TFT non è competitivo in Q4

### Pattern di equivalenza per quartile (aggiornato)

| Quartile | # equivalenti | Caratteristica del cluster |
|:---:|:---:|---|
| Q1 | **0** | seasonal_naive+TFT **unico best**, nessun pari |
| Q2 | 1 | 2 celle TFT-based (seasonal_naive + saits) |
| Q3 | 13 | TFT-based (5 imputer) + MLP_M5 (8 imputer) — regime molto competitivo |
| Q4 | 14 | LGB_M5 e MLP_M5 con qualunque imputer (TFT escluso) |

**Implicazioni**:
1. In **basso volume (Q1-Q2)** la scelta del forecaster (TFT) E dell'imputer (seasonal_naive) **entrambi contano**: pochissime alternative equivalenti
2. In **medio-alto volume (Q3)** molte combinazioni equivalenti: il volume di dati riduce l'importanza delle scelte specifiche
3. In **alto volume (Q4)** la **scelta dell'imputer è ininfluente**, solo il forecaster conta (LGB_M5/MLP_M5 wins)

### Caveat metodologici aperti

1. **Bug nello script `25_tft_full_training.py`**: `Best val_loss` riporta last val_loss invece del minimum (verificato su media_glob). Da fixare.
2. **`linear_interp` ha raggiunto il cap max_epochs=10** con val_loss ancora in miglioramento (0.0670 epoca 9 < 0.0672 epoca 6). Sensitivity max_epochs=20 raccomandata.
3. **Imputer del paper baseline non ancora inclusi**: iTransformer, TimesNet, CSDI, ImputeFormer (4/7 mancanti vs il paper Liu et al. 2025). Script pronti (`27_fase_b1_imputation_itransformer.py`, `28_fase_b1_imputation_timesnet.py`).
4. **MLP architecture non tuned**: scelta [128, 64] è ragionevole ma non ottimizzata. Sensitivity analysis pianificata.

## Loss uniformity: re-train MAE (data: 2026-05-12)

### Contesto

Confounder identificato: pre-fix, **MLP** trainato con MSE, **LGB** con `objective='regression'` (= MSE in LightGBM, nonostante metric='mae'), **TFT** con MAE, **Chronos** con quantile loss (pre-trained). Asimmetria della loss confondeva il confronto tra forecaster.

### Fix applicato

Tutti i forecaster ML uniformati a **MAE training loss**:
- `pipeline/02_fase_a_lgb.py`, `pipeline/07_fase_b2_forecast_lgb.py`: `objective: 'regression'` → `'regression_l1'`
- `pipeline/03_fase_a_mlp.py`, `pipeline/08_fase_b2_forecast_mlp.py`: `mse_loss` → `l1_loss`
- TFT (`25_tft_full_training.py`): già MAE
- Chronos: non modificabile (pre-trained), uso q=0.5 (mediana) come surrogate MAE-aligned

### Risultati del re-train (24 celle ri-trainate)

**Cells**: 4 Fase A (lgb_nolags, lgb_m5lags, mlp_nolags, mlp_m5lags) + 10 LGB_M5 × imputer + 10 MLP_M5 × imputer.

**Pattern emergente**:

| Metrica | Range MSE (vecchio) | Range MAE (nuovo) | Compressione |
|---|:---:|:---:|:---:|
| WAPE_med across cells | 0.74 - 1.30 (spread 0.56) | 0.98 - 1.00 (spread 0.014) | **-97%** |
| WPE_med across cells | -0.18 - -0.40 (spread 0.22) | -0.85 - -0.95 (spread 0.10) | **-55%** |

**Δ medio MAE-MSE** per cella:
- WAPE_med: **-10% (migliora con MAE, atteso, MAE→WAPE allineata)**
- WPE_med: **3-5× più negativo (peggiora con MAE, atteso, MAE→mediana su dati right-skewed)**

### Implicazione fondamentale per il paper

> **L'effetto dell'imputer "collassa" quando il forecaster usa la loss allineata alla metrica di valutazione (MAE per WAPE)**. Con MAE: spread WAPE cross-imputer = 0.014 (rumore). Con MSE: spread = 0.56 (effetto visibile).

Questo è il **finding chiave** del paper: i benefici dell'imputation di stockout sono **più piccoli di quanto si pensa** quando si fa la scelta corretta di loss.

### Backup MSE conservati

I parquet MSE-based sono in `pipeline/results/_mse_backup/` (24 file). Servono per:
1. Confronto MAE vs MSE come sensitivity analysis nel paper
2. Eventualmente rifare i numeri se cambiamo idea sulla loss

### Ranking post-MAE (matrice 72 celle, dopo uniformazione loss)

Salvato in `pipeline/results/ranked_combinations_MAE_uniform.parquet` e `ranked_combinations_per_quartile_MAE_uniform.parquet`.

#### Top 10 globale (post-MAE)

| Rank | Cell | WAPE med | WPE med |
|:---:|---|:---:|:---:|
| **1** | **lgb + MLP_M5** | **0.9811** | -0.898 |
| 2 | seasonal_naive + MLP_M5 | 0.9842 | -0.909 |
| 3 | media_glob + MLP_M5 | 0.9842 | -0.912 |
| **4** | saits + TFT (era rank 1 pre-MAE) | 0.9850 | -0.793 |
| 5 | media_cond + MLP_M5 | 0.9851 | -0.919 |
| 6 | mediana_glob + MLP_M5 | 0.9854 | -0.903 |
| 7 | dlinear + MLP_M5 | 0.9858 | -0.898 |
| 8 | media_glob + LGB_M5 | 0.9863 | -0.905 |
| 9 | media_cond + LGB_M5 | 0.9865 | -0.903 |
| 10 | mediana_glob + LGB_M5 | 0.9878 | -0.920 |

**Cambio principale**: `saits + TFT` (ex rank 1) ora è **rank 4**. Top 3 sono tutti MLP_M5. Top 10 quasi tutto MLP_M5 + LGB_M5.

#### Best per quartile (post-MAE) — DRASTICAMENTE CAMBIATO

| Quartile | Pre-MAE (MSE) | Post-MAE (MAE) | WAPE post | WPE post | Cambio |
|:---:|---|---|:---:|:---:|---|
| **Q1** (basso vol) | no_imp + TFT (1.007) | **degenere** — tutti = 1.000 con WPE=-1.0 | 1.000 | -1.000 | MAE collassa Q1 → predice 0 |
| **Q2** | seasonal_naive + TFT (1.001) | **media_cond + LGB_M5** | 0.9969 | -0.948 | TFT ↓, LGB_M5 ↑ |
| **Q3** (medio-alto) | saits + TFT (0.977) | **lgb + MLP_M5** | 0.9559 | -0.857 | TFT ↓ rank 1 → ~13, MLP_M5 ↑ |
| **Q4** (alto vol) | mediana_cond + LGB_M5 (0.743) | **mediana_glob + Global_Mean** 🆕 | **0.7619** | -0.141 | LGB_M5 ↓, **NAIVE ↑** |

**Sorprese**:
1. **Q1 collassa con MAE**: con basso volume la mediana = 0, MAE predice 0 sistematicamente. WAPE=1, WPE=-1. Degenere. **MAE non è adeguata per Q1**.
2. **Q4: i naive battono ML**: i forecaster naive (Global Mean, DoW Mean, MA K=21) non hanno loss da uniformare e mantengono WPE basso (~-0.14). I ML con MAE invece sotto-stimano (-0.46 per MLP_M5). I naive vincono per WAPE pool.

#### Implicazioni rivisitate per il paper

1. **Decision tree per practitioner aggiornato**:
   - **Q1 (basso volume)**: MAE non adeguata (degenere). Usare MSE, Huber, o loss combinata.
   - **Q2 (medio-basso)**: LGB_M5 con MAE
   - **Q3 (medio-alto)**: MLP_M5 con MAE
   - **Q4 (alto)**: **Naive globali** con buon imputer (mediana_glob)

2. **La loss conta tantissimo**:
   - Rank globale cambia completamente (saits+TFT → lgb+MLP_M5)
   - Q4 si sposta da ML (MSE) a naive (loss-agnostic)
   - Q1 diventa degenere se loss MAE su low volume

3. **Trade-off WAPE vs WPE acuto**:
   - Naive in Q4: WAPE 0.762, WPE -0.14 (poco bias)
   - LGB_M5 in Q4: WAPE 0.764, WPE -0.46 (più bias)
   - Differenza WAPE trascurabile, WPE drammatica

4. **Imputer effect resta smorzato dalla loss**:
   - Top 10 globale: WAPE range 0.981-0.988 (spread 0.007)
   - Conferma: con MAE la scelta dell'imputer è quasi irrilevante per il ranking, ma può ancora contare in Q4 (mediana_glob domina)

5. **Effetto interessante in Q4**: `mediana_glob` (mediana globale, imputer più semplice possibile) batte tutti gli altri imputer accoppiati a naive forecaster. **Più semplice è meglio** nel regime alto volume.

## HPO + re-train (data: 2026-06-08)

### HPO Optuna completato per 5 forecaster (TPE + MedianPruner)

Storage SQLite, load_if_exists per crash recovery. Metrica: val_WAPE_med per-serie in-stock (min_hours=34). Loss: MAE training, valutazione val days 84-90.

**Spazio HP esplorato**:

| Forecaster | N trials | Best val_WAPE_med | HP chiave |
|---|---|---|---|
| LGB_no_lags | 30 | **0.9788** | num_leaves=63, lr=0.107, min_child=451, bagging=0.2, feature=0.5 |
| LGB_M5 | 30 | **0.9854** | num_leaves=110, lr=0.184, min_child=166, bagging=0.2, feature=0.6 |
| MLP_no_lags | 45 | **0.9781** | `[256,128]` dropout=0.10 lr=8.17e-4 bs=1024 emb_scale=1.5 wd=1.38e-6 |
| **MLP_M5** | 45 | **0.9724** ★ | `[128,64]` dropout=0.0 lr=3.51e-3 bs=1024 emb_scale=2.0 wd=1.59e-6 |
| TFT | 30 | **0.9891** | head_dim=8, heads=4, hidden=32, bs=256, lr=2.47e-3 |

**MLP_M5 = best globale val (0.9724)**. Spazio MLP esteso post-HPO con single-layer architectures `[64]`, `[128]`, `[256]` (TPE le ha esplorate ma multi-layer vince).

**Constraint OOM TFT**: spazio HP ammette combinazioni `head_dim × heads` fino a 256, ma su 16 GB RAM (single-machine, CPU-only) hidden > 32 crashava (SIGKILL durante Trial 1). Soluzione: early `optuna.TrialPruned()` per hidden_size > 32 → 21/30 trial pruned, 8 completati su 3 combinazioni valide.

**Accelerazioni TFT applicate post-Trial 0** (Trial 0 ~88 min, poi degradò):
- `MAX_EPOCHS`: 25 → 6
- `PATIENCE`: 5 → 2
- `MAX_TRAIN_SAMPLES`: 200K → 100K (window subsample)
- `batch_size`: [32,64,128,256] → [256,512,1024]

### Re-train celle con HPO HPs (env var `HPO_VARIANT=1`)

Modificati 5 script per leggere HP da `pipeline/results/hpo_*_best.json` e salvare output con suffisso `_hpo`:
- `02_fase_a_lgb.py`, `03_fase_a_mlp.py` (Fase A no_imp baselines)
- `07_fase_b2_forecast_lgb.py`, `08_fase_b2_forecast_mlp.py` (M5 imputer cells)
- `25_tft_full_training.py` (TFT imputer cells)

Wrapper batch: `pipeline/run_retrain_hpo.sh` + `pipeline/run_retrain_tft_hpo.sh`.

**Risultati (34 celle ri-trainate, 30/34 migliorano)**:

| Forecaster | Celle | Migliorate | Δ medio hourly_wape_med |
|---|---|---|---|
| LGB_M5 × 10 imputer | 10 | **10/10** | -0.62% |
| MLP_M5 × 10 imputer | 10 | 9/10 | -0.50% (`media_cond` +0.43%) |
| Fase A no_imp (LGB×2 + MLP×2) | 4 | **4/4** | -0.63% |
| TFT × 10 imputer | 10 | 7/10 | -1.55% (sui 7); regressioni `no_imp` +0.46%, `saits` +0.66%, `seasonal_naive` +0.48% |

**Sanity check pre-rollout** (saits + MLP_M5): val→test consistent, no overfitting. WAPE_h med 0.9882 → 0.9790 (-0.93%), WPE bias ridotto -0.91 → -0.88.

**Compute totale**: ~50h. LGB cells ~25min, MLP cells ~20min, TFT cells ~99min medio (variabilità 60-243 min per cella, anche per PC sospensioni).

### Implicazioni per il paper

1. **HPO migliora consistentemente** (88% delle celle) ma effetto **piccolo** (-0.6% medio) — coerente con finding del paper: con MAE training loss la scelta degli iperparametri è quasi irrilevante per il ranking imputer.
2. **TFT non recupera competitività** con HPO: val WAPE 0.989 vs MLP_M5 0.972. Il bottleneck è architetturale (modello sequenziale su 50K serie eterogenee), non HP.
3. **Spread imputer ancora più compresso post-HPO**: la pulizia HPs uniforma le performance → l'imputer effect domina su tutto tranne forse Q4 (alta concentrazione di volume).

## FASE 1 + FASE 2: Nuovi imputer (data: 2026-06-10)

Aggiunti 4 imputer dal paper baseline (Liu et al. 2025): iTransformer, TimesNet, ImputeFormer, CSDI. Tutti via PyPOTS, con HPs slim per accelerare su CPU (MPS instabile per alcuni modelli — NaN errors su TimesNet).

### Imputer training

| Imputer | WAPE_recovery | WPE_recovery | Params | Note |
|---|:---:|:---:|:---:|---|
| ImputeFormer | **0.8666** ★ | -0.7415 | slim | Best recovery |
| iTransformer | 0.9302 | -0.4501 | medium | |
| TimesNet | 1.0405 | -0.8650 | 36K | CPU only (MPS NaN); slim model |
| CSDI | 1.3951 | +0.0794 | 3.7K | Ultra-slim diffusion; over-imputes; bassissimo bias |

CSDI ha bias quasi-zero (WPE_recovery +0.08) ma alto WAPE (1.39) — sovra-stima ovunque, compensando l'under-bias degli altri. Pattern interessante per ensembling.

### Forecaster cells (4 imputer × 3 forecaster = 12 cell pianificate, 10 ottenute)

- **FASE 1 OK**: iTransformer + TimesNet × {LGB_M5, MLP_M5, TFT} = 6 cell ✓
- **FASE 2 parziale**: CSDI + ImputeFormer × {LGB_M5, MLP_M5} = 4 cell ✓
- **FASE 2 KO**: CSDI/ImputeFormer × TFT = 2 cell **OOM-killed** (memoria sistema sotto pressione dopo CSDI training; long_data ~15GB su 16GB Mac, kill esterno macOS memory pressure)

Le 4 cell TFT mancanti sono **escluse dalla matrice finale**. Decisione: CSDI ha solo LGB+MLP cells (non TFT) ma viene incluso comunque — alternativa "drop CSDI" considerata e scartata perché 2/3 cell sono comunque informative.

### Matrice finale: 45 cell HPO totali

```
Top 5 (WAPE_h median, lower = better):
1. TimesNet + MLP_M5     = 0.9731  ★ BEST GLOBALE
2. LGB imputer + MLP_M5  = 0.9759
3. forward_fill + MLP_M5 = 0.9764
4. mediana_cond + MLP_M5 = 0.9773
5. CSDI + MLP_M5         = 0.9776

Bottom 5: tutti TFT cells (1.00-1.02 WAPE_med)
```

**Pattern emergenti**:
1. **MLP_M5 domina la testa**: top 6 sono tutti MLP_M5 con imputer diversi. Lo spread (0.9731-0.9793) è solo **0.6%** — coerente con finding: scelta dell'imputer è secondaria con loss MAE allineata.
2. **TFT è worst**: con HPO+MAE TFT non recupera competitività. Con MSE era rank 1 (val=0.989 era best pre-HPO), ma su test con WAPE_med non riesce a superare MLP_M5 lite. La causa è strutturale (TFT su 50K serie eterogenee+CPU).
3. **TimesNet (peggior recovery) → miglior forecasting**: la cella TimesNet+MLP_M5 batte tutte. Conferma che la qualità dell'imputation NON è proxy della qualità del forecasting downstream — finding metodologico rilevante.
4. **CSDI (over-imputer) competitivo**: rank 5 con MLP_M5, bias quasi-zero (WPE -0.90 vs altri -0.89). Differenza minima ma supporta tesi che over-imputation è meno deleteria di under-imputation.

### Limitazioni residue

- **TFT × CSDI/ImputeFormer mancano**: incompletezza nella matrice. Re-run richiede liberazione RAM (riavvio PC).
- **Imputer slim non comparabili al paper baseline**: HPs ridotti per esigenze CPU (TimesNet 36K params vs originali 2.3M).
- **Sample size training imputer**: 80/20 split per-serie su 50K. PyPOTS overhead alto.

## Risposte alle Research Questions (data: 2026-06-11, riorganizzato)

Matrice finale: **113 cell** (TimesFM completato su 14 imputer, allineato agli altri forecaster). Framework statistico unico per il paper: **Friedman + Kendall's W + Nemenyi CD** (Demšar 2006, JMLR).

---

## Sezione 1 — Best cell + equivalence set (framework Friedman + W + CD)

**Framework metodologico unico**:
- **Friedman χ²**: test rejection H0 "tutte le k celle hanno distribuzione di rank uguale".
- **Kendall's W** = χ² / [N · (k−1)] ∈ [0,1]: effect size globale (concordanza ranking cross-serie).
  - W < 0.1 = negligible · 0.1–0.3 = small · 0.3–0.5 = moderate · ≥ 0.5 = large.
- **Nemenyi CD** post-hoc: due celle indistinguibili sse |Δ mean_rank| ≤ CD.
  - CD = q_α(k,∞)/√2 · √(k(k+1)/(6N)) con α=0.05.

Lo stesso framework viene applicato a 3 scope diversi: globale (1.1), stratificato per quartile di volume (1.2), ristretto per forecaster (1.3).

**Niente TOST**, niente soglia Cliff δ < 0.147 come decision rule (scripts 43, 44, 47 → supplementary).
Cliff δ resta usato come effect size descrittivo nelle sezioni 2 e 3 (mai come decision rule).

### 1.1 Globale — il best assoluto sulla matrice intera (script 45)

Matrice [49.939 serie × 109 celle]. Output principale del paper.

| Livello | k | N | **Kendall W** | CD | Best cell | # CD-equiv |
|---|:---:|:---:|:---:|:---:|---|:---:|
| **Globale** | 113 | 49,939 | **0.454 (moderate)** | 0.903 | **itransformer__MLP_M5** | **2** |

Equivalence set (2 cells):
1. `itransformer__mlp_m5lags` (mean rank 22.27, best)
2. `lgb__mlp_m5lags` (mean rank 22.78, Δ = 0.51 ≤ CD)

**Finding 1.1 (a)**: il best globale è `itransformer__MLP_M5`. Solo `lgb__MLP_M5` gli sta statisticamente alla pari. Entrambe le 2 celle sono **MLP_M5**: la famiglia di forecaster vincente è isolata. Il ranking è generalizzabile (W = 0.454 moderate).

#### 1.1 (b) Trade-off WAPE × |WPE|: Pareto frontier globale (script 35)

Pareto su `(WAPE_h_med, |WPE_h_med|)` — accuracy vs bias. Frontier = celle non-dominate.

**26 / 113 cells Pareto-optimal**. I tre punti di riferimento:

| Ruolo | Cella | WAPE | |WPE| |
|---|---|:---:|:---:|
| Best WAPE (accuracy-extreme) | `timesnet__MLP_M5` | **0.973** | 0.886 |
| Knee point (trade-off bilanciato) | `mediana_glob__dow_mean` | 1.101 | **0.190** |
| Min |WPE| (bias-extreme) | `linear_interp__timesfm` 🆕 | 1.295 | **0.061** |

**Estremi della frontier** (per dare un'idea della spreadinside):
- Lato WAPE: `MLP_M5/TFT` con vari imputer (WAPE 0.97–1.00, |WPE| 0.77–0.89).
- Lato |WPE|: **naive aggregati** (Global/DoW/MA con vari imputer, |WPE| 0.06–0.19, WAPE 1.10–1.16).
- TimesFM entra solo in posizione min-|WPE| (con linear_interp).

**Finding 1.1 (b)**: il trade-off è **strutturale**. Il best-WAPE ha |WPE| sempre elevato (≥ 0.77) — i forecaster ML/DL **sotto-stimano sistematicamente**. Per ridurre il bias servono naive aggregati che pagano in WAPE. Il knee point `mediana_glob__dow_mean` rappresenta un compromesso ragionevole per practitioner che valuta sia accuracy che bias.

### 1.2 Per regime di volume — robustezza del best (script 46)

Stratificazione per quartile di volume (Q1-Q4, ~12.500 serie per Q).

| Q | Range vol | Friedman best | Kendall W | CD | # CD-equiv |
|---|---|---|:---:|:---:|:---:|
| Q1 (basso) | [11, 40] | **lgb__MLP_M5** | 0.653 (large) | 1.804 | 12 |
| Q2 | (40, 54] | **lgb__MLP_M5** | 0.586 (large) | 1.805 | 13 |
| Q3 (medio-alto) | (54, 86] | **itransformer__MLP_M5** | 0.417 (moderate) | 1.806 | 4 |
| Q4 (alto) | (86, 5326] | **itransformer__MLP_M5** | 0.396 (moderate) | 1.807 | 2 |

**Finding 1.2 (a)**: crossover **soft**. La famiglia di forecaster vincente (MLP_M5) è **invariata** in tutti i regimi; l'imputer ottimale cambia tra `lgb` (basso volume Q1/Q2) e `itransformer` (alto volume Q3/Q4). Il regime più discriminante è Q4 (solo 2 celle CD-equivalenti); Q1 è il più saturato (12 equivalenti).

**Nota su W vs CD-equiv set size**: W e #equiv non sono ridondanti. Q1 ha W large + molti equiv (separazione forte tra famiglie ma debole dentro MLP_M5); Q4 ha W moderate + pochi equiv (naive competono → meno accordo globale, ma MLP_M5 differenziato → top isolato).

#### 1.2 (b) Pareto frontier per quartile (script 36)

Stessa metrica (WAPE × |WPE|) applicata dentro ciascun quartile. La frontier **cambia composizione** col regime.

| Q | # Pareto | Top-WAPE | Min-|WPE| | Famiglie sulla frontier |
|---|:---:|---|---|---|
| Q1 (basso vol) | **28** / 113 | `mediana_cond__LGB_M5` (1.000) | `mediana_cond__TFT` (\|WPE\|=0.57) | LGB_M5, MLP_M5, TFT (dominante in coda low-bias) |
| Q2 | 22 | `mediana_cond__LGB_M5` (0.995) | `forward_fill__Chronos-bolt` (\|WPE\|=0.39) | LGB_M5, MLP_M5, TFT, Chronos al margine |
| Q3 (medio-alto) | 15 | `lgb__MLP_M5` (0.958) | `mediana_cond__MA_K21` (\|WPE\|=0.13) | MLP_M5, TFT in testa; **naive emergono al low-bias** |
| Q4 (alto vol) | **12** / 113 | `itransformer__MLP_M5` (**0.757**) | `media_cond__MA_K21` (\|WPE\|=**0.06**) | MLP_M5 (top WAPE), naive (low bias) — gap minimo |

**Finding 1.2 (b)**: tre pattern emergono dalla Pareto stratificata:

1. **# Pareto-optimal decresce con il volume** (Q1: 28 → Q4: 12) → stesso pattern del CD-equiv set: alto volume = più discriminante.
2. **In Q1/Q2 (basso volume) la frontier è popolata da TFT** che occupa la coda low-bias (linear_interp__TFT, mediana_glob__TFT) → TFT è il forecaster trade-off in basso volume.
3. **In Q3/Q4 (alto volume) la frontier si specializza**: MLP_M5/TFT al top-WAPE, **naive aggregati (Global/DoW/MA) al low-bias**. In Q4 il knee diventa sub-optimale rispetto a Q1: gap WAPE↔|WPE| si comprime (0.76 vs 0.06 in Q4 — naive batte ML su |WPE| pagando solo +3% WAPE!).
4. **Chronos** appare solo nella Pareto di Q2 (forward_fill__Chronos-bolt) → posizione di nicchia. **TimesFM** entra solo nella Pareto globale (min |WPE| con linear_interp) e mai in quella per quartile.

### 1.3 Per forecaster — l'imputation aiuta vs no_imp? (script 48, 49)

Per ciascun forecaster, framework applicato sulla sottomatrice ristretta ai suoi k≈10-14 imputer (incluso `no_imp`). Domanda specifica: `no_imp` è dentro l'equivalence set CD del Friedman best?

| Forecaster | k | Friedman best | no_imp pos. | Δrank | Kendall W | Cat. W | Imputer aiuta? |
|---|:---:|---|:---:|:---:|:---:|---|---|
| Global Mean | 14 | mediana_glob | 6°/14 | +2.444 | **0.469** | moderate | **SÌ generalizzabile** |
| MA_K21 | 14 | mediana_glob | 4°/14 | +1.216 | **0.464** | moderate | **SÌ generalizzabile** |
| DoW Mean | 14 | mediana_glob | 5°/14 | +1.973 | **0.445** | moderate | **SÌ generalizzabile** |
| **TimesFM** | 14 | imputeformer | 2°/14 | +0.163 | **0.174** | small | **SÌ generalizzabile** |
| **Chronos-bolt** | 14 | imputeformer | 2°/14 | +1.339 | **0.222** | small | **SÌ generalizzabile** |
| **TFT** | 13 | dlinear | 2°/13 | +0.459 | **0.220** | small | **SÌ generalizzabile** |
| **MLP_M5** | 14 | itransformer | 9°/14 | +0.592 | **0.029** | negligible | NO (effetto marginale) |
| **LGB_M5** | 14 | mediana_glob | 8°/14 | +0.401 | **0.009** | negligible | NO (effetto marginale) |

**Pattern per quartile** (no_imp CD-equiv ⇒ imputer NON aiuta):

| Q | Imputer non aiuta | Imputer aiuta |
|---|---|---|
| Q1 (basso) | LGB_M5, MLP_M5, TimesFM | Chronos, TFT, naive |
| Q2 | LGB_M5, MLP_M5 | tutti gli altri |
| Q3-Q4 | nessuno | tutti |

**Finding 1.3**: dicotomia chiara basata sull'architettura del forecaster:
- **Imputer-sensitive** (Chronos, TFT, TimesFM, naive aggregati): l'imputer **aiuta sempre**, ranking generalizzabile (W ≥ 0.22). Best imputer coerente: **imputeformer** per foundation models, **mediana_glob** per naive, **dlinear** per TFT.
- **Imputer-irrelevant** (MLP_M5, LGB_M5): W ≈ 0 in ogni regime → il ranking degli imputer è praticamente random cross-serie → in basso volume `no_imp` è anche CD-equivalent al best (l'imputer non aiuta neppure statisticamente).

#### 1.3.1 Robustness sensitivity — training loss MAE vs MSE (script 38)

Per verificare che il finding "W ≈ 0 per MLP_M5/LGB_M5" non sia un artefatto della loss MSE non-allineata alla metrica WAPE, abbiamo ri-trainato 24 celle (LGB_M5, LGB_nolags, MLP_M5, MLP_nolags × 10 imputer) con loss MAE. Confronto pairwise Wilcoxon + Cliff δ.

**Effetto principale (MAE > MSE)**:
- MAE migliora WAPE in modo deterministico in tutte le 24 celle (Δ WAPE medio: LGB_M5 −0.128, MLP_M5 −0.099).
- Vantaggio LGB su MLP si comprime del 50% sotto MAE (spread inter-forecaster: 0.05 → 0.024).

**Effetto sull'imputer impact (claim chiave da validare)**:
- Spread inter-imputer entro ciascun forecaster: MSE ~0.01–0.02 vs MAE ~0.008–0.024.
- **Lo spread rimane piccolo sotto entrambe le loss** → il finding "imputer effect negligible per MLP_M5/LGB_M5" è strutturale, non causato dalla loss.

**Conclusione 1.3.1**: il claim 1.3 ("imputer is irrelevant for MLP_M5/LGB_M5") sopravvive al sensitivity test sulla loss → finding robusto.

---

## Sezione 2 — Recovery quality predice forecasting? (script 42 + 42b)

Per ogni serie i (i=1..50K), Spearman ρ_i tra `WAPE_recovery` (9 imputer, Traccia A) e `WAPE_forecasting_per_serie` (Traccia B). Distribuzione dei ~50K ρ_i sintetizzata con Cliff δ vs 0 + CI bootstrap 95% + categoria Romano.

| Forecaster | Famiglia | median ρ | Cliff δ vs 0 | CI 95% | Categoria Romano | Predice? |
|---|---|:---:|:---:|:---:|---|---|
| **MA_K21** ★ | naive | +0.800 | **+0.868** | [+0.864, +0.872] | **LARGE** | SÌ (massimo) |
| **DoW Mean** | naive | +0.867 | **+0.823** | [+0.817, +0.828] | **LARGE** | SÌ |
| **Global Mean** | naive | +0.883 | **+0.813** | [+0.808, +0.818] | **LARGE** | SÌ |
| **TimesFM** | foundation | +0.333 | **+0.556** | [+0.549, +0.563] | **LARGE** | SÌ |
| **Chronos-bolt** | foundation | +0.367 | **+0.472** | [+0.464, +0.479] | **medium** | SÌ |
| MLP_M5 | ML+lag | +0.100 | +0.183 | [+0.174, +0.191] | small | debole |
| LGB_M5 | ML+lag | +0.100 | +0.142 | [+0.134, +0.151] | borderline | quasi no |
| TFT | DL+lag | −0.143 | −0.239 | [−0.247, −0.230] | small (inverso) | no, segno opposto |

**Finding 2**: la recovery predice il forecasting in modo **crescente** all'aumentare della dipendenza diretta dai dati imputati:
- **Naive** (LARGE δ ≈ +0.82–0.87): la predizione è quasi una funzione lineare dei valori imputati → recovery → forecasting in modo quasi deterministico.
- **Foundation models** (medium–LARGE δ ≈ +0.47–0.56): ricevono il segnale imputato ma lo trasformano con il modello pre-addestrato → effetto attenuato ma reale.
- **ML/DL con lag features M5** (negligible–small): i lag agiscono come buffer di compensazione → decoupling dalla qualità dell'imputer. `TimesNet` (worst recovery 1.04) genera la migliore cell forecasting per MLP_M5 (0.973) → conferma diretta del decoupling.

Questa sezione fornisce la **chiave esplicativa** del Finding 1.3: l'imputer è irrelevante per MLP_M5/LGB_M5 perché i lag M5 isolano il forecasting dalla qualità dell'imputation.

---

## Sezione 3 — Foundation models per retail

- **Chronos-bolt** × no_imp: WAPE_h_med = 1.007 → competitivo, Pareto solo a Q2 con `forward_fill`.
- **TimesFM 2.5-200M** × 14 imputer: WAPE_h_med best = 1.191 (imputeformer), peggio di Chronos del 18% (CPU only, 5x più lento). I 4 imputer aggiunti tardivamente (lgb, mediana_cond, media_cond, media_glob) producono WAPE_med 1.24–1.27 → vanno tutti al fondo del ranking interno di TimesFM, abbassando Kendall W (k=10: 0.229 → k=14: 0.174).

**Finding 3**: entrambi i foundation models sono **recovery-sensitive** (vedere Sezione 2 — Cliff δ vs 0: TimesFM +0.556 LARGE, Chronos +0.472 medium). Best imputer coerente per entrambi: **imputeformer**. Nonostante la recovery-sensitivity, **i foundation rimangono dominati da MLP_M5** sulla matrice principale (mean rank ≈ 35-40 vs 22.3 di `itransformer__MLP_M5`). Utili come baseline zero-shot ma non competitivi su retail deperibile con lag features disponibili.

## Sintesi findings per il paper (riorganizzato per sezione)

**Sezione 1 — Best cell + equivalence set (Friedman + W + CD)**
1. **Best globale**: `itransformer__MLP_M5` (mean rank 22.27 su 109 celle), equivalence set di 2 cells entrambe MLP_M5. Kendall W = 0.430 moderate.
2. **Per regime di volume**: crossover soft — la famiglia MLP_M5 vince in tutti i quartili, l'imputer ottimale cambia (lgb in Q1/Q2 basso volume, itransformer in Q3/Q4 medio-alto/alto). Q4 più discriminante (2 equiv), Q1 più saturato (12 equiv).
3. **Per forecaster**: dicotomia chiara. MLP_M5/LGB_M5 hanno W ≈ 0 (negligible) → imputer praticamente irrilevante. Foundation, TFT, naive hanno W ≥ 0.22 (small-moderate) → imputer aiuta, best coerente (imputeformer / dlinear / mediana_glob). **Sensitivity loss MAE vs MSE**: il claim "imputer doesn't matter" è stabile sotto entrambe le loss (Sez. 1.3.1).

**Sezione 2 — Recovery quality predice forecasting?**
4. La recovery predice il forecasting in modo **crescente** con la dipendenza diretta dai dati imputati: naive δ ≈ +0.85 LARGE, foundation δ ≈ +0.5 medium-LARGE, ML+lag δ ≈ +0.15 small. I lag M5 sono buffer che disaccoppia ML/DL dalla qualità imputer.

**Sezione 3 — Foundation models**
5. Chronos e TimesFM sono **recovery-sensitive** ma **dominati da MLP_M5** sulla matrice. Best imputer per entrambi: imputeformer. Utili come baseline zero-shot, non competitivi su retail con lag disponibili.

→ **Messaggio scientifico chiave**: "*la famiglia MLP_M5 domina il benchmark in ogni regime; dentro MLP_M5 la scelta dell'imputer è praticamente irrilevante (W ≈ 0) perché i lag features M5 disaccoppiano il forecasting dalla recovery quality. L'imputer conta solo per i forecaster senza lag (foundation models e naive aggregati), dove la recovery predice direttamente il forecasting (Cliff δ vs 0 ≥ +0.47 medium-LARGE).*"

## Framework statistico del paper (riferimento unico)

| Sezione | Domanda | Test | Effect size | Script |
|---|---|---|---|---|
| Sez. 1.1, 1.2, 1.3 | Best cell + equiv set (k > 2 metodi) | Friedman χ² | **Kendall's W** + Nemenyi CD | 45, 46, 48, 49 |
| Sez. 1.3.1 | Robustness loss MAE vs MSE (pairwise A vs B) | Wilcoxon paired | Cliff δ | 38 |
| Sez. 2 | Recovery vs forecasting (correlazione) | Wilcoxon vs 0 (su ρ_i) | **Cliff δ vs 0** + CI bootstrap (cat. Romano) | 42, 42b |
| Sez. 3 | Foundation models | come Sez. 1 + 2 | come Sez. 1 + 2 | 45, 42b |

**Niente TOST. Niente soglia Cliff δ < 0.147 come decision rule per "equivalence" tra k metodi.**
Script TOST/threshold-based (43, 44, 47) restano come **supplementary** ma non sono citati nel paper.

## Riferimento bibliografico chiave

- **Liu et al. (2025)** — FreshRetailNet-50K: Latent Demand from 50,000 Stores for World-scale Stockout Prediction in Fresh Retail. arXiv:2505.16319.
- **Du et al. (2023)** — SAITS: Self-Attention-based Imputation for Time Series. NeurIPS.
- **Zeng et al. (2022)** — DLinear: Are Transformers Effective for Time Series Forecasting? AAAI 2023.
- **Ansari et al. (2024)** — Chronos: Learning the Language of Time Series. arXiv:2403.07815.
- **Du et al. (2023)** — PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series. arXiv:2305.18811.
