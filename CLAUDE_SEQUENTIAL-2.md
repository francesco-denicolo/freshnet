# PINN-Retail Sequential: Physics-Informed Transformer per Demand Recovery e Forecasting Unificati

## Obiettivo del paper

Dimostrare che incorporare i vincoli fisici dell'inventario (conservazione, boundary conditions) nella loss di un Transformer bidirezionale produce un modello unico che fa **simultaneamente** demand recovery e forecasting, superando la pipeline two-stage (imputation separata + forecasting separato).

**Venue target:** NeurIPS / ICLR / Journal of Computational Physics.

**Claim centrale:** esistono due modi per fornire supervisione dove i dati sono censurati — il mascheramento statistico (stato dell'arte: TimesNet, SAITS) e i vincoli fisici (nostra proposta). I vincoli fisici sono superiori perché sfruttano informazione strutturale (bilancio dell'inventario) che nessuna tecnica di mascheramento può catturare. Inoltre, i vincoli permettono di unificare recovery e forecasting in un unico modello end-to-end, eliminando la pipeline two-stage.

**Il paper FreshRetailNet-50K indica esplicitamente questa direzione:** "the study focuses on a two-stage demand recovery-forecasting framework and does not explicitly evaluate end-to-end models, which could limit joint optimization of bias correction and prediction accuracy." Noi rispondiamo esattamente a questa call.

---

## Dataset: FreshRetailNet-50K

- **Fonte**: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
- **Paper baseline**: arxiv 2505.16319, codice: https://github.com/Dingdong-Inc/frn-50k-baseline
- **Dimensione**: 4.500.000 righe (train) + 350.000 (eval)
- **Periodo**: Marzo–Giugno 2024, 90 giorni
- **Granularità**: Oraria (vendite e stock status)
- **Copertura**: 898 negozi, 18 città cinesi, 863 SKU deperibili
- **Stockout rate**: ~25% delle ore sono in stockout

### Colonne del dataset

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| city_id | int | ID città |
| store_id | int | ID negozio |
| management_group_id | int | Gruppo gestione |
| first/second/third_category_id | int | Categorie prodotto (3 livelli) |
| product_id | int | ID prodotto |
| dt | str | Data giornaliera |
| sale_amount | float | Vendite giornaliere totali |
| hours_sale | array[24] | Vendite per ora (0-23) |
| stock_hour6_22_cnt | int | Ore di stockout tra 6:00 e 22:00 |
| hours_stock_status | array[24] | Stato stock binario per ora (0=in stock, 1=stockout) |
| discount | float | Sconto promozionale |
| holiday_flag | int | Flag festività |
| activity_flag | int | Flag attività promozionale |
| precpt | float | Precipitazioni |
| avg_temperature | float | Temperatura media |
| avg_humidity | float | Umidità media |
| avg_wind_level | float | Livello vento medio |

### Cosa NON c'è nel dataset

- **NON c'è il livello di inventario continuo I(t)**. Solo lo stato binario.
- **NON ci sono i rifornimenti R(t)**.
- **NON c'è lo scarto per deterioramento W(t)**.

---

## Train / Validation / Test Split

### Struttura temporale del dataset

```
Train HuggingFace: 4.500.000 righe = 50.000 serie × 90 giorni
Eval HuggingFace:    350.000 righe = 50.000 serie ×  7 giorni (subito dopo il train)
Totale: 97 giorni di dati (90 nel train HF + 7 nell'eval HF)
```

IMPORTANTE: il dataset HuggingFace contiene 90 giorni nel file train e 7 giorni nel file eval.
L'eval è temporalmente consecutivo al train (covariate shift: temp +7°C, pioggia +59%).
Noi suddividiamo i 90 giorni del train HF in train interno + validation interna.
L'eval HF (7 giorni) è il nostro test set finale e non viene mai toccato durante il training.

### Split — tutto consecutivo, nessun buco

```
|------------ Train HuggingFace: 90 giorni ------------|-- Eval HF: 7 gg --|
|--- Il nostro Train (gg 1-83) ---|--- Val (gg 84-90) ---|--- Test (eval) ---|
         83 giorni                      7 giorni               7 giorni
```

**Train: giorni 1-83** del train HF (83 giorni)
Per allenare i modelli durante la fase di tuning degli iperparametri.

**Validation: giorni 84-90** del train HF (7 giorni)
Per early stopping, hyperparameter tuning, e selezione del miglior modello di imputation.

**Test: eval set HF** (7 giorni, subito dopo il giorno 90)
Per la valutazione finale (Traccia B: forecasting). Mai usato durante il training.

**Retraining: giorni 1-90** (tutti i 90 giorni del train HF)
Dopo aver scelto gli iperparametri migliori con train/val, si riallena il modello
finale su TUTTI i 90 giorni del train HF. Questo modello riallenato è quello
usato per tutte le valutazioni finali.

### Workflow completo per i nostri modelli

```
1. TUNING:     Train su gg 1-83, Val su gg 84-90 → scegli iperparametri migliori
2. RETRAINING: Riallena il modello migliore su gg 1-90 (tutti i 90 giorni del train HF)
3. TRACCIA A:  Valuta il modello riallenato su maschere MNAR test (gg 1-90)
4. TRACCIA B:  Valuta il modello riallenato su forecasting (eval set HF, 7 giorni)
```

### Split per i modelli del paper baseline (TimesNet, SAITS, etc.)

I modelli del paper si allenano su tutti i **90 giorni** del train HF con MNAR masking interno.
Stesso periodo del nostro retraining. Il confronto è perfettamente equo.

---

## Maschere MNAR predefinite (CRITICO)

### Due set di maschere indipendenti

Servono **due set di maschere MNAR** con scopi diversi:

**Set 1 — Maschere di VALIDATION (per la selezione del modello di imputation):**
- Applicate solo ai giorni 84-90 del train HF (il periodo di validation)
- Seed: 42
- Missing rate: 30% delle ore in-stock
- Usate per confrontare TUTTI i modelli di imputation (naive, ML, stato dell'arte)
  e scegliere il vincitore
- File: `data/mnar_masks_val.parquet`

**Set 2 — Maschere di TEST (per la valutazione finale, Traccia A):**
- Applicate ai giorni 1-90 (tutti i 90 giorni del train HF)
- Seed: 123 (diverso dal set 1!)
- Missing rate: 30% delle ore in-stock
- Usate per la valutazione finale di tutti i modelli (imputer vincitore e PINN)
  dopo il retraining su gg 1-90
- Tutti i modelli finali sono allenati su gg 1-90 e valutati su queste maschere
- File: `data/mnar_masks_test.parquet`

### Generazione maschere (identica per entrambi i set, cambia solo seed e periodo)

```python
def generate_mnar_masks(data, days_range, seed, missing_rate=0.3):
    """
    Genera maschere MNAR per le ore in-stock nel periodo specificato.
    Pattern MNAR: probabilità di mascheramento proporzionale alla distribuzione
    empirica degli stockout reali per ora del giorno.
    """
    rng = np.random.default_rng(seed=seed)
    
    # Calcola distribuzione empirica stockout per ora (dal train set)
    # p(mask|hour=h) ∝ tasso di stockout reale all'ora h
    # Es: ora 22 ha 42% stockout → alta probabilità di mascheramento
    #     ora 3 ha 6% stockout → bassa probabilità di mascheramento
    
    # Per ogni coppia (store, product) e ogni giorno nel periodo:
    # 1. Seleziona le ore in-stock (stock_status=0)
    # 2. Campiona missing_rate% secondo la distribuzione MNAR
    # 3. Marca come mascherate
    
    return masks  # DataFrame con (store_id, product_id, dt, hour, is_masked, ground_truth)
```

### Uso delle maschere

Ogni modello riceve i dati con le ore mascherate (S_obs=0, stock_status=1 per le ore mascherate).
Ogni modello viene valutato sulle stesse identiche ore (is_masked=True).
Nessuna ambiguità nel confronto.

---

## Selezione del miglior modello di imputation (Fase B, punto 3)

### Il principio

Tutti i modelli di imputation — naive, ML, e stato dell'arte — vengono valutati sulle
**stesse maschere MNAR di validation** (seed=42, gg 84-90), con la stessa metrica.
Il vincitore, qualunque famiglia sia, diventa l'imputer ufficiale che produce i
completed_sales per la Fase B punti (4) e (5).

Se un metodo semplice (media condizionata) batte TimesNet → si usa quello.
Se TimesNet è nettamente il migliore → si usa TimesNet.
La selezione è oggettiva e basata unicamente sulla WAPE_recovery.

### Il problema fondamentale

Non puoi valutare l'imputation sulle ore di stockout reale (il ground truth non esiste).
Usi le maschere MNAR di validation per simulare stockout su ore dove conosci il valore vero.

### I candidati

**Famiglia 1 — Imputation naive:**
- Media condizionata: per ogni (store, product, dow, hour), media di S_obs sulle ore in-stock.
  Nessun modello da allenare.
- Media globale: per ogni (store, product, hour), media su tutti i giorni in-stock.

**Famiglia 2 — Imputation ML:**
- LGB imputer: LightGBM allenato sulle ore in-stock.
  Features: store_id, product_id, ora, dow, discount, meteo.
  Target: S_obs. Applicato alle ore di stockout per stimare la domanda.

**Famiglia 3 — Imputation stato dell'arte (sequenziale, bidirezionale):**
- TimesNet, SAITS, iTransformer, DLinear
  Operano su sequenze con NaN (ore di stockout). Allenati con strategia ORT+MIT.
  Implementazione: PyPOTS dal repo Dingdong-Inc/frn-50k-baseline.

### Procedura di selezione (uguale per tutte le famiglie)

```
1. TRAIN:    Allena l'imputer su gg 1-83 (ore in-stock per naive/ML, ORT+MIT per sequenziali)
2. APPLY:    Applica alle ore mascherate dei gg 84-90 (maschere val, seed=42)
3. EVALUATE: Calcola WAPE_recovery e WPE_recovery sulle maschere val
4. COMPARE:  Confronta TUTTI i candidati (naive + ML + stato dell'arte)
5. SELECT:   Scegli il migliore per WAPE_recovery
6. RETRAIN:  Riallena il vincitore su gg 1-90 (tutti i 90 giorni)
7. PRODUCE:  Genera completed_sales: S_obs dove in-stock, D_hat dove stockout
8. FINAL:    Valuta il vincitore sulle maschere test (seed=123, gg 1-90) → numeri Traccia A
```

### Metrica di selezione

Selezione primaria: **WAPE_recovery** (coerente con il paper baseline).
Metrica secondaria riportata: **WPE_recovery** (indica il bias della ricostruzione).

### Output della selezione

- Tabella comparativa di tutti i candidati (WAPE_recovery, WPE_recovery su val)
- Il miglior modello selezionato (es. "TimesNet con n_layers=3, d_model=128")
- `data/completed_sales.parquet`: i dati imputati dal vincitore riallenato su 1-90
  Questi completed_sales vengono usati nella Fase B punti (4) e (5) per calcolare
  i lag features puliti e allenare i modelli di forecasting two-stage
- Numeri finali di recovery del vincitore sulle maschere test → riportati nel paper

---

## Formato dei campioni per il PINN Sequenziale

Ogni campione di training è una **finestra di T ore consecutive** per una coppia (store, product).

```
Esempio con T = 168 (7 giorni):
- Finestra 1: ore 1-168 (giorni 1-7)
- Finestra 2: ore 25-192 (giorni 2-8) [stride=24]
- ...
- Ultima finestra nel train: termina al giorno 83

Per validation: finestre che terminano nei giorni 84-90
Per test forecasting: finestre che terminano al giorno 90,
  con le ultime 24 ore come target (giorno 91 = primo giorno eval)
```

Le ultime FH=24 ore di ogni finestra sono l'orizzonte di forecasting.
Le prime T-24 ore sono il contesto storico dove operano i vincoli e la recovery.

---

## Contesto: risultati della Fase 1 (PINN MLP)

Nella fase precedente abbiamo implementato un PINN con architettura MLP (non sequenziale).
I risultati hanno rivelato:

| Modello | WAPE_in med | WPE_in med |
|---------|:-----------:|:----------:|
| **PINN MLP** | **1.040** | -0.474 |
| MLP single-stage (F) | 1.086 | -0.324 |
| Two-Stage MLP | 1.115 | -0.188 |
| Two-Stage LGB | 1.157 | -0.073 |

**Lezioni apprese:**
1. I vincoli fisici nella loss agiscono come **regolarizzatore potente** → miglior WAPE di tutti.
2. Ma il bias (WPE) è il **peggiore** → i vincoli NON risolvono il censoring negli input.
3. Causa: il PINN MLP riceve lag features precalcolate da S_obs (contaminati dagli zeri da stockout). I vincoli operano solo sulle 24 ore di output, non possono correggere input già contaminati.
4. Il two-stage riduce il bias (decontamina i lag) ma peggiora il WAPE del ~2.7% (rumore dell'imputation).
5. **Conclusione:** serve un'architettura che processi la sequenza completa e ricostruisca D*(t) anche nelle ore storiche. I vincoli devono operare su tutta la sequenza, non solo sull'output.

Questa è la motivazione per il PINN Sequenziale.

---

## Architettura: PINN Sequenziale

### Differenza fondamentale con il PINN MLP

Il PINN MLP riceve le feature di un giorno (inclusi lag precalcolati) e produce D*(h) per le 24 ore future. I vincoli operano solo sulle 24 ore predette. Lo storico entra come lag features già contaminati.

Il PINN Sequenziale riceve l'**intera sequenza** di T ore (storico + orizzonte futuro) e produce D*(t) per **ogni** timestep. I vincoli operano su tutta la sequenza. Non ci sono lag features precalcolate: la rete vede direttamente S_obs(t) e ricostruisce D*(t) ovunque.

### Input per ogni timestep t

Per ogni ora t nella sequenza di T timestep:
- **S_obs(t):** vendite osservate (inclusi zeri da stockout) — il segnale grezzo
- **Covariate temporali:** ora del giorno (0-23), giorno della settimana (0-6), giorno del mese
- **Covariate esogene:** discount, holiday_flag, activity_flag, temperatura, umidità, precipitazioni, vento
- **Embedding categoriali:** city_id, store_id, product_id, category L1/L2/L3

**stock_status NON è un input.** Motivazioni (confermate dalla Fase 1):
1. A inference le ore future non hanno stock_status → mismatch train/inference.
2. Anche col Transformer bidirezionale, il rischio di collasso (status=1 → D*≈0) esiste.
3. stock_status è usato SOLO nella loss: L_data esclude le ore di stockout,
   L_boundary forza I*≈0 durante gli stockout. La rete non ha bisogno di "sapere"
   quali ore sono censurate — il Transformer impara a riconoscere gli zeri anomali
   dal contesto (ore adiacenti con vendite normali).

L'input per un campione è una matrice T × d_features.

### Architettura

```
Input: sequenza di T ore, ciascuna con d_features dimensioni
  ↓
Proiezione lineare: d_features → d_model
  ↓
Positional Encoding (sinusoidale o apprendibile)
  ↓
Transformer Encoder (L layer, n_heads teste, d_ff feed-forward)
  — Self-attention BIDIREZIONALE: ogni ora vede tutte le altre
  — Produce h(t) ∈ R^d_model per ogni timestep
  ↓
Due teste MLP separate:
  Head D: h(t) → Linear → ReLU → Linear → Softplus → D*(t)  (domanda latente, > 0)
  Head I: h(t) → Linear → ReLU → Linear → Softplus → I*(t)  (inventario latente, ≥ 0)
  ↓
Output: D*(t) e I*(t) per ogni t = 1, ..., T
```

**Iperparametri da esplorare:**
- T (lunghezza finestra): 72h (3gg), 168h (7gg), 336h (14gg)
- d_model: 64, 128, 256
- L (layer Transformer): 2, 3, 4, 6
- n_heads: 4, 8
- d_ff: 2*d_model, 4*d_model
- Dropout: 0, 0.1, 0.2
- Learning rate: 1e-4, 3e-4, 1e-3

### Perché il Transformer bidirezionale è cruciale

Per uno stockout alle 15:00, il Transformer vede:
- Le vendite alle 14:00 (30 unità) — contesto precedente
- Le vendite alle 16:00 (25 unità, se il prodotto è tornato) — contesto successivo
- Lo stock_status alle 15:00 = 1 — "questo zero è censurato"
- Il vincolo di conservazione — "l'inventario è sceso a zero qui"

Tutte queste informazioni convergono per ricostruire D*(15:00) ≈ 27.
Un MLP non può fare questo perché non vede il contesto temporale.

---

## Loss function — 3 termini su TUTTA la sequenza

```
L_total = L_data + λ₁·L_cons + λ₂·L_boundary
```

Ottimizzata con ALM: L_ALM = L_data + Σ_k [λ_k · V_k + (ρ_k/2) · V_k²]

**CRITICO: la differenza con il PINN MLP è che ogni termine opera su TUTTA la sequenza (T ore), non solo sulle 24 ore di output.**

### Termine 1 — L_data (aderenza ai dati, SOLO ore in-stock, TUTTA la sequenza)

```
L_data = (1/|T_in|) · Σ_{t: status(t)=0} (D*(t) − S_obs(t))²
dove T_in = {t ∈ [1..T] : stock_status(t) = 0}
```

Nessun downweighting: le ore di stockout sono escluse. Nelle ore in-stock,
S_obs = D (domanda vera), quindi D* deve coincidere con S_obs.
Questo termine fornisce supervisione abbondante (~75% delle ore).

### Termine 2 — L_boundary (condizioni al contorno, TUTTA la sequenza)

```
L_boundary = (1/|T_so|) · Σ_{t: status(t)=1} I*(t)²
           + (1/|T_in|) · Σ_{t: status(t)=0} ReLU(D*(t) − I*(t))²
```

Due sotto-vincoli:
- **Stockout → I* ≈ 0**: quando stock_status=1, l'inventario deve essere esaurito.
- **In-stock → I* ≥ D***: quando stock_status=0, l'inventario è sufficiente a soddisfare la domanda.

Operando su tutta la sequenza, questo vincolo forza la traiettoria I*(t) ad essere
coerente con gli stockout osservati sia nelle ore storiche sia nelle ore future.

### Termine 3 — L_cons (conservazione dell'inventario, TUTTA la sequenza)

```
L_cons = (1/(T−1)) · Σ_{t=1}^{T-1} ReLU(−[I*(t+1) − I*(t) + min(D*(t), I*(t))])²
```

Fisica: I(t+1) = I(t) − min(D(t), I(t)) + R(t), con R(t) ≥ 0.
Il termine penalizza solo quando R(t) implicito < 0 (fisicamente impossibile).
Disuguaglianza (non uguaglianza) perché il rifornimento R(t) avviene ma non è osservato.

**MECCANISMO DI DE-CENSORING:** Questo vincolo, operando sulle ore STORICHE,
è ciò che ricostruisce la domanda durante gli stockout passati.
Se I* scende da 40 a 0 tra le 12:00 e le 14:00, la conservazione forza
D*(12:00-14:00) ≈ 40 unità totali. Il Transformer poi propaga questa
informazione nelle ore di stockout adiacenti via attention.

### Ottimizzazione ALM

```
L_ALM = L_data + λ₁·V_cons + (ρ₁/2)·V_cons² + λ₂·V_boundary + (ρ₂/2)·V_boundary²
```

Training a 3 fasi per ogni iterazione ALM:
1. Warmup: alcune epoche con solo L_data (la rete impara i pattern base)
2. Passo primale: aggiorna Θ con Adam minimizzando L_ALM
3. Passo duale: λ_k ← max(0, λ_k + ρ_k · V_k)
4. Adattamento: se V_k non migliora, ρ_k ← γ · ρ_k (γ ≈ 1.5-2.0)

---

## Modelli da implementare

### Modello D — Transformer Vanilla (lower bound)

Stessa architettura del PINN Sequenziale. Loss: solo L_data.
Nessun vincolo fisico, nessun mascheramento.
Le D*(t) durante gli stockout non hanno supervisione.
Serve come ablation: isola il contributo dell'architettura Transformer da quello dei vincoli.

### Modello A — Transformer Masked (proxy per TimesNet)

Stessa architettura del PINN Sequenziale.
Durante il training: maschera il 30% delle ore in-stock (pattern MNAR),
azzera le vendite, allena il modello a ricostruirle.

Loss:
```
L = L_data (ore in-stock NON mascherate) + L_reconstruction (ore mascherate vs ground truth)
L_reconstruction = (1/|T_mask|) · Σ_{t ∈ T_mask} (D*(t) − S_true(t))²
```

Nessun vincolo fisico. La supervisione durante gli stockout viene dal mascheramento.
Il confronto con il PINN isola PERFETTAMENTE il contributo dei vincoli fisici
perché l'unica differenza è la loss.

### Modello B — PINN Sequenziale (contributo principale)

Come descritto nella sezione "Architettura" sopra.
Loss: L_data + λ₁·L_cons + λ₂·L_boundary, ottimizzata con ALM.
La supervisione durante gli stockout viene dai vincoli fisici.
Nessun mascheramento artificiale.

### Modello C — PINN Sequenziale + Masked (il più potente)

Combina entrambi i meccanismi di supervisione.

Loss:
```
L = L_data (ore in-stock non mascherate)
  + L_reconstruction (ore mascherate vs ground truth)
  + λ₁·L_cons + λ₂·L_boundary
```

Supervisione statistica (maschere) E fisica (vincoli) insieme.

---

## Piano sperimentale

### Struttura a tre fasi

```
Fase A — Nessuna imputation (baseline)
  (1) Naive su dati sporchi (train gg 1-90, test eval HF)
  (2) ML/DL su dati sporchi (train gg 1-83, val gg 84-90, retrain gg 1-90, test eval HF)

Fase B — Two-stage: imputation + forecasting (stato dell'arte)
  (3) Imputation con naive, ML, TimesNet, SAITS, iTransformer, DLinear
      → selezione miglior imputer su maschere MNAR val (gg 84-90, seed=42)
      → retrain miglior imputer su gg 1-90
      → valutazione finale su maschere MNAR test (gg 1-90, seed=123)
      → produzione completed_sales per i punti (4) e (5)
  (4) Naive su dati puliti (completed_sales, train gg 1-90, test eval HF)
  (5) ML/DL su dati puliti (train gg 1-83, val gg 84-90, retrain gg 1-90, test eval HF)

Fase C — PINN end-to-end su dati sporchi (nostro contributo principale)
  (6) PINN Sequenziale con input sporco
      → selezione iperparametri su train gg 1-83, val gg 84-90
      → retrain su gg 1-90
      → valutazione recovery su maschere MNAR test (stesse della Fase B)
      → valutazione forecasting su test set (eval HF)
```

### Confronti chiave e cosa dimostrano

- **Fase A vs Fase B:** il censoring è un problema reale. L'imputation migliora il WPE.
- **Fase B punto (3) vs Fase C recovery:** PINN vs TimesNet/SAITS sulla recovery pura.
  Se il PINN batte gli imputer dedicati → vincoli fisici > mascheramento statistico.
- **Fase B punto (5) vs Fase C forecasting:** two-stage vs end-to-end.
  Se il PINN con input sporco batte ML/DL con input pulito → pipeline two-stage superflua.
- **Fase A punto (2) vs Fase C forecasting:** stesso input (sporco), modello diverso.
  Isola il contributo dei vincoli fisici come regolarizzatore.

### Esperimento 1 — Traccia A: Demand Recovery (simulazione MNAR)

**Obiettivo:** Valutare la qualità della ricostruzione della domanda latente nelle ore censurate.

**Setup (valutazione finale, dopo retraining su 1-90):**
1. Tutti i modelli sono allenati su giorni 1-90.
2. Le maschere MNAR test (seed=123) sono applicate ai giorni 1-90.
3. Per le ore mascherate: S_obs(t) ← 0, stock_status(t) ← 1.
4. Ogni modello produce D_hat(t) per le ore mascherate.
5. Confronto con ground_truth(t).

**Cosa riceve ogni modello:**
- La sequenza con stockout reali + artificiali.
- Il modello produce D*(t) per ogni timestep.
- Si raccolgono le D*(t) solo per le ore mascherate artificialmente.

**Metriche:**
- WAPE_recovery = Σ |D_hat(t) − ground_truth(t)| / Σ ground_truth(t)
- WPE_recovery = Σ (D_hat(t) − ground_truth(t)) / Σ ground_truth(t)
- Decoupling Score ρ_DS = correlazione pesata Pearson tra tasso stockout
  e domanda ricostruita per coppia store-product

**Confronti e cosa dimostrano:**
- D (Vanilla) vs A (Masked): valore del mascheramento come supervisione
- D (Vanilla) vs B (PINN): valore dei vincoli fisici come supervisione
- **A (Masked) vs B (PINN): confronto centrale — mascheramento vs vincoli**
- B (PINN) vs C (PINN+Masked): complementarità dei due meccanismi
- B (PINN) vs TimesNet (paper): posizionamento vs stato dell'arte

**Modelli del paper baseline:** I numeri di TimesNet (WAPE 27.62%, WPE 1.43%),
iTransformer, SAITS, DLinear si possono prendere dal paper oppure rieseguire
con il codice PyPOTS dal repo Dingdong-Inc/frn-50k-baseline:
```bash
cd latent_demand_recovery/exp
python app.py --model TimesNet --missing_rate 0.3
```

### Esperimento 2 — Traccia B: Demand Forecasting

**Obiettivo:** Valutare la qualità delle previsioni per le 24 ore future.

**Setup:**
- Test set: dataset eval (7 giorni).
- Rolling evaluation a 1 giorno (come nella fase precedente).
- I modelli sequenziali producono D*(t) per tutta la sequenza;
  le ultime 24 ore sono le previsioni di forecasting.

**Metriche:**
- WAPE_instock e WPE_instock (solo ore in-stock del test set)
- Sia pooled sia mediana per-serie

**Confronti e cosa dimostrano:**
- B (PINN Seq) vs D (Vanilla): vincoli migliorano il forecasting?
- B (PINN Seq) vs A (Masked): PINN end-to-end vs imputation integrata?
- B (PINN Seq) vs TimesNet→TFT (two-stage paper): end-to-end vs pipeline?
- B (PINN Seq) vs PINN MLP (fase 1): la sequenzialità risolve il bias?
  PINN MLP: WAPE 1.040 (ottimo) ma WPE -0.474 (pessimo).
  PINN Seq dovrebbe avere WPE molto migliore grazie alla recovery integrata.
- B (PINN Seq) vs Two-Stage MLP (fase 1): l'end-to-end batte il two-stage?

### Esperimento 3 — Traccia C: Consistenza Fisica (solo PINN)

**Obiettivo:** Mostrare che il PINN produce previsioni fisicamente plausibili e informazione diagnostica.

**Metriche:**
- Violazione media vincolo conservazione: media |r_cons(t)| su test set
- Violazione media vincolo boundary: media I*(t) nelle ore stockout
- Shadow prices λ₁* (conservazione) e λ₂* (boundary) a convergenza
- Distribuzione shadow prices per categoria, fascia oraria, negozio

**Visualizzazioni qualitative (3-4 serie esemplari):**
- S_obs(t) (vendite osservate con zeri da stockout)
- D*(t) del PINN (domanda ricostruita)
- D*(t) del Transformer Masked (per confronto)
- I*(t) del PINN (traiettoria inventario latente)
- stock_status(t) (annotazioni stockout)

### Esperimento 4 — Ablation Studies

**4a — Contributo di ogni vincolo:**

| Variante | L_data | L_cons | L_boundary | WAPE_in | WPE_in | WAPE_rec |
|----------|:------:|:------:|:----------:|---------|--------|----------|
| Solo data | ✓ | | | ? | ? | ? |
| + conservazione | ✓ | ✓ | | ? | ? | ? |
| + boundary | ✓ | | ✓ | ? | ? | ? |
| PINN completo | ✓ | ✓ | ✓ | ? | ? | ? |

**4b — Tipo di encoder:**

| Encoder | Parametri | WAPE_in | WPE_in | WAPE_rec |
|---------|-----------|---------|--------|----------|
| MLP (fase 1) | ~113K | 1.040 | -0.474 | N/A |
| LSTM bidirezionale | ? | ? | ? | ? |
| Transformer encoder | ? | ? | ? | ? |

**4c — Lunghezza della finestra:**

| T (ore) | Giorni | WAPE_in | WAPE_rec | Tempo training |
|---------|--------|---------|----------|----------------|
| 72 | 3 | ? | ? | ? |
| 168 | 7 | ? | ? | ? |
| 336 | 14 | ? | ? | ? |

### Esperimento 5 — Confronto Aggregato con Paper Baseline

Aggrega previsioni orarie → giornaliere. Calcola WAPE e WPE giornalieri
solo sui giorni senza stockout completo.

| Modello | WAPE_daily | WPE_daily |
|---------|-----------|-----------|
| TFT raw (paper) | 31.75% | -7.37% |
| TimesNet → TFT (paper) | 29.02% | 2.58% |
| PINN Sequenziale | ? | ? |

---

## Ordine di esecuzione

### Fase 0 — Maschere MNAR + modelli del paper baseline (Settimana 1)

**Obiettivo:** generare le maschere predefinite, ottenere i numeri di riferimento
per la Traccia A PRIMA di implementare il PINN Sequenziale.

**Step 0.1 — Genera le maschere MNAR predefinite:**
```bash
# Notebook: notebooks/15_generate_mnar_masks.py
# Output 1: data/mnar_masks_val.parquet  (giorni 84-90, seed=42, per selezione modelli)
# Output 2: data/mnar_masks_test.parquet (giorni 1-90, seed=123, per valutazione finale)
# Missing rate: 0.3, pattern: MNAR empirico
```
Questi file vengono usati da TUTTI i modelli. Non rigenerarli mai.

**Step 0.2 — Clona e studia il repo baseline:**
1. Clona il repo: `git clone https://github.com/Dingdong-Inc/frn-50k-baseline`
2. Crea ambiente: `conda create --name py3.8_frn python=3.8 && conda activate py3.8_frn`
3. Installa dipendenze: `pip install -r ./requirements.txt`
4. **CRITICO: studia il codice** per capire:
   - Come generano le maschere MNAR (quale distribuzione, quale seed)
   - Come calcolano WAPE, WPE, Decoupling Score (formule esatte)
   - Come preprocessano i dati (normalizzazione, formato input per PyPOTS)
   - Come modificare il codice per usare le nostre maschere predefinite

**Step 0.3 — Esegui TUTTI i modelli di imputation candidati:**

Famiglia 1 (naive) e Famiglia 2 (ML):
```bash
# Notebook: notebooks/15b_imputation_naive_ml.py
# Media condizionata, Media globale, LGB imputer
# Train su gg 1-83 (ore in-stock), valutazione su maschere val (gg 84-90)
```

Famiglia 3 (stato dell'arte sequenziale):
```bash
cd latent_demand_recovery/exp
# Modifica app.py per: train=gg 1-83, maschere val=mnar_masks_val.parquet
python app.py --model TimesNet --missing_rate 0.3
python app.py --model SAITS --missing_rate 0.3
python app.py --model iTransformer --missing_rate 0.3
python app.py --model DLinear --missing_rate 0.3
```

Confronta TUTTI i candidati sulle maschere val → seleziona il vincitore per WAPE_recovery.

Retrain del vincitore su gg 1-90, valutazione finale su maschere test:
```bash
# Retrain vincitore su tutti i 90 giorni, maschere test=mnar_masks_test.parquet
python app.py --model <best_model>  # con configurazione retraining
```

**Step 0.4 — Salva tutti i risultati:**
- Tabella comparativa TUTTI i candidati (naive + ML + stato dell'arte) su maschere val
- Il vincitore selezionato e i suoi numeri finali su maschere test
- `data/completed_sales.parquet`: S_obs dove in-stock, D_hat dove stockout
  (dal vincitore riallenato su gg 1-90)

**Output della Fase 0:**
- `data/mnar_masks_val.parquet` e `data/mnar_masks_test.parquet`
- Tabella risultati recovery (val e test) per TUTTI i candidati di imputation
- `data/completed_sales.parquet` dal vincitore (per Fase B punti 4-5)
- Comprensione del protocollo di valutazione esatto del paper

### Fase 1 — PINN Sequenziale (Settimane 2-3)

Con i numeri di riferimento in mano:

1. Implementa l'architettura Transformer encoder con due teste (D*, I*)
   - Dataloader per finestre sequenziali di T ore
   - Embedding per store/product/city/category
   - Positional encoding
2. Implementa Modello D (Transformer Vanilla, solo L_data) — training e valutazione
   - Verifica che l'architettura funzioni e produca previsioni ragionevoli
   - Questo è il "micro-step 1" di debugging
3. Implementa Modello B (PINN Sequenziale):
   - Aggiungi L_cons e L_boundary alla loss
   - Prima con λ fissi (micro-step 2), poi con ALM (micro-step 3)
4. **TUNING:** allena D e B su giorni 1-83, valida su 84-90 (maschere val per recovery, metriche forecast)
5. **RETRAINING:** riallena D e B con gli iperparametri migliori su giorni 1-90
6. Valuta i modelli riallenati su Traccia A (maschere MNAR test, seed=123, giorni 1-90)
7. Valuta i modelli riallenati su Traccia B (forecasting, giorni 91-97)
8. Esperimento 5 (confronto aggregato giornaliero vs paper — poche righe)

**Criterio di successo:** B (PINN) batte D (Vanilla) significativamente su entrambe le tracce.

### Fase 2 — Rafforza il paper (Settimane 3-4)

7. Implementa Modello A (Transformer Masked) — stessa architettura, loss con mascheramento
8. Implementa Modello C (PINN + Masked) — entrambi i meccanismi
9. **TUNING** su 1-83/84-90, poi **RETRAINING** su 1-90 per A e C
10. Valuta A e C riallenati su Traccia A (maschere test, giorni 1-90) e Traccia B (giorni 91-97)
11. Esperimento 3 (consistenza fisica, visualizzazioni)

**Confronto centrale del paper:** A (Masked) vs B (PINN) sulla Traccia A, entrambi allenati su 1-90.

### Fase 3 — Completezza (Settimana 5)

11. Ablation studies (Esperimento 4: contributo di ogni vincolo, encoder, finestra)
12. Pulizia codice e documentazione
13. Scrittura paper

---

## Training del PINN Sequenziale — dettagli implementativi

### Costruzione campioni di training

Ogni campione è una finestra di T ore consecutive per una coppia (store, product).
Con stride S (es. S=24, un giorno), si generano (83*24 - T) / S campioni per serie in fase di tuning.
Con 50.000 serie → milioni di campioni totali. Campionamento random nel dataloader.
In fase di retraining si usano 90*24 ore (giorni 1-90).

Per ogni campione:
- Input: matrice T × d_features (S_obs, covariate, embedding — NO stock_status)
- Output del modello: D*(t) e I*(t) per ogni t = 1..T
- Le loss sono calcolate sulla finestra completa (stock_status usato solo nella loss)

### Forecast horizon

Le ultime FH = 24 ore della finestra sono il forecast horizon.
Per queste ore, S_obs NON è disponibile a inference (è ciò che prevedi).

**Due opzioni per il training:**
- **Teacher forcing:** durante il training, usa S_obs reale anche
  per le ultime 24 ore (li conosci perché sono dati storici). I vincoli operano
  su tutta la finestra T, incluse le ultime 24 ore.
- **Masking delle ultime 24h:** durante il training, maschera S_obs
  delle ultime 24 ore (simula l'inference). I vincoli operano solo sulle prime T-24 ore.

La prima opzione è più semplice e dà più supervisione. La seconda è più coerente
con l'inference. Testare entrambe come ablation.

Nota: poiché stock_status NON è un input della rete, non c'è problema di mismatch
per le ore future. L'unica feature non disponibile a inference è S_obs delle ore future.

### Memory e compute

Un Transformer encoder con d_model=128, L=3, T=168:
- Self-attention: O(T² · d_model) = O(168² · 128) ≈ 3.6M operazioni per layer
- Parametri encoder: ~400K
- Parametri totali (con embedding e teste): ~500K-1M
- Batch size: dipende dalla GPU. Con T=168, batch=64 richiede ~2GB GPU memory.
- Training: ~100-200 epoche, stima 2-4 ore su GPU moderna.

### Gestione embedding per modello globale

Con 898 negozi e 863 prodotti, le embedding tables sono grandi.
- store_id → embedding dim 32: 898 × 32 = ~29K parametri
- product_id → embedding dim 32: 863 × 32 = ~28K parametri
- city_id → embedding dim 8: 18 × 8 = 144 parametri
- dow → embedding dim 4: 7 × 4 = 28 parametri

Le embedding vengono replicate per ogni timestep della sequenza
(broadcast: il prodotto è lo stesso per tutta la finestra).

---

## Risultati della Fase 1 (riferimento)

### Baseline naive e ML (test, mediana per-serie in-stock)

| Modello | WAPE_in med | WPE_in med | WAPE_in pool |
|---------|:-----------:|:----------:|:------------:|
| **PINN MLP** | **1.0404** | -0.4743 | **0.8357** |
| MLP (var A) | 1.0815 | -0.4045 | 0.8686 |
| MLP (var F) | 1.0859 | -0.3235 | 0.8588 |
| 2-Stage MLP | 1.1146 | -0.1883 | 0.8721 |
| DoW Mean | 1.1176 | -0.2279 | 0.9291 |
| LGB (var A) | 1.1186 | -0.2365 | 0.9197 |
| Global Mean | 1.1243 | -0.2272 | 0.9319 |
| LGB (var F) | 1.1268 | -0.2013 | 0.8827 |
| MA (K=14) | 1.1341 | -0.1722 | 0.9072 |
| 2-Stage LGB | 1.1573 | -0.0727 | 0.8952 |
| Naive Direct | 1.2192 | -0.1520 | 1.0605 |

### PINN MLP dettagli

- Architettura: MLP [128, 64] + ReLU condiviso, due teste (D*, I*) con Softplus
- 113,916 parametri
- Loss: L_data (MSE in-stock only) + L_boundary + L_cons (disuguaglianza R≥0)
- ALM: warmup 3 epoche, max 15 iter × 3 epoche interne, early stop iter 8
- Shadow prices: λ_boundary=0.068, λ_conservation≈0
- V_conservation ≈ 0 dall'inizio (disuguaglianza R≥0 trivialmente soddisfatta)

---

## Scoperte dall'EDA

1. **Codifica stock_status**: 0=in stock, 1=stockout
2. **Tasso stockout**: 24.9% delle ore sono in stockout
3. **Vendite e stockout**: media vendite in-stock=0.054 (28.6% ore con vendite>0), stockout=0.004 (2.9%)
4. **Pattern orario stockout**: minimo ore notturne (~6%), massimo tardo pomeriggio-sera (42% ore 22)
5. **Vendite bimodali**: picchi ore 8-10 e 15-17. Weekend +25% vendite
6. **Esogene**: discount < 0.7 → vendite +59%, holiday → +27%, meteo effetto debole
7. **Eval set**: 7 giorni subito dopo train, covariate shift (temp +7°C, pioggia +59%)

---

## Decisioni confermate dalla Fase 1

1. **No deterioramento**: il dataset non ha dati di scarto.
2. **Due teste (D*, I*)**: domanda latente e inventario latente separate.
3. **Modello globale**: un unico modello su tutte le 50.000 serie.
4. **L_data solo in-stock**: ore di stockout escluse da L_data.
5. **L_cons come disuguaglianza**: permette R(t) ≥ 0.
6. **Non-negatività implicita via softplus**: non serve L_nonneg esplicito.

## Nuove decisioni per la Fase 2

7. **stock_status NON è un input** (confermato anche per il Sequenziale):
   a inference le ore future non hanno stock_status. Se fosse input per le ore storiche
   ma non per le future, ci sarebbe un mismatch train/inference.
   stock_status è usato SOLO nella loss (L_data esclude stockout, L_boundary usa status).
8. **Encoder Transformer bidirezionale (non MLP)**: necessario per il contesto e la recovery.
9. **Vincoli su tutta la sequenza**: non solo sulle 24 ore di output.
10. **Niente lag features precalcolate**: la rete vede direttamente la sequenza S_obs.
11. **Split consecutivo**: train 1-83, val 84-90, test 91-97, retraining 1-90.
12. **Due set maschere MNAR**: val (seed=42, giorni 84-90) e test (seed=123, giorni 1-90).

---

## Struttura progetto

```
pinn-retail/
├── CLAUDE.md                    ← questo file
├── data/
│   ├── frn50k_train.parquet
│   └── frn50k_eval.parquet
├── baseline_paper/              ← FASE 0: codice del paper baseline
│   ├── frn-50k-baseline/       ← clone del repo Dingdong-Inc
│   ├── results_recovery/       ← output di TimesNet, SAITS, etc.
│   ├── mnar_masks/             ← maschere MNAR salvate per riuso
│   └── notes_baseline.md       ← appunti su come funziona il codice
├── notebooks/
│   ├── 01_eda.py                ← EDA (fase 1 precedente)
│   ├── 04-12_*.py               ← baseline e PINN MLP (fase 1 precedente)
│   ├── 15_generate_mnar_masks.py ← FASE 0: genera maschere MNAR predefinite (val seed=42, test seed=123)
│   ├── 15b_imputation_naive_ml.py ← FASE 0: imputation con media condizionata e LGB
│   ├── 16_run_paper_baseline.py ← FASE 0: esegui TimesNet/SAITS/etc. con maschere predefinite
│   ├── 17_analyze_baseline.py   ← FASE 0: confronto tutti gli imputer, selezione vincitore
│   ├── 20_pinn_sequential.py    ← FASE 1: PINN Sequenziale (Modello B)
│   ├── 21_transformer_vanilla.py ← FASE 1: Transformer senza vincoli (Modello D)
│   ├── 22_transformer_masked.py  ← FASE 2: Transformer con mascheramento (Modello A)
│   ├── 23_pinn_masked.py         ← FASE 2: PINN + Masked (Modello C)
│   ├── 24_traccia_a_recovery.py  ← Valutazione MNAR tutti i modelli
│   ├── 25_traccia_b_forecasting.py ← Confronto forecasting tutti i modelli
│   ├── 26_traccia_c_physics.py   ← Consistenza fisica e visualizzazioni
│   ├── 27_ablation.py            ← Ablation studies
│   └── 28_daily_comparison.py    ← Confronto aggregato giornaliero vs paper
├── src/
│   ├── models/
│   │   ├── transformer_encoder.py ← Architettura Transformer condivisa
│   │   └── pinn_sequential.py     ← PINN loss + ALM
│   ├── data/
│   │   ├── sequence_dataset.py    ← Dataset per finestre sequenziali
│   │   └── mnar_masking.py        ← Generazione maschere MNAR
│   └── evaluation/
│       ├── metrics.py             ← WAPE, WPE (dalla fase 1)
│       └── decoupling_score.py    ← Decoupling Score ρ_DS
└── results/
    └── *.parquet                  ← Risultati per-serie per confronto
```

---

## Riferimenti chiave

- Raissi et al. (2019) — PINNs fondativi, J. Computational Physics
- Vaswani et al. (2017) — Transformer, "Attention Is All You Need"
- Shin et al. (2020) — Convergenza PINNs
- Bertsekas (1982) — ALM convergenza
- FreshRetailNet-50K paper (2025) — arxiv 2505.16319
- TimesNet — Wu et al. (2023), ICLR
- SAITS — Du et al. (2023), Expert Systems with Applications
- iTransformer — Liu et al. (2024), ICLR
- TFT — Lim et al. (2021), IJF
