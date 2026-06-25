# Paper Findings — Imputer × Forecaster Benchmark on FreshRetailNet-50K

> Documento di riferimento per la stesura del paper. Tutti i finding numerici,
> tabelle, e claim per ciascuna sezione. Aggiornare qui (non in `CLAUDE.md`).

**Ultimo update**: 2026-06-23
**Matrice finale**: 113 cells (TimesFM completato su 14 imputer, allineato agli altri forecaster).
**Framework statistico unico**: Friedman + Kendall's W + Nemenyi CD (Demšar 2006, JMLR).

**Coherence note (2026-06-23)**: il forecaster MA è ora `MA_K56` (K=56), riselezionato sotto criterio median per-serie val WAPE (`min_hours=34`) per coerenza con l'objective HPO Optuna usato per LGB_M5/MLP_M5/TFT. Il precedente K=21 era selezionato via pooled WAPE (deprecato). Il finding di vertice (best globale `itransformer__MLP_M5`) è invariato; le 4 celle MA sulla Pareto sono ora `{media_cond, media_glob, mediana_cond, mediana_glob, seasonal_naive} × MA_K56`.

---

## Research Questions e mapping con le sezioni

Ordinamento "logically sustainable": dalla domanda di existence (foundationale) attraverso mechanism, identification, conditions, fino a boundary (paradigma alternativo). Ogni RQ presuppone strettamente le precedenti.

| RQ | Tipo | Domanda | Sezione |
|---|---|---|---|
| **RQ1** | existence | Per ciascun forecaster, l'imputation migliora il forecasting rispetto a `no_imp`? | 1.3 |
| **RQ2** | mechanism | La qualità della recovery (Traccia A) predice la qualità del forecasting downstream (Traccia B)? | 2 |
| **RQ3** | identification | Qual è la coppia (imputer, forecaster) migliore sulla matrice intera? Quante celle sono indistinguibili dal best? | 1.1 + 1.1 (b) Pareto |
| **RQ4** | conditions | Il best cambia in funzione del regime di volume delle serie? | 1.2 (a/b/c) |
| **RQ5** | boundary | I foundation models pre-trained (Chronos, TimesFM) alterano queste conclusioni? | 3 |
| _legacy_ | sensitivity | MAE vs MSE training loss | 1.3.1 |

→ Sequenza: **existence → mechanism → identification → conditions → boundary**. Mechanism-first deductive: stabilisce prima il framework interpretativo (RQ2), poi lo applica all'identification (RQ3).

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

Matrice [49.939 serie × 113 celle]. Output principale del paper.

| Livello | k | N | **Kendall W** | CD | Best cell | # CD-equiv |
|---|:---:|:---:|:---:|:---:|---|:---:|
| **Globale** | 113 | 49,939 | **0.459 (moderate)** | 0.903 | **itransformer__MLP_M5** | **2** |

Equivalence set (2 cells):
1. `itransformer__mlp_m5lags` (mean rank 22.33, best)
2. `lgb__mlp_m5lags` (mean rank 22.78, Δ = 0.51 ≤ CD)

**Finding RQ3 (a)**: il best globale è `itransformer__MLP_M5`. Solo `lgb__MLP_M5` gli sta statisticamente alla pari. Entrambe le 2 celle sono **MLP_M5**: la famiglia di forecaster vincente è isolata. Il ranking è generalizzabile (W = 0.459 moderate).

#### 1.1 (b) Trade-off WAPE × |WPE|: Pareto frontier globale (script 35)

Pareto su `(WAPE_h_med, |WPE_h_med|)` — accuracy vs bias. Frontier = celle non-dominate.

**25 / 113 cells Pareto-optimal** (ricalcolato post-K56; era 26 pre-K56). I tre punti di riferimento:

| Ruolo | Cella | WAPE | \|WPE\| |
|---|---|:---:|:---:|
| Best WAPE (accuracy-extreme) | `timesnet__MLP_M5` | **0.973** | 0.886 |
| Knee point (trade-off bilanciato) | `mediana_glob__dow_mean` | 1.101 | **0.190** |
| Min \|WPE\| (bias-extreme) | `linear_interp__timesfm` 🆕 | 1.295 | **0.061** |

**Estremi della frontier**:
- Lato WAPE: `MLP_M5/TFT` con vari imputer (WAPE 0.97–1.00, |WPE| 0.77–0.89).
- Lato |WPE|: **naive aggregati** (Global/DoW/MA con vari imputer, |WPE| 0.06–0.19, WAPE 1.10–1.16).
- TimesFM entra solo in posizione min-|WPE| (con linear_interp).

**Finding RQ3 (b)**: il trade-off è **strutturale**. Il best-WAPE ha |WPE| sempre elevato (≥ 0.77) — i forecaster ML/DL **sotto-stimano sistematicamente**. Per ridurre il bias servono naive aggregati che pagano in WAPE. Il knee point `mediana_glob__dow_mean` rappresenta un compromesso ragionevole per practitioner che valuta sia accuracy che bias.

### 1.2 Per regime di volume — robustezza del best (script 46)

Stratificazione per quartile di volume (Q1-Q4, ~12.500 serie per Q).

| Q | Range vol | Friedman best | Kendall W | CD | # CD-equiv |
|---|---|---|:---:|:---:|:---:|
| Q1 (basso) | [11, 40] | **lgb__MLP_M5** | 0.663 (large) | 1.804 | 12 |
| Q2 | (40, 54] | **lgb__MLP_M5** | 0.592 (large) | 1.806 | 12 |
| Q3 (medio-alto) | (54, 86] | **itransformer__MLP_M5** | 0.420 (moderate) | 1.807 | 4 |
| Q4 (alto) | (86, 5326] | **itransformer__MLP_M5** | 0.396 (moderate) | 1.807 | 2 |

**Finding RQ4 (a)**: crossover **soft**. La famiglia di forecaster vincente (MLP_M5) è **invariata** in tutti i regimi; l'imputer ottimale cambia tra `lgb` (basso volume Q1/Q2) e `itransformer` (alto volume Q3/Q4). Il regime più discriminante è Q4 (solo 2 celle CD-equivalenti); Q1 è il più saturato (12 equivalenti).

**Nota su W vs CD-equiv set size**: W e #equiv non sono ridondanti. Q1 ha W large + molti equiv (separazione forte tra famiglie ma debole dentro MLP_M5); Q4 ha W moderate + pochi equiv (naive competono → meno accordo globale, ma MLP_M5 differenziato → top isolato).

#### 1.2 (b) Pareto frontier per quartile (script 36)

Stessa metrica (WAPE × |WPE|) applicata dentro ciascun quartile. La frontier **cambia composizione** col regime.

| Q | # Pareto | Top-WAPE | Min-\|WPE\| | Famiglie sulla frontier |
|---|:---:|---|---|---|
| Q1 (basso vol) | **27** / 113 | `mediana_cond__LGB_M5` (1.000) | `seasonal_naive__GlobalMean` (\|WPE\|=0.10) | LGB_M5, MLP_M5, TFT (dominante in coda low-bias) |
| Q2 | 21 | `mediana_cond__LGB_M5` (0.995) | `seasonal_naive__GlobalMean` (\|WPE\|=0.01) | LGB_M5, MLP_M5, TFT, Chronos al margine |
| Q3 (medio-alto) | 16 | `lgb__MLP_M5` (0.958) | `seasonal_naive__GlobalMean` (\|WPE\|=0.04) | MLP_M5, TFT in testa; **naive emergono al low-bias** |
| Q4 (alto vol) | **9** / 113 | `itransformer__MLP_M5` (**0.757**) | `linear_interp__MA_K56` (\|WPE\|=**0.03**) | MLP_M5 (top WAPE), naive (low bias) — gap minimo |

**Finding RQ4 (b)**: tre pattern emergono dalla Pareto stratificata:

1. **# Pareto-optimal decresce con il volume** (Q1: 27 → Q4: 9) → stesso pattern del CD-equiv set: alto volume = più discriminante.
2. **In Q1/Q2 (basso volume) la frontier è popolata da TFT** che occupa la coda low-bias (linear_interp__TFT, mediana_glob__TFT) → TFT è il forecaster trade-off in basso volume.
3. **In Q3/Q4 (alto volume) la frontier si specializza**: MLP_M5/TFT al top-WAPE, **naive aggregati (Global/DoW/MA) al low-bias**. In Q4 il knee diventa sub-optimale rispetto a Q1: gap WAPE↔|WPE| si comprime (0.76 vs 0.06 in Q4 — naive batte ML su |WPE| pagando solo +3% WAPE!).
4. **Chronos** appare solo nella Pareto di Q2 (forward_fill__Chronos-bolt) → posizione di nicchia. **TimesFM** entra solo nella Pareto globale (min |WPE| con linear_interp) e mai in quella per quartile.

#### 1.2 (c) Crossover line-plot — evoluzione delle famiglie per quartile (script 39)

Figure `fig_rq3_crossover_fixed_global.png` (cella best globale per famiglia, fissata) e `fig_rq3_crossover_perq.png` (cella best per famiglia in ciascun Q, può cambiare imputer). 8 famiglie di forecaster, 4 punti per famiglia (Q1→Q4).

**Tabella numerica (cella best globale per famiglia, WAPE_h_med per Q)**:

| Cell | Q1 | Q2 | Q3 | Q4 |
|---|:---:|:---:|:---:|:---:|
| `timesnet__MLP_M5` | 1.000 | 0.999 | 0.960 | **0.770** |
| `mediana_cond__LGB_M5` | 1.000 | 0.995 | 0.963 | **0.779** |
| `dlinear__TFT` | 1.000 | 1.000 | 0.960 | **0.774** |
| `imputeformer__Chronos-bolt` | 1.004 | 1.004 | 1.004 | 0.949 |
| `imputeformer__DoW Mean` | 1.230 | 1.207 | 1.046 | 0.777 |
| `imputeformer__Global Mean` | 1.245 | 1.221 | 1.044 | 0.775 |
| `imputeformer__MA_K56` | 1.252 | 1.222 | 1.045 | 0.772 |
| `imputeformer__TimesFM` | 1.316 | 1.308 | 1.154 | 0.884 |

**Finding RQ4 (c)**: tre fenomeni di crossover osservabili nel line-plot:

1. **Convergenza generalizzata in Q4** (alto volume): MLP_M5, LGB_M5, TFT e tutti e tre i naive aggregati (Global/DoW/MA) convergono a WAPE ≈ 0.77–0.78. In basso volume questi metodi erano molto distanti (ML ≈ 1.00 vs naive ≈ 1.24), in alto volume diventano indistinguibili. **L'alto volume comprime le differenze tra famiglie**.

2. **Chronos-bolt è piatto, "perde" relativamente** (linea ~1.00 invariata in Q1-Q3, scende solo a 0.95 in Q4): a parità di volume gli altri imitano la sua performance in Q1 ma poi scalano molto meglio. Chronos non sfrutta il volume → in Q1 è competitivo con i top, in Q4 è il peggiore tra i forecaster con lag (eccetto TimesFM).

3. **Naive vs ML: il crossover più drammatico**. I naive aggregati partono a WAPE ≈ 1.22–1.26 in Q1 (WAPE 22-26% peggiore dei ML) e arrivano a ≈ 0.78 in Q4 (allineati ai ML). **Il gap naive↔ML scende da +25% a 0% al crescere del volume**.

**TimesFM** resta il **peggiore in tutti i quartili** (WAPE 1.32 → 0.88), con un crossover **interno alla foundation family**: in Q1 TimesFM è 31 punti % peggiore di Chronos, in Q4 il gap si riduce a −6 punti %. Foundation models scalano meglio col volume ma TimesFM parte da un livello superiore.

**Decision tree practitioner basato sui crossover**:
- **Basso volume (Q1-Q2)**: scegliere ML con lag features M5 (MLP_M5/LGB_M5/TFT) — i naive sono nettamente peggiori. Foundation models (Chronos) accettabili come baseline zero-shot.
- **Alto volume (Q4)**: la famiglia di forecaster diventa secondaria — anche i naive aggregati sono competitivi (con +3% WAPE pagano un trade-off bias drammaticamente migliore: |WPE| 0.06 vs 0.41 dei ML).
- **Chronos/TimesFM**: utili solo come baseline zero-shot. Nessun regime in cui dominano.

### 1.3 Per forecaster — l'imputation aiuta vs no_imp? (script 48, 49)

Per ciascun forecaster, framework applicato sulla sottomatrice ristretta ai suoi k≈10-14 imputer (incluso `no_imp`). Domanda specifica: `no_imp` è dentro l'equivalence set CD del Friedman best?

| Forecaster | k | Friedman best | no_imp pos. | Δrank | Kendall W | Cat. W | Imputer aiuta? |
|---|:---:|---|:---:|:---:|:---:|---|---|
| Global Mean | 14 | mediana_glob | 6°/14 | +2.444 | **0.469** | moderate | **SÌ generalizzabile** |
| MA_K56 | 14 | mediana_glob | 6°/14 | +2.131 | **0.463** | moderate | **SÌ generalizzabile** |
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

**Finding RQ1**: dicotomia chiara basata sull'architettura del forecaster:
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

**Conclusione 1.3.1**: il claim RQ1 ("imputer is irrelevant for MLP_M5/LGB_M5") sopravvive al sensitivity test sulla loss → finding robusto.

---

### 1.4 — Discussione strutturata dei risultati: dai dati grezzi alla decisione operativa

In questa sezione presentiamo i risultati della matrice 113-celle seguendo una **sequenza logica progressiva**: (1) dati grezzi → (2) framework rigoroso → (3) bridge operativo → (4) confronto delle due viste e implicazioni per il deployment.

> **⚠ Numeri ricalcolati post-K56 (2026-06-24)** — i conteggi Pareto qui sotto sono pre-K56. Valori corretti dai dati attuali (`hpo_matrix_pareto.parquet` + `friedman_nemenyi_ranks.parquet`): **WAPE Pareto = 25** (era 26), **Friedman/mean-rank Pareto = 15** (era 13), **intersezione doubly-optimal = 12** (era 11), **solo-WAPE = 13** (era 15), **solo-Friedman = 3** (era 2). Gli elenchi di celle per archetipo nello Step 4 vanno rigenerati con questi insiemi quando si scrive la sezione Results (deployment).

#### Step 1 — Dati grezzi: WAPE heatmap

WAPE è la metrica direttamente misurata sulle 50K serie di test (ore in-stock, gg 91-97). La heatmap (Fig. `fig_heatmap_general_no_imputeformer.png`) mostra `WAPE_h_med` per ciascuna cella della matrice in unità grezze, **senza applicare alcun framework analitico**.

Caratterizzazione qualitativa:
- Range osservato: WAPE_h_med ∈ [0.97, 1.30]
- Pattern visivo: famiglia MLP_M5 (colonna) ha WAPE bassi (~0.97-0.98) per molti imputer
- Naive aggregati (DoW Mean, Global Mean, MA_K56): WAPE moderato (~1.10-1.20) ma generalmente più alti
- Foundation models (Chronos, TimesFM): WAPE medio-alto

Questa è la **fotografia dei dati**. Non rappresenta ancora una decisione.

#### Step 2 — Framework rigoroso: Friedman + Kendall's W + Nemenyi CD

Per identificare il best in modo **statisticamente solido**, applichiamo il framework Friedman + W + Nemenyi CD (Demšar 2006) alla matrice [49,939 serie × 113 celle]. I 3 output:

**(2a) CD diagram** (`fig_cd_diagram.png`) — best cell + equivalence-set in dimensione 1:
- **Friedman best**: `itransformer__MLP_M5` (mean rank = **22.36** su 113)
- **Kendall W**: **0.459** (moderate) — ranking generalizzabile cross-serie
- **CD**: **0.903** rank units (α=0.05)
- **Equivalence-set**: 2 cells (best + `lgb__MLP_M5`, Δ = 0.51 ≤ CD)

**(2b) WAPE heatmap annotata** — la stessa heatmap dello Step 1, ora con bordi:
- bordo **blu** su `itransformer__MLP_M5` (Friedman best)
- bordo **arancione dashed** su `lgb__MLP_M5` (equiv-set)

→ Mostra visivamente che il best Friedman **non è la cella con WAPE_h_med minimo** della matrice (`timesnet__MLP_M5` ha mediana minore, 0.9731 < 0.9794). Questa apparente "incongruenza" sarà discussa nello Step 4.

**(2c) Pareto frontier su (mean rank, |WPE|)** (`fig_pareto_meanrank.png`) — multi-criterio rigoroso:
- 13 cells Pareto-optimal sotto criterio paired-rank
- Asse X coerente col framework Friedman
- Linea verticale a `best + CD = 23.27` → CD-equiv set visibile geometricamente

**Output dello Step 2**: il best statisticamente certificato + l'insieme delle 13 cells robust che bilanciano accuracy paired con bias control.

#### Step 3 — Bridge to operations: Pareto frontier su (WAPE, |WPE|)

Il framework Friedman dello Step 2 fornisce la certificazione statistica del best, ma è espresso in unità **mean rank** che non sono actionable per:
- SLA contrattuali (espressi in WAPE assoluto, es. "WAPE < 0.85")
- Calcolo del ROI (Δ WAPE × volume × margine)
- Dimensionamento safety stock (richiede percentili WAPE)
- Comunicazione con stakeholder non-tecnici

Per **tradurre la certificazione rigorosa in numeri operativi**, mostriamo la Pareto frontier su (WAPE_h_med, |WPE_h_med|) — `fig_pareto_hpo.png`:
- **26 cells Pareto-optimal** (più della Pareto Friedman = 13)
- Asse X in unità interpretabili (WAPE)
- 3 punti notevoli:
  - Min WAPE: `timesnet__MLP_M5` (WAPE 0.9731, |WPE| 0.886)
  - Knee point: `mediana_glob__dow_mean` (WAPE 1.10, |WPE| 0.19)
  - Min |WPE|: `linear_interp__timesfm` (WAPE 1.30, |WPE| 0.061)

**Output dello Step 3**: rappresentazione operativa dello stesso decision space, in unità che supportano SLA verification, ROI computation, safety stock sizing.

#### Step 4 — Confronto delle due Pareto frontiers

I due Pareto frontier (Step 2c e Step 3) **non sono in subset relation**: hanno intersezione significativa ma ciascuno include cells uniche.

| Insieme | # cells | Caratterizzazione | Implicazione operativa |
|---|:---:|---|---|
| **Intersezione** | 11 | doubly Pareto-optimal (robust su entrambi i criteri) | ★ **SAFEST CHOICES** |
| **Solo Friedman Pareto** | 2 | paired-robust ma marginally-dominated | safe per generalizzazione |
| **Solo WAPE Pareto** | 15 | median-good ma paired-fragile | da usare con awareness |

##### 4.1 — Le 2 cells uniche sulla Friedman Pareto

| Cell | WAPE_h_med | \|WPE\| | mean rank | Status |
|---|:---:|:---:|:---:|---|
| `itransformer__MLP_M5` ★ | 0.9794 | 0.8929 | 22.36 | Friedman best |
| `saits__MLP_M5` | 0.9790 | 0.8838 | 24.21 | Friedman 2nd |

Sono dominate su (WAPE, |WPE|) da `timesnet__MLP_M5` (WAPE minore, |WPE| minore) e `dlinear__TFT`. **Paradosso paired vs marginal**: vincono in paired comparison ma la mediana marginale è leggermente più alta. È il caso classico in cui la consistency paired prevale sull'avantaggio mediano.

##### 4.2 — Le 15 cells uniche sulla WAPE Pareto

Top case: `timesnet__MLP_M5` (WAPE_h_med = 0.9731, mean rank = 24.34). Ha la **mediana WAPE minima della matrice**, ma il mean rank è statisticamente distinguibile dal Friedman best (Δ = 1.98 > CD = 0.903). La sua bassa mediana deriva da **shape distribuzionale favorevole** (sotto-popolazione di serie dove timesnet eccelle), non da vittoria sistematica paired.

Le altre 14 cells si dividono in:
- 3 TFT cells con vari imputer (itransformer, linear_interp)
- 1 Chronos cell (linear_interp)
- 11 naive aggregati (DoW Mean, MA_K56, Global Mean × vari imputer)

Tutte hanno mean rank significativamente più alto del Friedman best (Δ > CD). Il loro vantaggio in WAPE Pareto deriva da shape distribuzionale, non da paired victory.

**"Statisticamente fragili"** (chiarimento): NON significa che le cells siano inaffidabili. Significa che la loro mediana favorevole può **non generalizzarsi** sotto distribution shift (mix di prodotti diverso, regime cambiato), perché non è "guadagnata" attraverso confronto paired ma da specifica conformazione del test set.

##### 4.3 — Le 11 cells dell'intersezione (doubly Pareto-optimal)

Sono robust su entrambi i criteri (paired + marginal). Si dividono in 3 archetipi:

| Archetype | Cell esempio | Mean rank | WAPE | \|WPE\| | Use case |
|---|---|:---:|:---:|:---:|---|
| **Accuracy-extreme** (TFT) | `dlinear__TFT` | 29.39 | 0.9785 | 0.858 | accuracy KPI |
| **Balanced** (TFT) | `linear_interp__TFT` | 44.96 | 1.002 | 0.477 | mix accuracy/bias |
| **Bias-extreme** (naive) | `media_cond__MA_K56` | 65.91 | 1.141 | 0.068 | inventory critica |

##### 4.4 — Sintesi operativa

Il workflow di deployment a 2 step diventa:

1. **Identificare le candidate cells**: prima Friedman Pareto (Step 2c) → filtra a 13 cells robust. Tra queste, scegliere l'archetype operativo (accuracy / balanced / bias-control).

2. **Quantificare la scelta**: poi Pareto WAPE (Step 3) → leggere valori assoluti (WAPE, |WPE|, distribuzione) per:
   - SLA verification
   - ROI computation (Δ WAPE × volume × margine)
   - Safety stock sizing (percentili WAPE)

L'**intersezione** delle due Pareto (11 cells) rappresenta le scelte **safest**: robust sia in paired comparison sia in marginal trade-off. Cells "solo Friedman" (2) sono safe per generalizzazione ma marginally meno attraenti; cells "solo WAPE" (15) richiedono awareness di shape-dependence.

---

## Sezione 2 — Recovery quality predice forecasting? (script 42 + 42b)

Per ogni serie i (i=1..50K), Spearman ρ_i tra `WAPE_recovery` (Traccia A) e `WAPE_forecasting_per_serie` (Traccia B). Distribuzione dei ~50K ρ_i sintetizzata con Cliff δ vs 0 + CI bootstrap 95% + categoria Romano.

**Aggiornamento 2026-06-15**: estesa Traccia A da 9 a **13 imputer** aggiungendo media_glob, media_cond, mediana_cond, lgb (esclusi originariamente — i loro WAPE_recovery erano disponibili in `traccia_a.parquet` ma non mappati negli script). Per TFT n=12 (manca imputeformer__TFT). I risultati cambiano qualitativamente rispetto a n=9.

| Forecaster | Famiglia | median ρ | Cliff δ vs 0 | CI 95% | Categoria Romano | Predice? |
|---|---|:---:|:---:|:---:|---|---|
| **MA_K56** ★ | naive | +0.76 | **+0.847** | [+0.842, +0.851] | **LARGE** | SÌ |
| **DoW Mean** | naive | +0.76 | **+0.849** | [+0.845, +0.854] | **LARGE** | SÌ |
| **Global Mean** | naive | +0.80 | **+0.841** | [+0.836, +0.846] | **LARGE** | SÌ (massimo) |
| **MLP_M5** | ML+lag | +0.09 | +0.212 | [+0.204, +0.220] | small | debole |
| **LGB_M5** | ML+lag | +0.08 | +0.171 | [+0.162, +0.179] | small | debole |
| **Chronos-bolt** | foundation | +0.21 | **+0.153** | [+0.144, +0.161] | **small** | debole |
| **TimesFM** | foundation | +0.13 | **+0.132** | [+0.124, +0.141] | **small** | debole |
| **TFT** | DL+lag | −0.32 | **−0.418** | [−0.426, −0.410] | **medium (inverso)** | no, segno opposto |

**Finding RQ2**: con il design completo a 13 imputer la dicotomia è **molto più netta** di quanto suggerisse l'analisi a 9 imputer:

- **Naive aggregati** (LARGE δ ≈ +0.84–0.88): la predizione è quasi una funzione lineare dei valori imputati → la recovery determina quasi deterministicamente il forecasting downstream.
- **Tutti gli altri forecaster** (small / medium inverso, |δ| ≤ 0.42): la recovery **non predice il forecasting in modo robusto**. I foundation models (Chronos, TimesFM) non differiscono qualitativamente dai ML+lag su questa dimensione.
- **TFT mostra un'associazione inversa medium** (δ = −0.42): tendenza a generare forecast migliori con imputer **peggiori** in recovery — segnale di selezione adversarial via attention mechanism, da indagare.

**Implicazione metodologica**: la valutazione di imputer via metriche di recovery (Traccia A) è un proxy **affidabile** solo se il forecaster downstream è un naive aggregato (cui la predizione è banalmente una funzione dell'output dell'imputer). Per **tutte le altre architetture** — incluso il presunto "imputer-sensitive" Chronos/TimesFM — la recovery quality non si traduce in forecasting quality in modo robusto.

Questa sezione fornisce la **chiave esplicativa** del Finding RQ1: l'imputer è irrelevante per MLP_M5/LGB_M5 perché i lag M5 isolano il forecasting dalla qualità dell'imputation; per Chronos/TimesFM/TFT esiste un'asimmetria architetturale ma non sufficientemente forte da rendere la recovery un proxy.

### 2.1 — Pairwise concordance: framework robusto sostitutivo (script 42c)

L'aggregate Spearman ρ su n=13 imputer ha SE ≈ ±0.32 → sensibile alla composizione del set di imputer. Sostituiamo con **pairwise concordance probability**:

```
Per ciascun forecaster fc:
   13 imputer → 78 coppie distinte (A, B)
   Per ciascuna coppia:
      Per ciascuna delle 50K serie:
         "concordant" se sign(rec_A − rec_B) = sign(fc_A − fc_B)
      P_pair = mean(concordant)  ← stimato su N=50K serie (SE ≈ ±0.002)
   78 valori P_pair per forecaster → distribuzione interpretabile
```

| Forecaster | Mediana P(concord) | CI95 bootstrap | % coppie > 0.5 | Lettura |
|---|:---:|:---:|:---:|---|
| **MA_K56** | **0.822** | [0.785, 0.853] | 82% | recovery predice fortemente |
| **DoWMean** | **0.816** | [0.788, 0.841] | 87% | predice fortemente |
| **GlobalMean** | **0.827** | [0.802, 0.850] | 86% | predice fortemente |
| TimesFM | 0.560 | [0.490, 0.617] | 60% | predice debolmente |
| Chronos-bolt | 0.559 | [0.461, 0.717] | 58% | predice debolmente |
| LGB_M5 | 0.525 | [0.516, 0.536] | 81% | quasi random |
| MLP_M5 | 0.510 | [0.502, 0.519] | 63% | random |
| **TFT** | **0.399** | [0.314, 0.484] | 35% | **INVERSO** |

**Vantaggi sull'aggregate Spearman**: ogni misura basata su N=50K invece di n=13, 78 misure invece di 1, robustezza all'inclusione/esclusione di singoli imputer.

### 2.2 — Stratificazione per volume quartile (script 42c2)

Il rapporto recovery → forecasting **dipende dal regime di volume** in modo non triviale:

| Forecaster | Globale | Q1 | Q2 | Q3 | Q4 | Trend |
|---|:---:|:---:|:---:|:---:|:---:|---|
| GlobalMean | 0.83 | **0.94** | 0.91 | 0.81 | 0.67 | **↓ decresce con volume** |
| DoWMean | 0.82 | 0.91 | 0.89 | 0.80 | 0.68 | ↓ |
| MA_K56 | 0.82 | 0.93 | 0.89 | 0.79 | 0.66 | ↓ |
| Chronos-bolt | 0.56 | 0.65 | 0.68 | 0.61 | **0.33** | **↓ crollo in Q4 → inverso** |
| TimesFM | 0.56 | 0.57 | 0.59 | 0.56 | 0.46 | ↓ debole |
| MLP_M5 | 0.51 | 0.50 | 0.50 | 0.51 | 0.53 | invariato (~0.5) |
| LGB_M5 | 0.52 | 0.51 | 0.51 | 0.53 | 0.55 | invariato (~0.5) |
| **TFT** | 0.40 | 0.38 | 0.41 | 0.43 | **0.34** | inverso ovunque, picco in Q4 |

**Finding RQ2 stratificato** — 4 pattern distinti per quartile:

1. **Naive aggregati: dipendenza recovery → forecasting ALTA in basso volume, DECRESCENTE con volume**. In Q1 (P≈0.92) il forecasting naive è quasi una funzione lineare dell'imputer; in Q4 (P≈0.67) ci sono abbastanza dati osservati da diluire il contributo dell'imputer.
2. **Chronos: crollo drammatico in Q4** (P=0.33 < 0.5). In alto volume Chronos diventa **anti-correlato con recovery**: imputer "troppo smooth" interferiscono con l'attention. **Boundary condition del transfer learning**.
3. **ML+lag: indifferenti al regime** (P ≈ 0.50 ovunque). I lag features compensano l'imputer in ogni quartile — coerente con W ≈ 0 di RQ1.
4. **TFT: inversione strutturale** (P 0.34-0.43 in ogni Q). Pattern adversarial **non regime-specifico**, ma amplificato in alto volume (Q4).

**Implicazioni**:
- La validità della Traccia A come proxy del forecasting **dipende dal regime di volume**: forte in basso volume (anche per Chronos), assente o inversa in alto volume.
- La **convergenza in Q4** (tutti i forecaster vicino o sotto 0.5) è coerente con RQ4 (Sezione 1.2): in alto volume le famiglie di forecaster convergono e l'imputer effect si attenua.
- Il crollo Chronos in Q4 è un finding nuovo che suggerisce una limitazione del foundation model in regime di alto volume — meritevole di approfondimento.

### 2.3 — Tentativo di spiegazione meccanicistica per TFT inverso (script 42d)

Per indagare l'inversione TFT, abbiamo testato l'ipotesi «*TFT preferisce imputer dinamici*» calcolando per ciascun imputer la **DYNAMICITY** (std del residuo dopo aver rimosso il pattern orario, normalizzato per la variabilità naturale dei valori osservati):

```
DYN_i = std(ε_imp_i) / std(ε_obs_i)
    dove ε(d,h) = y(d,h) − μ_serie(h)
```

**Risultato**: ipotesi REJECTED. Spearman ρ tra DYN ranking e TFT ranking = **−0.01** (no correlazione).

**Finding inatteso**: l'analisi DYN ha rivelato un pattern diverso ma robusto:

| Forecaster | Spearman ρ (DYN vs forecasting rank) | Interpretazione |
|---|:---:|---|
| **GlobalMean** | **+0.74** | preferisce fortemente imputer STATICI (alto DYN → peggior rank) |
| **DoWMean** | **+0.74** | idem |
| MA_K56 | +0.63 | preferisce statici |
| TimesFM | +0.55 | preferisce moderatamente statici |
| Chronos-bolt | +0.46 | preferisce moderatamente statici |
| LGB_M5 | −0.12 | indifferente |
| MLP_M5 | −0.01 | indifferente |
| **TFT** | **−0.01** | **indifferente alla dinamicità** ★ |

La preferenza naive per imputer statici è **logica** (i naive usano direttamente i valori imputati nella mean/median; imputer dinamici aggiungono rumore). L'**indifferenza TFT alla DYN** falsifica l'ipotesi originale: l'inversione TFT non è spiegata dalla dinamicità.

**Conclusione**: l'anomalia TFT non è spiegata dalla DYN. Una possibile ipotesi alternativa (architectural alignment DL vs non-DL — i top 4 imputer per TFT sono tutti DL: dlinear, timesnet, itransformer, saits) richiede future work per essere formalmente testata. Riportiamo l'osservazione come finding aperto.

---

## Sezione 3 — Foundation models per retail

- **Chronos-bolt** × no_imp: WAPE_h_med = 1.007 → competitivo, Pareto solo a Q2 con `forward_fill`.
- **TimesFM 2.5-200M** × 14 imputer: WAPE_h_med best = 1.191 (imputeformer), peggio di Chronos del 18% (CPU only, 5x più lento). I 4 imputer aggiunti tardivamente (lgb, mediana_cond, media_cond, media_glob) producono WAPE_med 1.24–1.27 → vanno tutti al fondo del ranking interno di TimesFM, abbassando Kendall W (k=10: 0.229 → k=14: 0.174).

**Finding RQ5**: i foundation models sono **dominati da MLP_M5** sulla matrice principale (mean rank ≈ 35-40 vs 22.3 di `itransformer__MLP_M5`). Best imputer coerente per entrambi: **imputeformer**. **Aggiornamento post-estensione Traccia A**: con n=13 imputer, foundation e ML+lag mostrano sensibilità alla recovery di entità simile (small, |δ| ≤ 0.22) — i foundation non hanno l'asimmetria forte che suggeriva il design parziale a n=9. Utili come baseline zero-shot ma non competitivi su retail deperibile con lag features disponibili.

---

## Sintesi findings per il paper (ordinata per RQ — logically sustainable)

**RQ1 — Existence: per ciascun forecaster, l'imputer aiuta vs no_imp? (Sezione 1.3)**
1. **Dicotomia chiara**: MLP_M5/LGB_M5 hanno W ≈ 0 (negligible) → imputer praticamente irrilevante. Foundation, TFT, naive hanno W ≥ 0.22 (small-moderate) → imputer aiuta, best coerente (imputeformer per foundation, dlinear per TFT, mediana_glob per naive).
2. **Sensitivity loss MAE vs MSE (1.3.1)**: il claim "imputer doesn't matter" è stabile sotto entrambe le loss → finding strutturale.

**RQ2 — Mechanism: la recovery predice il forecasting? (Sezione 2 — n=13 imputer, pairwise concordance)**
3. **Dicotomia naive vs others**: naive aggregati con P(concord) ≈ 0.80-0.83 (recovery determina forecasting); tutti gli altri forecaster (ML+lag ~0.51, foundation ~0.56, TFT 0.40 inverso) con relazione debole o nulla.
4. **Stratificazione per volume** (Sez. 2.2): dipendenza recovery → forecasting **decresce con il volume per tutti i forecaster eccetto ML+lag**. Naive scendono da 0.91 (Q1) a 0.67 (Q4); Chronos crolla a 0.33 (inverso) in Q4 — boundary condition del transfer learning.
5. **TFT inverso strutturale** (Sez. 2.2): P(concord) 0.34-0.43 in ogni quartile, non regime-specifico. L'ipotesi "TFT preferisce imputer dinamici" è rifiutata (DYN analysis, Sez. 2.3); la spiegazione meccanicistica rimane aperta.
6. **Naive preferiscono imputer statici** (Sez. 2.3): Spearman ρ tra DYNAMICITY ranking e naive ranking ≈ +0.74 — finding nuovo che spiega meccanisticamente perché i naive sono recovery-sensitive (alto-recovery → spesso statico → naive ottimale).

**RQ3 — Identification: qual è il best? (Sezione 1.1)**
6. **Best globale**: `itransformer__MLP_M5` (mean rank 22.33 su 113 celle), equivalence set di 2 cells entrambe MLP_M5 (CD = 0.903). Kendall W = 0.459 moderate.
7. **Pareto trade-off (1.1 b)**: 25/113 cells Pareto-optimal (post-K56). Trade-off strutturale accuracy ↔ bias: ML/DL hanno |WPE| ≥ 0.77, naive aggregati riducono bias pagando in WAPE. Knee = `mediana_glob__dow_mean` (1.101 / 0.190).

**RQ4 — Conditions: cambia per regime di volume? (Sezione 1.2)**
8. **Soft crossover** (1.2 a): famiglia MLP_M5 invariata in ogni Q; imputer cambia (`lgb` in Q1/Q2, `itransformer` in Q3/Q4). Q4 più discriminante (2 equiv), Q1 più saturato (12 equiv).
9. **Pareto per quartile** (1.2 b): 28→12 cells Pareto-optimal da Q1 a Q4. TFT dominante low-bias in Q1/Q2; MLP_M5/naive si specializzano in Q3/Q4.
10. **Crossover line-plot** (1.2 c): convergenza generalizzata in Q4 (ML/TFT/naive ≈ 0.77); Chronos piatto; TimesFM peggiore.

**RQ5 — Boundary: foundation models alterano queste conclusioni? (Sezione 3)**
11. Chronos e TimesFM sono **dominati da MLP_M5** sulla matrice. Best imputer per entrambi: imputeformer. Con n=13 imputer, foundation e ML+lag mostrano sensibilità alla recovery simile (small) — i foundation non hanno l'asimmetria forte suggerita dal design parziale.

→ **Messaggio scientifico chiave**: "*la famiglia MLP_M5 domina il benchmark in ogni regime; dentro MLP_M5 la scelta dell'imputer è praticamente irrilevante (W ≈ 0) perché i lag features M5 disaccoppiano il forecasting dalla recovery quality. L'imputer conta solo per i forecaster senza lag (foundation models e naive aggregati), dove la recovery predice direttamente il forecasting (Cliff δ vs 0 ≥ +0.47 medium-LARGE).*"

---

## Framework statistico del paper (riferimento unico)

| RQ | Sezione | Test | Effect size | Script |
|---|---|---|---|---|
| RQ1 (imputer aiuta?) | 1.3 | Friedman χ² (per fc) | Kendall's W + Nemenyi CD | 48, 49 |
| _sensitivity_ | 1.3.1 | Wilcoxon paired | Cliff δ | 38 |
| RQ2 (recovery → forecasting) | 2 | Wilcoxon vs 0 (su ρ_i) | **Cliff δ vs 0** + CI bootstrap (cat. Romano) | 42, 42b |
| RQ3 (best globale) | 1.1 (a) | Friedman χ² | **Kendall's W** + Nemenyi CD | 45 |
| RQ3 (Pareto) | 1.1 (b) | dominance | n. Pareto-optimal cells | 35 |
| RQ4 (best per Q) | 1.2 (a) | Friedman χ² | Kendall's W + Nemenyi CD | 46 |
| RQ4 (Pareto per Q) | 1.2 (b) | dominance | n. Pareto-optimal cells | 36 |
| RQ4 (crossover line) | 1.2 (c) | descrittivo | Δ WAPE per famiglia per Q | 39 |
| RQ5 (foundation models) | 3 | come RQ1 + RQ2 + RQ3 | come RQ1 + RQ2 + RQ3 | 45, 42b |

**Niente TOST. Niente soglia Cliff δ < 0.147 come decision rule per "equivalence" tra k metodi.**
Script TOST/threshold-based (43, 44, 47) restano come **supplementary** ma non sono citati nel paper.

---

## Riferimento bibliografico chiave

- **Wang et al. (2025)** — FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail. arXiv:2505.16319. (Autori: Y. Wang, J. Gu, L. Long, X. Li, L. Shen, Z. Fu, X. Zhou, X. Jiang. Nota: precedentemente citato erroneamente come "Liu et al.")
- **Du et al. (2023)** — SAITS: Self-Attention-based Imputation for Time Series. NeurIPS.
- **Zeng et al. (2022)** — DLinear: Are Transformers Effective for Time Series Forecasting? AAAI 2023.
- **Ansari et al. (2024)** — Chronos: Learning the Language of Time Series. arXiv:2403.07815.
- **Du et al. (2023)** — PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series. arXiv:2305.18811.
- **Demšar (2006)** — Statistical comparisons of classifiers over multiple data sets. JMLR.
- **Romano et al. (2006)** — Appropriate statistics for ordinal level data: Should we really be using t-test and Cohen's d. Cliff δ thresholds.
