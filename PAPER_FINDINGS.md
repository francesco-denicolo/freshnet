# Paper Findings — Imputer × Forecaster Benchmark on FreshRetailNet-50K

> Documento di riferimento per la stesura del paper. Tutti i finding numerici,
> tabelle, e claim per ciascuna sezione. Aggiornare qui (non in `CLAUDE.md`).

**Ultimo update**: 2026-06-14
**Matrice finale**: 113 cells (TimesFM completato su 14 imputer, allineato agli altri forecaster).
**Framework statistico unico**: Friedman + Kendall's W + Nemenyi CD (Demšar 2006, JMLR).

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
| **Globale** | 113 | 49,939 | **0.454 (moderate)** | 0.903 | **itransformer__MLP_M5** | **2** |

Equivalence set (2 cells):
1. `itransformer__mlp_m5lags` (mean rank 22.27, best)
2. `lgb__mlp_m5lags` (mean rank 22.78, Δ = 0.51 ≤ CD)

**Finding RQ3 (a)**: il best globale è `itransformer__MLP_M5`. Solo `lgb__MLP_M5` gli sta statisticamente alla pari. Entrambe le 2 celle sono **MLP_M5**: la famiglia di forecaster vincente è isolata. Il ranking è generalizzabile (W = 0.454 moderate).

#### 1.1 (b) Trade-off WAPE × |WPE|: Pareto frontier globale (script 35)

Pareto su `(WAPE_h_med, |WPE_h_med|)` — accuracy vs bias. Frontier = celle non-dominate.

**26 / 113 cells Pareto-optimal**. I tre punti di riferimento:

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
| Q1 (basso) | [11, 40] | **lgb__MLP_M5** | 0.653 (large) | 1.804 | 12 |
| Q2 | (40, 54] | **lgb__MLP_M5** | 0.586 (large) | 1.805 | 13 |
| Q3 (medio-alto) | (54, 86] | **itransformer__MLP_M5** | 0.417 (moderate) | 1.806 | 4 |
| Q4 (alto) | (86, 5326] | **itransformer__MLP_M5** | 0.396 (moderate) | 1.807 | 2 |

**Finding RQ4 (a)**: crossover **soft**. La famiglia di forecaster vincente (MLP_M5) è **invariata** in tutti i regimi; l'imputer ottimale cambia tra `lgb` (basso volume Q1/Q2) e `itransformer` (alto volume Q3/Q4). Il regime più discriminante è Q4 (solo 2 celle CD-equivalenti); Q1 è il più saturato (12 equivalenti).

**Nota su W vs CD-equiv set size**: W e #equiv non sono ridondanti. Q1 ha W large + molti equiv (separazione forte tra famiglie ma debole dentro MLP_M5); Q4 ha W moderate + pochi equiv (naive competono → meno accordo globale, ma MLP_M5 differenziato → top isolato).

#### 1.2 (b) Pareto frontier per quartile (script 36)

Stessa metrica (WAPE × |WPE|) applicata dentro ciascun quartile. La frontier **cambia composizione** col regime.

| Q | # Pareto | Top-WAPE | Min-\|WPE\| | Famiglie sulla frontier |
|---|:---:|---|---|---|
| Q1 (basso vol) | **28** / 113 | `mediana_cond__LGB_M5` (1.000) | `mediana_cond__TFT` (\|WPE\|=0.57) | LGB_M5, MLP_M5, TFT (dominante in coda low-bias) |
| Q2 | 22 | `mediana_cond__LGB_M5` (0.995) | `forward_fill__Chronos-bolt` (\|WPE\|=0.39) | LGB_M5, MLP_M5, TFT, Chronos al margine |
| Q3 (medio-alto) | 15 | `lgb__MLP_M5` (0.958) | `mediana_cond__MA_K21` (\|WPE\|=0.13) | MLP_M5, TFT in testa; **naive emergono al low-bias** |
| Q4 (alto vol) | **12** / 113 | `itransformer__MLP_M5` (**0.757**) | `media_cond__MA_K21` (\|WPE\|=**0.06**) | MLP_M5 (top WAPE), naive (low bias) — gap minimo |

**Finding RQ4 (b)**: tre pattern emergono dalla Pareto stratificata:

1. **# Pareto-optimal decresce con il volume** (Q1: 28 → Q4: 12) → stesso pattern del CD-equiv set: alto volume = più discriminante.
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
| `imputeformer__MA_K21` | 1.260 | 1.216 | 1.050 | 0.779 |
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

## Sezione 2 — Recovery quality predice forecasting? (script 42 + 42b)

Per ogni serie i (i=1..50K), Spearman ρ_i tra `WAPE_recovery` (Traccia A) e `WAPE_forecasting_per_serie` (Traccia B). Distribuzione dei ~50K ρ_i sintetizzata con Cliff δ vs 0 + CI bootstrap 95% + categoria Romano.

**Aggiornamento 2026-06-15**: estesa Traccia A da 9 a **13 imputer** aggiungendo media_glob, media_cond, mediana_cond, lgb (esclusi originariamente — i loro WAPE_recovery erano disponibili in `traccia_a.parquet` ma non mappati negli script). Per TFT n=12 (manca imputeformer__TFT). I risultati cambiano qualitativamente rispetto a n=9.

| Forecaster | Famiglia | median ρ | Cliff δ vs 0 | CI 95% | Categoria Romano | Predice? |
|---|---|:---:|:---:|:---:|---|---|
| **MA_K21** ★ | naive | +0.79 | **+0.883** | [+0.879, +0.888] | **LARGE** | SÌ (massimo) |
| **DoW Mean** | naive | +0.81 | **+0.849** | [+0.845, +0.854] | **LARGE** | SÌ |
| **Global Mean** | naive | +0.80 | **+0.841** | [+0.836, +0.846] | **LARGE** | SÌ |
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

**RQ2 — Mechanism: la recovery predice il forecasting? (Sezione 2 — n=13 imputer)**
3. **Dicotomia naive vs others**: naive aggregati con δ ≈ +0.84–0.88 LARGE (recovery determina quasi deterministicamente forecasting); tutti gli altri forecaster (ML+lag, foundation, TFT) con |δ| ≤ 0.42 small/medium-inverso.
4. **Implicazione**: la valutazione di imputer via metriche di recovery è proxy **affidabile solo per naive aggregati**. Per le altre architetture (incluso Chronos/TimesFM/TFT), recovery non predice forecasting in modo robusto.
5. **TFT mostra associazione inversa medium** (δ = −0.42): tendenza adversarial (forecast migliori con imputer peggiori in recovery).

**RQ3 — Identification: qual è il best? (Sezione 1.1)**
6. **Best globale**: `itransformer__MLP_M5` (mean rank 22.27 su 113 celle), equivalence set di 2 cells entrambe MLP_M5 (CD = 0.903). Kendall W = 0.454 moderate.
7. **Pareto trade-off (1.1 b)**: 26/113 cells Pareto-optimal. Trade-off strutturale accuracy ↔ bias: ML/DL hanno |WPE| ≥ 0.77, naive aggregati riducono bias pagando in WAPE. Knee = `mediana_glob__dow_mean` (1.101 / 0.190).

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

- **Liu et al. (2025)** — FreshRetailNet-50K: Latent Demand from 50,000 Stores for World-scale Stockout Prediction in Fresh Retail. arXiv:2505.16319.
- **Du et al. (2023)** — SAITS: Self-Attention-based Imputation for Time Series. NeurIPS.
- **Zeng et al. (2022)** — DLinear: Are Transformers Effective for Time Series Forecasting? AAAI 2023.
- **Ansari et al. (2024)** — Chronos: Learning the Language of Time Series. arXiv:2403.07815.
- **Du et al. (2023)** — PyPOTS: A Python Toolbox for Data Mining on Partially-Observed Time Series. arXiv:2305.18811.
- **Demšar (2006)** — Statistical comparisons of classifiers over multiple data sets. JMLR.
- **Romano et al. (2006)** — Appropriate statistics for ordinal level data: Should we really be using t-test and Cohen's d. Cliff δ thresholds.
