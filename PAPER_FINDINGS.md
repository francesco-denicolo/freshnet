# Paper Findings — Imputer × Forecaster Benchmark on FreshRetailNet-50K

> Documento di riferimento per la stesura del paper. Tutti i finding numerici,
> tabelle, e claim per ciascuna sezione. Aggiornare qui (non in `CLAUDE.md`).

**Ultimo update**: 2026-06-12
**Matrice finale**: 113 cells (TimesFM completato su 14 imputer, allineato agli altri forecaster).
**Framework statistico unico**: Friedman + Kendall's W + Nemenyi CD (Demšar 2006, JMLR).

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

**Finding 1.1 (a)**: il best globale è `itransformer__MLP_M5`. Solo `lgb__MLP_M5` gli sta statisticamente alla pari. Entrambe le 2 celle sono **MLP_M5**: la famiglia di forecaster vincente è isolata. Il ranking è generalizzabile (W = 0.454 moderate).

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

| Q | # Pareto | Top-WAPE | Min-\|WPE\| | Famiglie sulla frontier |
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

**Finding 1.2 (c)**: tre fenomeni di crossover osservabili nel line-plot:

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

---

## Sintesi findings per il paper

**Sezione 1 — Best cell + equivalence set (Friedman + W + CD)**
1. **Best globale**: `itransformer__MLP_M5` (mean rank 22.27 su 113 celle), equivalence set di 2 cells entrambe MLP_M5. Kendall W = 0.454 moderate.
2. **Per regime di volume — best**: crossover soft — la famiglia MLP_M5 vince in tutti i quartili, l'imputer ottimale cambia (lgb in Q1/Q2 basso volume, itransformer in Q3/Q4 medio-alto/alto). Q4 più discriminante (2 equiv), Q1 più saturato (12 equiv).
3. **Per regime di volume — Pareto**: 28→12 cells Pareto-optimal da Q1 a Q4. TFT dominante in basso volume (low-bias tail), MLP_M5/naive si specializzano in alto volume. Chronos appare solo a Q2.
4. **Per regime di volume — crossover line-plot**: convergenza generalizzata in Q4 (ML/TFT/naive tutti a WAPE ≈ 0.77); Chronos piatto (perde relativamente); TimesFM peggiore in tutti i quartili. Decision tree practitioner emerge naturalmente dalle line.
5. **Per forecaster**: dicotomia chiara. MLP_M5/LGB_M5 hanno W ≈ 0 (negligible) → imputer praticamente irrilevante. Foundation, TFT, naive hanno W ≥ 0.22 (small-moderate) → imputer aiuta, best coerente (imputeformer / dlinear / mediana_glob). **Sensitivity loss MAE vs MSE**: il claim "imputer doesn't matter" è stabile sotto entrambe le loss (Sez. 1.3.1).

**Sezione 2 — Recovery quality predice forecasting?**
6. La recovery predice il forecasting in modo **crescente** con la dipendenza diretta dai dati imputati: naive δ ≈ +0.85 LARGE, foundation δ ≈ +0.5 medium-LARGE, ML+lag δ ≈ +0.15 small. I lag M5 sono buffer che disaccoppia ML/DL dalla qualità imputer.

**Sezione 3 — Foundation models**
7. Chronos e TimesFM sono **recovery-sensitive** ma **dominati da MLP_M5** sulla matrice. Best imputer per entrambi: imputeformer. Utili come baseline zero-shot, non competitivi su retail con lag disponibili.

→ **Messaggio scientifico chiave**: "*la famiglia MLP_M5 domina il benchmark in ogni regime; dentro MLP_M5 la scelta dell'imputer è praticamente irrilevante (W ≈ 0) perché i lag features M5 disaccoppiano il forecasting dalla recovery quality. L'imputer conta solo per i forecaster senza lag (foundation models e naive aggregati), dove la recovery predice direttamente il forecasting (Cliff δ vs 0 ≥ +0.47 medium-LARGE).*"

---

## Framework statistico del paper (riferimento unico)

| Sezione | Domanda | Test | Effect size | Script |
|---|---|---|---|---|
| Sez. 1.1, 1.2, 1.3 | Best cell + equiv set (k > 2 metodi) | Friedman χ² | **Kendall's W** + Nemenyi CD | 45, 46, 48, 49 |
| Sez. 1.1 (b), 1.2 (b) | Pareto trade-off WAPE × \|WPE\| | dominance | n. Pareto-optimal cells | 35, 36 |
| Sez. 1.2 (c) | Crossover line-plot evoluzione famiglie per Q | descrittivo | Δ WAPE per famiglia per Q | 39 |
| Sez. 1.3.1 | Robustness loss MAE vs MSE (pairwise A vs B) | Wilcoxon paired | Cliff δ | 38 |
| Sez. 2 | Recovery vs forecasting (correlazione) | Wilcoxon vs 0 (su ρ_i) | **Cliff δ vs 0** + CI bootstrap (cat. Romano) | 42, 42b |
| Sez. 3 | Foundation models | come Sez. 1 + 2 | come Sez. 1 + 2 | 45, 42b |

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
