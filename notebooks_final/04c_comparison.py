"""
04c_comparison.py — Confronto Fase A: boxplot + test statistici
================================================================
Carica i risultati per-serie di tutti gli 8 modelli baseline (Fase A)
e produce:
  1. Boxplot WAPE orario per-serie (in-stock)
  2. Boxplot WPE orario per-serie (in-stock)
  3. Boxplot WAPE giornaliero per-serie (in-stock)
  4. Tabella riepilogativa (pooled + mediana + IQR)
  5. Test di Wilcoxon signed-rank (pairwise, su WAPE orario)
  6. Matrice p-value heatmap
  7. Bootstrap CI sulla differenza di WAPE mediana

Eseguire con: freshnet/bin/python notebooks_final/04c_comparison.py
"""

import sys
import os
import functools
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Caricamento risultati per-serie
# ---------------------------------------------------------------------------
print('=' * 72)
print('  CONFRONTO FASE A — Tutti i modelli baseline')
print('=' * 72)

MODEL_FILES = {
    'Global Mean':   'naive_global_mean_test_per_series.parquet',
    'DoW Mean':      'naive_dow_mean_test_per_series.parquet',
    'Naive Direct':  'naive_naive_direct_test_per_series.parquet',
    'MA (K=21)':     'naive_ma_k21_test_per_series.parquet',
    'LGB (no lags)': 'lgb_nolags_test_per_series.parquet',
    'LGB (M5 lags)': 'lgb_m5lags_test_per_series.parquet',
    'MLP (no lags)': 'mlp_nolags_test_per_series.parquet',
    'MLP (M5 lags)': 'mlp_m5lags_test_per_series.parquet',
}

print('\n1. Caricamento risultati per-serie (test)...')
model_data = {}
for label, fname in MODEL_FILES.items():
    path = os.path.join(RESULTS_DIR, fname)
    df = pd.read_parquet(path)
    model_data[label] = df
    print(f'  {label:<18} {len(df):,} serie')

MODELS = list(MODEL_FILES.keys())
N_MODELS = len(MODELS)

# ---------------------------------------------------------------------------
# 2. Tabella riepilogativa
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  2. TABELLA RIEPILOGATIVA (test, in-stock)')
print('=' * 72)

METRICS = ['hourly_wape', 'hourly_wpe', 'daily_wape', 'daily_wpe']

print(f'\n  {"Model":<18} '
      f'{"WAPE_h med":>11} {"WAPE_h Q25":>11} {"WAPE_h Q75":>11} '
      f'{"WPE_h med":>10} '
      f'{"WAPE_d med":>11} {"WPE_d med":>10}')
print('  ' + '-' * 90)

for label in MODELS:
    df = model_data[label]
    wh = df['hourly_wape'].dropna()
    wph = df['hourly_wpe'].dropna()
    wd = df['daily_wape'].dropna()
    wpd = df['daily_wpe'].dropna()
    q25, q75 = wh.quantile(0.25), wh.quantile(0.75)
    print(f'  {label:<18} '
          f'{wh.median():>11.4f} {q25:>11.4f} {q75:>11.4f} '
          f'{wph.median():>10.4f} '
          f'{wd.median():>11.4f} {wpd.median():>10.4f}')

# ---------------------------------------------------------------------------
# 3. Boxplot
# ---------------------------------------------------------------------------
print('\n3. Generazione boxplot...')

COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2',
          '#CCB974', '#DD8452', '#64B5CD', '#8C564B']

def make_boxplot(metric, ylabel, title, figname, show_zero=False):
    fig, ax = plt.subplots(figsize=(14, 7))

    box_data = []
    medians = []
    for label in MODELS:
        vals = model_data[label][metric].dropna()
        if metric.startswith('wape'):
            clipped = vals.clip(upper=vals.quantile(0.99))
        else:
            clipped = vals.clip(lower=vals.quantile(0.01), upper=vals.quantile(0.99))
        box_data.append(clipped.values)
        medians.append(vals.median())

    bp = ax.boxplot(box_data, tick_labels=MODELS, patch_artist=True, widths=0.65,
                    showfliers=False)
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for ml in bp['medians']:
        ml.set_color('red')
        ml.set_linewidth(2)

    # Annotate medians
    for i, med in enumerate(medians):
        ax.text(i + 1, med, f' {med:.4f}', ha='left', va='center',
                fontsize=9, fontweight='bold', color='red')

    if show_zero:
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='x', rotation=25, labelsize=10)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, figname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Salvata: {figname}')

make_boxplot('hourly_wape', 'WAPE (hourly, in-stock)',
             'Fase A — WAPE orario per-serie (test, in-stock)',
             'fig05_boxplot_hourly_wape.png')

make_boxplot('hourly_wpe', 'WPE (hourly, in-stock)',
             'Fase A — WPE orario per-serie (test, in-stock)',
             'fig06_boxplot_hourly_wpe.png', show_zero=True)

make_boxplot('daily_wape', 'WAPE (daily, in-stock)',
             'Fase A — WAPE giornaliero per-serie (test, in-stock)',
             'fig07_boxplot_daily_wape.png')

make_boxplot('daily_wpe', 'WPE (daily, in-stock)',
             'Fase A — WPE giornaliero per-serie (test, in-stock)',
             'fig08_boxplot_daily_wpe.png', show_zero=True)

# ---------------------------------------------------------------------------
# 4. Wilcoxon signed-rank test (pairwise, hourly WAPE)
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  4. TEST DI WILCOXON SIGNED-RANK (WAPE orario, pairwise)')
print('=' * 72)

# Build aligned WAPE series (same series across all models)
# Merge on (store_id, product_id), keep only series present in all models
print('\n  Allineamento serie...')
merged = None
for label in MODELS:
    df = model_data[label][['store_id', 'product_id', 'hourly_wape']].copy()
    df = df.rename(columns={'hourly_wape': label})
    if merged is None:
        merged = df
    else:
        merged = merged.merge(df, on=['store_id', 'product_id'], how='inner')

# Drop series with NaN in any model
merged_clean = merged.dropna()
n_series = len(merged_clean)
print(f'  Serie allineate (no NaN): {n_series:,}')

# Pairwise Wilcoxon tests with rank-biserial correlation (effect size)
#
# Rank-biserial r = 1 - (2W) / (n*(n+1)/2)  where W = Wilcoxon W-statistic
# and n = number of non-zero differences.
# Interpretation (Kerby 2014, adapted from Cohen):
#   |r| < 0.10  → negligible
#   |r| < 0.30  → small
#   |r| < 0.50  → medium
#   |r| >= 0.50 → large
#
# Practical relevance threshold: |Δ median WAPE| >= 0.02 (~2% of typical WAPE)
# A comparison is "practically significant" only if BOTH p < 0.05 AND |r| >= 0.10.

EFFECT_THRESH = 0.10   # minimum |r| for practical relevance
DELTA_THRESH = 0.02    # minimum |Δ median| for practical relevance

print(f'\n  Soglie di rilevanza pratica: |r| >= {EFFECT_THRESH}, |Δmed| >= {DELTA_THRESH}')

print(f'\n  {"Model A":<18} {"Model B":<18} {"med(A-B)":<10} '
      f'{"r_rb":>8} {"effect":>10} {"p-value":>12} {"practical":>10}')
print('  ' + '-' * 100)

pvalue_matrix = np.ones((N_MODELS, N_MODELS))
effect_matrix = np.zeros((N_MODELS, N_MODELS))
wilcoxon_results = []

for i, j in combinations(range(N_MODELS), 2):
    a_label, b_label = MODELS[i], MODELS[j]
    a_vals = merged_clean[a_label].values
    b_vals = merged_clean[b_label].values
    diff = a_vals - b_vals

    # Remove zeros (Wilcoxon requires non-zero differences)
    nonzero = diff != 0
    n_nz = int(nonzero.sum())

    if n_nz < 10:
        w_stat, p_val, r_rb = np.nan, np.nan, np.nan
    else:
        res = stats.wilcoxon(a_vals[nonzero], b_vals[nonzero],
                             alternative='two-sided')
        w_stat, p_val = res.statistic, res.pvalue
        # Rank-biserial correlation
        # scipy's wilcoxon returns the smaller of W+ and W-
        # r = 1 - 2*W / (n*(n+1)/2)
        # But with large N, use the method_result if available
        # For matched pairs: r = 1 - (2 * T) / (n * (n + 1) / 2)
        # where T is the test statistic (sum of ranks of smaller sign)
        n_pairs = n_nz
        r_rb = 1.0 - (2.0 * w_stat) / (n_pairs * (n_pairs + 1) / 2)

    pvalue_matrix[i, j] = p_val
    pvalue_matrix[j, i] = p_val
    effect_matrix[i, j] = r_rb
    effect_matrix[j, i] = -r_rb  # sign flipped

    med_a = np.median(a_vals)
    med_b = np.median(b_vals)
    med_diff = np.median(diff)

    # Effect size label
    abs_r = abs(r_rb) if not np.isnan(r_rb) else 0
    if abs_r >= 0.50:
        eff_label = 'large'
    elif abs_r >= 0.30:
        eff_label = 'medium'
    elif abs_r >= 0.10:
        eff_label = 'small'
    else:
        eff_label = 'negligible'

    # Practical significance: needs both statistical sig AND meaningful effect
    is_stat_sig = (not np.isnan(p_val)) and p_val < 0.05
    is_practical = is_stat_sig and abs_r >= EFFECT_THRESH and abs(med_diff) >= DELTA_THRESH
    pract_label = 'YES' if is_practical else 'no'

    wilcoxon_results.append({
        'model_a': a_label, 'model_b': b_label,
        'med_a': med_a, 'med_b': med_b, 'med_diff': med_diff,
        'w_stat': w_stat, 'p_value': p_val, 'r_rb': r_rb,
        'effect_label': eff_label, 'practical': is_practical,
    })

    p_str = f'{p_val:.2e}' if (not np.isnan(p_val)) else 'N/A'
    r_str = f'{r_rb:>8.4f}' if (not np.isnan(r_rb)) else '     N/A'
    print(f'  {a_label:<18} {b_label:<18} {med_diff:<10.4f} '
          f'{r_str} {eff_label:>10} {p_str:>12} {pract_label:>10}')

# Summary: group models into equivalence classes
print('\n  --- Gruppi di equivalenza pratica ---')
print('  (modelli con differenza NON praticamente significativa)')
# Build adjacency: two models are "equivalent" if their comparison is not practical
equiv_pairs = []
for r in wilcoxon_results:
    if not r['practical']:
        equiv_pairs.append((r['model_a'], r['model_b']))

# Simple clustering via connected components
from collections import defaultdict
adj = defaultdict(set)
for a, b in equiv_pairs:
    adj[a].add(b)
    adj[b].add(a)

visited = set()
groups = []
for m in MODELS:
    if m in visited:
        continue
    group = []
    stack = [m]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        group.append(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                stack.append(neighbor)
    groups.append(sorted(group, key=lambda x: merged_clean[x].median()))

groups.sort(key=lambda g: merged_clean[g[0]].median())
for gi, group in enumerate(groups):
    meds = [f'{m} ({merged_clean[m].median():.4f})' for m in group]
    print(f'  Gruppo {gi+1}: {", ".join(meds)}')

# ---------------------------------------------------------------------------
# 5. Effect size heatmap (rank-biserial r)
# ---------------------------------------------------------------------------
print('\n5. Heatmap effect size + p-value...')

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Panel A: Effect size (rank-biserial r)
ax = axes[0]
# effect_matrix[i,j] = r for "model i vs model j" (positive = i worse than j)
im = ax.imshow(effect_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
ax.set_xticks(range(N_MODELS))
ax.set_yticks(range(N_MODELS))
ax.set_xticklabels(MODELS, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(MODELS, fontsize=9)

for i in range(N_MODELS):
    for j in range(N_MODELS):
        if i == j:
            ax.text(j, i, '—', ha='center', va='center', fontsize=8)
        else:
            r = effect_matrix[i, j]
            abs_r = abs(r)
            if abs_r >= 0.10:
                fw = 'bold'
            else:
                fw = 'normal'
            txt = f'{r:.3f}'
            color = 'white' if abs_r > 0.3 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                    color=color, fontweight=fw)

cbar = fig.colorbar(im, ax=ax, label='rank-biserial r')
ax.set_title('Effect size (rank-biserial r)\nrow - col: positive = row worse', fontsize=12)

# Panel B: p-value
ax = axes[1]
log_pvals = -np.log10(pvalue_matrix + 1e-300)
np.fill_diagonal(log_pvals, 0)

im2 = ax.imshow(log_pvals, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(N_MODELS))
ax.set_yticks(range(N_MODELS))
ax.set_xticklabels(MODELS, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(MODELS, fontsize=9)

for i in range(N_MODELS):
    for j in range(N_MODELS):
        if i == j:
            ax.text(j, i, '—', ha='center', va='center', fontsize=8)
        else:
            p = pvalue_matrix[i, j]
            if p < 0.001:
                txt = f'{p:.1e}'
            else:
                txt = f'{p:.3f}'
            color = 'white' if log_pvals[i, j] > 5 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7, color=color)

cbar2 = fig.colorbar(im2, ax=ax, label='-log10(p-value)')
ax.set_title('Wilcoxon p-value\n(WAPE orario, test, in-stock)', fontsize=12)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig09_effect_size_pvalue.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig09_effect_size_pvalue.png')

# ---------------------------------------------------------------------------
# 6. Bootstrap CI sulla differenza di mediana WAPE
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  6. BOOTSTRAP CI — Differenza mediana WAPE orario')
print('=' * 72)

N_BOOT = 10000

# Find best model (lowest median WAPE)
median_wapes = {label: merged_clean[label].median() for label in MODELS}
best_model = min(median_wapes, key=median_wapes.get)
print(f'\n  Modello migliore (mediana WAPE): {best_model} ({median_wapes[best_model]:.4f})')
print(f'\n  Bootstrap CI (N={N_BOOT}) per la differenza: modello X - {best_model}')

print(f'\n  {"Model":<18} {"med(X)":<10} {"med(diff)":<10} '
      f'{"CI 2.5%":>10} {"CI 97.5%":>10} {"sig":>5}')
print('  ' + '-' * 68)

best_vals = merged_clean[best_model].values

for label in MODELS:
    if label == best_model:
        continue

    other_vals = merged_clean[label].values
    diffs = other_vals - best_vals

    # Bootstrap
    boot_medians = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        idx = np.random.randint(0, len(diffs), size=len(diffs))
        boot_medians[b] = np.median(diffs[idx])

    ci_lo = np.percentile(boot_medians, 2.5)
    ci_hi = np.percentile(boot_medians, 97.5)
    med_diff = np.median(diffs)

    # Significant if CI doesn't contain 0
    sig = '*' if (ci_lo > 0 or ci_hi < 0) else 'ns'

    print(f'  {label:<18} {median_wapes[label]:<10.4f} {med_diff:<10.4f} '
          f'{ci_lo:>10.4f} {ci_hi:>10.4f} {sig:>5}')

# ---------------------------------------------------------------------------
# 7. Ranking plot
# ---------------------------------------------------------------------------
print('\n7. Ranking plot...')

fig, ax = plt.subplots(figsize=(12, 6))

sorted_models = sorted(MODELS, key=lambda m: median_wapes[m])
sorted_medians = [median_wapes[m] for m in sorted_models]
sorted_colors = [COLORS[MODELS.index(m)] for m in sorted_models]

bars = ax.barh(range(len(sorted_models)), sorted_medians, color=sorted_colors, alpha=0.8)

# Annotate
for i, (med, label) in enumerate(zip(sorted_medians, sorted_models)):
    ax.text(med + 0.002, i, f'{med:.4f}', va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(sorted_models)))
ax.set_yticklabels(sorted_models, fontsize=11)
ax.set_xlabel('Median WAPE (hourly, in-stock)', fontsize=12)
ax.set_title('Fase A — Ranking modelli baseline (test)', fontsize=14)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Add vertical line at best
ax.axvline(sorted_medians[0], color='red', linestyle='--', alpha=0.5, linewidth=1)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig10_ranking.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig10_ranking.png')

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  8. CONCLUSIONE')
print('=' * 72)

# Count practical wins for best model
n_practical_wins = sum(1 for r in wilcoxon_results
                       if r['practical'] and (
                           (r['model_a'] == best_model and r['med_diff'] < 0) or
                           (r['model_b'] == best_model and r['med_diff'] > 0)))

print(f'\n  Miglior modello (WAPE mediana): {best_model} ({median_wapes[best_model]:.4f})')
print(f'  Praticamente migliore di {n_practical_wins}/{N_MODELS-1} altri modelli')
print(f'    (Wilcoxon p<0.05 AND |r|>={EFFECT_THRESH} AND |Δmed|>={DELTA_THRESH})')

print(f'\n  Gruppi di equivalenza pratica:')
for gi, group in enumerate(groups):
    meds = [f'{m} ({merged_clean[m].median():.4f})' for m in group]
    print(f'    Gruppo {gi+1}: {", ".join(meds)}')

print(f'\n  Nota: con N={n_series:,} serie, quasi tutti i confronti sono')
print(f'  statisticamente significativi (p≈0). L\'effect size (rank-biserial r)')
print(f'  e la soglia di rilevanza pratica (|Δmed|>={DELTA_THRESH}) permettono')
print(f'  di distinguere differenze reali da artefatti della potenza statistica.')

print('\n' + '=' * 72)
print('  DONE — 04c_comparison.py')
print('=' * 72)
