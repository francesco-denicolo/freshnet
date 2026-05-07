"""
09_analysis.py — Fase C: Analisi matrice imputer × forecaster (ore 6-22)
=========================================================================
Costruisce la matrice completa, heatmap, boxplot, test statistici.

Eseguire con: freshnet/bin/python notebooks_622/09_analysis.py
"""
import sys, os, functools, numpy as np, pandas as pd
from scipy import stats
from itertools import combinations
from collections import defaultdict

print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SEED = 42; np.random.seed(SEED)

# ===========================================================================
# 1. Load all per-series results
# ===========================================================================
print('=' * 72)
print('  FASE C — ANALISI MATRICE (ore 6-22)')
print('=' * 72)

# Define the full matrix structure
IMPUTERS_ORDER = ['No imputation', 'Media condizionata', 'Media globale',
                   'Mediana condizionata', 'Mediana globale',
                   'LGB imputer', 'DLinear',
                   'Forward Fill', 'Seasonal Naive', 'Linear Interp', 'SAITS']
FORECASTERS_ORDER = ['Global Mean', 'DoW Mean', 'MA (K=21)',
                      'LGB (no lags)', 'LGB (M5 lags)',
                      'MLP (no lags)', 'MLP (M5 lags)',
                      'Chronos-bolt']

# Map (imputer, forecaster) -> parquet filename
FC_FILE_MAP = {
    'Global Mean': 'global_mean', 'DoW Mean': 'dow_mean',
    'MA (K=21)': 'ma_k21', 'Naive Direct': 'naive_direct',
    'LGB (no lags)': 'lgb_nolags', 'LGB (M5 lags)': 'lgb_m5lags',
    'MLP (no lags)': 'mlp_nolags', 'MLP (M5 lags)': 'mlp_m5lags',
    'Chronos-bolt': 'chronos_bolt',
}
IMP_FILE_MAP = {
    'Media condizionata': 'media_cond', 'Media globale': 'media_glob',
    'Mediana condizionata': 'mediana_cond', 'Mediana globale': 'mediana_glob',
    'LGB imputer': 'lgb',
    'DLinear': 'dlinear',
    'Forward Fill': 'forward_fill', 'Seasonal Naive': 'seasonal_naive',
    'Linear Interp': 'linear_interp', 'SAITS': 'saits',
}

def get_parquet_path(imp, fc):
    fc_safe = FC_FILE_MAP[fc]
    # Chronos-bolt always uses the {imp}__{fc} pattern (even for no_imp)
    if fc == 'Chronos-bolt':
        imp_safe = 'no_imp' if imp == 'No imputation' else IMP_FILE_MAP[imp]
        return os.path.join(RESULTS_DIR, f'{imp_safe}__{fc_safe}_test_per_series.parquet')
    if imp == 'No imputation':
        if fc in ['Global Mean','DoW Mean','MA (K=21)','Naive Direct']:
            return os.path.join(RESULTS_DIR, f'naive_{fc_safe}_test_per_series.parquet')
        else:
            return os.path.join(RESULTS_DIR, f'{fc_safe}_test_per_series.parquet')
    else:
        imp_safe = IMP_FILE_MAP[imp]
        if fc in ['LGB (no lags)', 'MLP (no lags)']:
            return os.path.join(RESULTS_DIR, f'{fc_safe}_test_per_series.parquet')
        elif fc in ['Global Mean','DoW Mean','MA (K=21)']:
            return os.path.join(RESULTS_DIR, f'{imp_safe}__{fc_safe}_test_per_series.parquet')
        else:
            return os.path.join(RESULTS_DIR, f'{imp_safe}__{fc_safe}_test_per_series.parquet')

print('\n1. Loading results...')
matrix_data = {}  # (imp, fc) -> ps_df
matrix_pooled = {}  # (imp, fc) -> {'hourly_wape': ..., 'hourly_wpe': ...}

for imp in IMPUTERS_ORDER:
    for fc in FORECASTERS_ORDER:
        path = get_parquet_path(imp, fc)
        if not os.path.exists(path):
            print(f'  MISSING: {imp} × {fc} -> {path}')
            continue
        ps = pd.read_parquet(path)
        matrix_data[(imp, fc)] = ps
        # Compute pooled from per-series (approximate via median)
        # Chronos-bolt doesn't produce daily metrics (predict_length spans 7 days flat)
        cols = [c for c in ['hourly_wape','hourly_wpe','daily_wape','daily_wpe'] if c in ps.columns]
        med = {c: ps[c].dropna().median() for c in cols}
        matrix_pooled[(imp, fc)] = med

print(f'  Loaded {len(matrix_data)} cells')

# ===========================================================================
# 2. Print full matrix
# ===========================================================================
print('\n' + '=' * 72)
print('  2. MATRICE WAPE_h MEDIANA (test, in-stock, ore 6-22)')
print('=' * 72)

header = f'  {"Imputer":<24}'
for fc in FORECASTERS_ORDER:
    header += f' {fc:>12}'
print(header)
print('  ' + '-' * (24 + 13 * len(FORECASTERS_ORDER)))

for imp in IMPUTERS_ORDER:
    row = f'  {imp:<24}'
    for fc in FORECASTERS_ORDER:
        key = (imp, fc)
        if key in matrix_pooled:
            v = matrix_pooled[key]['hourly_wape']
            # Mark best in row
            row += f' {v:>12.4f}'
        else:
            row += f' {"—":>12}'
    print(row)

# Also print WPE matrix
print('\n' + '=' * 72)
print('  3. MATRICE WPE_h MEDIANA (test, in-stock, ore 6-22)')
print('=' * 72)

header = f'  {"Imputer":<24}'
for fc in FORECASTERS_ORDER:
    header += f' {fc:>12}'
print(header)
print('  ' + '-' * (24 + 13 * len(FORECASTERS_ORDER)))

for imp in IMPUTERS_ORDER:
    row = f'  {imp:<24}'
    for fc in FORECASTERS_ORDER:
        key = (imp, fc)
        if key in matrix_pooled:
            v = matrix_pooled[key]['hourly_wpe']
            row += f' {v:>12.4f}'
        else:
            row += f' {"—":>12}'
    print(row)

# ===========================================================================
# 3. Heatmaps
# ===========================================================================
print('\n4. Generazione heatmap...')

def make_heatmap(metric, title, figname, cmap='RdYlGn_r', fmt='.4f', vmin=None, vmax=None,
                  mark_best=True, diverging=False):
    n_imp = len(IMPUTERS_ORDER)
    n_fc = len(FORECASTERS_ORDER)
    data = np.full((n_imp, n_fc), np.nan)

    for i, imp in enumerate(IMPUTERS_ORDER):
        for j, fc in enumerate(FORECASTERS_ORDER):
            if (imp, fc) in matrix_pooled:
                data[i, j] = matrix_pooled[(imp, fc)][metric]

    if vmin is None: vmin = np.nanmin(data) - 0.005
    if vmax is None: vmax = np.nanmax(data) + 0.005

    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_fc))
    ax.set_yticks(range(n_imp))
    ax.set_xticklabels(FORECASTERS_ORDER, rotation=30, ha='right', fontsize=11)
    ax.set_yticklabels(IMPUTERS_ORDER, fontsize=11)

    best_val = np.nanmin(data) if mark_best else None

    for i in range(n_imp):
        for j in range(n_fc):
            v = data[i, j]
            if np.isnan(v):
                continue
            is_best = mark_best and abs(v - best_val) < 1e-5
            fw = 'bold' if is_best else 'normal'
            fs = 11 if is_best else 10
            if diverging:
                norm_v = abs(v) / max(abs(vmin), abs(vmax))
                color = 'white' if norm_v > 0.5 else 'black'
            else:
                norm_v = (v - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-10)
                color = 'white' if norm_v > 0.6 else 'black'
            txt = f'{v:{fmt}}'
            if is_best:
                txt += ' ★'
            ax.text(j, i, txt, ha='center', va='center', fontsize=fs,
                    fontweight=fw, color=color)

    # Gridlines
    for i in range(n_imp + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1.5)
    for j in range(n_fc + 1):
        ax.axvline(j - 0.5, color='white', linewidth=1.5)
    # Separator between naive and ML forecasters
    ax.axvline(2.5, color='black', linewidth=2.5)

    cbar = fig.colorbar(im, ax=ax, label=metric, shrink=0.8)
    ax.set_title(title, fontsize=13, pad=15)
    ax.set_xlabel('Forecaster', fontsize=12)
    ax.set_ylabel('Imputer', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, figname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Salvata: {figname}')

make_heatmap('hourly_wape',
             'Matrice Imputer × Forecaster — WAPE orario mediana (test, ore 6-22, in-stock)',
             'fig01_heatmap_wape_median.png')

# WPE: diverging colormap centered at 0
vlim_wpe = 0.4
make_heatmap('hourly_wpe',
             'Matrice Imputer × Forecaster — WPE orario mediana (test, ore 6-22, in-stock)',
             'fig02_heatmap_wpe_median.png',
             cmap='RdBu', vmin=-vlim_wpe, vmax=0.1,
             mark_best=False, diverging=True)

# ===========================================================================
# 4. Boxplot per forecaster (effect of imputation)
# ===========================================================================
print('\n5. Boxplot per forecaster...')

COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#DD8452']

for fc in ['MLP (M5 lags)', 'LGB (M5 lags)', 'DoW Mean', 'Chronos-bolt']:
    fig, ax = plt.subplots(figsize=(12, 6))
    box_data, box_labels, medians = [], [], []

    for imp in IMPUTERS_ORDER:
        if (imp, fc) not in matrix_data:
            continue
        ps = matrix_data[(imp, fc)]
        vals = ps['hourly_wape'].dropna()
        box_data.append(vals.clip(upper=vals.quantile(0.99)).values)
        box_labels.append(imp)
        medians.append(vals.median())

    if not box_data:
        plt.close(fig); continue
    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.6, showfliers=False)
    for patch, color in zip(bp['boxes'], COLORS[:len(box_data)]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    for ml in bp['medians']:
        ml.set_color('red'); ml.set_linewidth(2)

    for i, med in enumerate(medians):
        ax.text(i+1, med, f' {med:.4f}', ha='left', va='center', fontsize=9,
                fontweight='bold', color='red')

    ax.set_ylabel('WAPE (hourly, in-stock)', fontsize=12)
    ax.set_title(f'{fc} — Effetto dell\'imputation (ore 6-22)', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fc_safe = fc.lower().replace(' ','_').replace('(','').replace(')','')
    fig.savefig(os.path.join(FIG_DIR, f'fig03_boxplot_{fc_safe}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Salvata: fig03_boxplot_{fc_safe}.png')

# ===========================================================================
# 5. Wilcoxon + effect size for ALL forecasters
# ===========================================================================
print('\n' + '=' * 72)
print('  6. TEST STATISTICI — Per ogni forecaster, confronto imputer pairwise')
print('=' * 72)

# Forecasters where imputation matters (exclude no-lags which are invariant)
FC_TO_TEST = ['Global Mean', 'DoW Mean', 'MA (K=21)', 'LGB (M5 lags)', 'MLP (M5 lags)', 'Chronos-bolt']

all_wilcoxon = []  # collect all results for summary

for fc_test in FC_TO_TEST:
    print(f'\n  {"="*60}')
    print(f'  Forecaster: {fc_test}')
    print(f'  {"="*60}')

    # Merge all imputers on (store_id, product_id)
    merged = None
    for imp in IMPUTERS_ORDER:
        if (imp, fc_test) not in matrix_data:
            continue
        ps = matrix_data[(imp, fc_test)][['store_id','product_id','hourly_wape']].copy()
        ps = ps.rename(columns={'hourly_wape': imp})
        if merged is None:
            merged = ps
        else:
            merged = merged.merge(ps, on=['store_id','product_id'], how='inner')

    merged_clean = merged.dropna()
    print(f'  Serie allineate: {len(merged_clean):,}')

    imp_list = [imp for imp in IMPUTERS_ORDER if imp in merged_clean.columns]

    print(f'\n  {"Imputer A":<24} {"Imputer B":<24} {"Δmed":>8} {"r_rb":>8} '
          f'{"effect":>10} {"% A wins":>9} {"p-value":>12}')
    print('  ' + '-' * 100)

    for i, j in combinations(range(len(imp_list)), 2):
        a, b = imp_list[i], imp_list[j]
        av, bv = merged_clean[a].values, merged_clean[b].values
        diff = av - bv
        nz = diff != 0
        if nz.sum() < 10:
            continue
        res = stats.wilcoxon(av[nz], bv[nz], alternative='two-sided')
        w, p = res.statistic, res.pvalue
        n_nz = int(nz.sum())
        r_rb = 1.0 - (2.0 * w) / (n_nz * (n_nz + 1) / 2)

        med_diff = np.median(diff)
        abs_r = abs(r_rb)
        eff = 'large' if abs_r >= 0.5 else ('medium' if abs_r >= 0.3 else (
              'small' if abs_r >= 0.1 else 'negligible'))
        pct_a_wins = 100.0 * (1 - r_rb) / 2

        all_wilcoxon.append({
            'forecaster': fc_test, 'a': a, 'b': b,
            'med_diff': med_diff, 'r_rb': r_rb, 'effect': eff,
            'pct_a_wins': pct_a_wins, 'p': p})

        p_str = f'{p:.2e}' if p < 0.001 else f'{p:.4f}'
        print(f'  {a:<24} {b:<24} {med_diff:>8.4f} {r_rb:>8.4f} '
              f'{eff:>10} {pct_a_wins:>8.1f}% {p_str:>12}')

# Summary: for each forecaster, best imputer and effect vs no-imputation
print('\n' + '=' * 72)
print('  7. RIEPILOGO: No imputation vs Best imputer per forecaster')
print('=' * 72)

print(f'\n  {"Forecaster":<20} {"Best imputer":<24} {"Δmed":>8} {"r_rb":>8} '
      f'{"effect":>10} {"% best wins":>11}')
print('  ' + '-' * 85)

for fc_test in FC_TO_TEST:
    # Find the comparison "No imputation" vs each other imputer
    comparisons = [r for r in all_wilcoxon
                   if r['forecaster'] == fc_test and r['a'] == 'No imputation']
    if not comparisons:
        continue

    # Best imputer = the one with most negative Δmed (lowest WAPE improvement)
    best = min(comparisons, key=lambda r: r['med_diff'])
    if best['med_diff'] >= 0:
        # No imputer improves over no-imputation
        print(f'  {fc_test:<20} {"(nessuno)":<24} {"—":>8} {"—":>8} '
              f'{"—":>10} {"—":>11}')
    else:
        pct_best = 100 - best['pct_a_wins']  # flip because A = no-imp
        print(f'  {fc_test:<20} {best["b"]:<24} {best["med_diff"]:>8.4f} '
              f'{best["r_rb"]:>8.4f} {best["effect"]:>10} {pct_best:>10.1f}%')

# Also find worst (most harmed by imputation)
print(f'\n  {"Forecaster":<20} {"Worst imputer":<24} {"Δmed":>8} {"r_rb":>8} '
      f'{"effect":>10} {"% worst wins":>12}')
print('  ' + '-' * 88)

for fc_test in FC_TO_TEST:
    comparisons = [r for r in all_wilcoxon
                   if r['forecaster'] == fc_test and r['a'] == 'No imputation']
    if not comparisons:
        continue
    worst = max(comparisons, key=lambda r: r['med_diff'])
    if worst['med_diff'] <= 0:
        print(f'  {fc_test:<20} {"(nessuno peggiora)":<24}')
    else:
        pct_worst = worst['pct_a_wins']  # A = no-imp wins
        print(f'  {fc_test:<20} {worst["b"]:<24} {worst["med_diff"]:>8.4f} '
              f'{worst["r_rb"]:>8.4f} {worst["effect"]:>10} {pct_worst:>11.1f}%')

# Save all wilcoxon results
wilcoxon_df = pd.DataFrame(all_wilcoxon)
wilcoxon_df.to_parquet(os.path.join(RESULTS_DIR, 'wilcoxon_all.parquet'), index=False)
print(f'\n  Salvato: wilcoxon_all.parquet ({len(wilcoxon_df)} confronti)')

# Keep wilcoxon_results for backward compat (effect size heatmap uses it)
wilcoxon_results = [r for r in all_wilcoxon if r['forecaster'] == 'MLP (M5 lags)']
imp_list = [imp for imp in IMPUTERS_ORDER]

# ===========================================================================
# 8. Effect size heatmap — one per forecaster
# ===========================================================================
print('\n8. Effect size heatmaps...')

for fc_plot in FC_TO_TEST:
    fc_results = [r for r in all_wilcoxon if r['forecaster'] == fc_plot]
    if not fc_results:
        continue

    n_imp = len(IMPUTERS_ORDER)
    effect_mat = np.zeros((n_imp, n_imp))
    for r in fc_results:
        if r['a'] in IMPUTERS_ORDER and r['b'] in IMPUTERS_ORDER:
            i, j = IMPUTERS_ORDER.index(r['a']), IMPUTERS_ORDER.index(r['b'])
            effect_mat[i, j] = r['r_rb']
            effect_mat[j, i] = -r['r_rb']

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(effect_mat, cmap='RdBu_r', aspect='auto', vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(n_imp)); ax.set_yticks(range(n_imp))
    ax.set_xticklabels(IMPUTERS_ORDER, rotation=35, ha='right', fontsize=10)
    ax.set_yticklabels(IMPUTERS_ORDER, fontsize=10)

    for i in range(n_imp):
        for j in range(n_imp):
            if i == j:
                ax.text(j, i, '—', ha='center', va='center', fontsize=9)
            else:
                rv = effect_mat[i, j]
                fw = 'bold' if abs(rv) >= 0.1 else 'normal'
                color = 'white' if abs(rv) > 0.3 else 'black'
                ax.text(j, i, f'{rv:.3f}', ha='center', va='center', fontsize=9,
                        fontweight=fw, color=color)

    fig.colorbar(im, ax=ax, label='rank-biserial r')
    ax.set_title(f'Effect size tra imputer — {fc_plot}\n(row - col, positive = row peggiore)', fontsize=12)
    fig.tight_layout()
    fc_safe = fc_plot.lower().replace(' ','_').replace('(','').replace(')','').replace('=','')
    fig.savefig(os.path.join(FIG_DIR, f'fig04_effect_{fc_safe}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Salvata: fig04_effect_{fc_safe}.png')

# ===========================================================================
# 7. Saturation curve
# ===========================================================================
print('\n8. Curva di saturazione...')

# Load Traccia A results
traccia_a = pd.read_parquet(os.path.join(RESULTS_DIR, 'traccia_a.parquet'))
# Append SOTA imputer results if available
for sota in ['dlinear', 'saits', 'timesnet', 'itransformer']:
    ta_path = os.path.join(RESULTS_DIR, f'traccia_a_{sota}.parquet')
    if os.path.exists(ta_path):
        traccia_a = pd.concat([traccia_a, pd.read_parquet(ta_path)], ignore_index=True)
imp_wape_recovery = dict(zip(traccia_a['imputer'], traccia_a['wape_recovery']))
# Add "no imputation" as worst possible recovery
imp_wape_recovery['No imputation'] = 1.0  # S_obs = no recovery

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

for ax_idx, (fc, ax) in enumerate(zip(['MLP (M5 lags)', 'LGB (M5 lags)', 'Chronos-bolt'], axes)):
    x_vals, y_vals, labels = [], [], []
    for imp in IMPUTERS_ORDER:
        if (imp, fc) not in matrix_pooled:
            continue
        imp_label = imp
        if imp_label in imp_wape_recovery:
            xv = imp_wape_recovery[imp_label]
        elif imp == 'No imputation':
            xv = 1.0
        else:
            continue
        yv = matrix_pooled[(imp, fc)]['hourly_wape']
        x_vals.append(xv); y_vals.append(yv); labels.append(imp)

    # Cycle colors if we have more points than COLORS
    pt_colors = [COLORS[i % len(COLORS)] for i in range(len(x_vals))]
    ax.scatter(x_vals, y_vals, s=100, zorder=5, color=pt_colors)
    for xi, yi, li in zip(x_vals, y_vals, labels):
        ax.annotate(li, (xi, yi), textcoords='offset points',
                    xytext=(5, 8), fontsize=8, ha='left')

    ax.set_xlabel('WAPE_recovery imputer (Traccia A)', fontsize=11)
    ax.set_ylabel(f'WAPE_h mediana — {fc}', fontsize=11)
    ax.set_title(f'Saturazione: {fc}', fontsize=12)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()  # Better imputer on the left

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig05_saturation.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig05_saturation.png')

# ===========================================================================
# 8. Robustness: ΔWAPE per forecaster
# ===========================================================================
print('\n' + '=' * 72)
print('  9. ROBUSTEZZA AL CENSORING')
print('=' * 72)

print(f'\n  ΔWAPE = WAPE(no imp) - WAPE(best imp)')
print(f'  Il forecaster con ΔWAPE più piccolo è il più robusto.\n')

print(f'  {"Forecaster":<20} {"WAPE no-imp":>12} {"Best imp":>20} {"WAPE best":>12} {"ΔWAPE":>8}')
print('  ' + '-' * 76)

for fc in FORECASTERS_ORDER:
    if ('No imputation', fc) not in matrix_pooled:
        continue
    wape_noimp = matrix_pooled[('No imputation', fc)]['hourly_wape']

    best_imp, best_wape = None, wape_noimp
    for imp in IMPUTERS_ORDER[1:]:  # skip "No imputation"
        if (imp, fc) not in matrix_pooled:
            continue
        w = matrix_pooled[(imp, fc)]['hourly_wape']
        if w < best_wape:
            best_wape, best_imp = w, imp

    if best_imp is None:
        delta = 0.0
        best_imp = '(none)'
    else:
        delta = wape_noimp - best_wape

    print(f'  {fc:<20} {wape_noimp:>12.4f} {best_imp:>20} {best_wape:>12.4f} {delta:>8.4f}')

# ===========================================================================
# 9. Best model vs all others
# ===========================================================================
print('\n' + '=' * 72)
print('  10. MIGLIOR COMBINAZIONE vs TUTTE LE ALTRE')
print('=' * 72)

# Select the best imputer for each forecaster
best_combos = {}
for fc in FORECASTERS_ORDER:
    cells = [(imp, matrix_pooled[(imp, fc)]['hourly_wape'])
             for imp in IMPUTERS_ORDER if (imp, fc) in matrix_pooled]
    if cells:
        best_imp, best_wape = min(cells, key=lambda x: x[1])
        best_combos[fc] = {'imputer': best_imp, 'wape_med': best_wape}

combo_labels = []
for fc in FORECASTERS_ORDER:
    if fc not in best_combos:
        continue
    bc = best_combos[fc]
    label = f'{fc}' if bc['imputer'] == 'No imputation' else f'{fc} + {bc["imputer"]}'
    combo_labels.append((label, bc['imputer'], fc, bc['wape_med']))

combo_labels.sort(key=lambda x: x[3])
best_label, best_imp, best_fc, best_wape = combo_labels[0]

print(f'\n  Miglior combinazione: {best_label} (WAPE_med={best_wape:.4f})')
print(f'\n  Confronto con le altre combinazioni (best imputer per forecaster):')

# Build merged dataframe
merged_combos = None
for label, imp, fc, _ in combo_labels:
    ps = matrix_data[(imp, fc)][['store_id', 'product_id', 'hourly_wape']].copy()
    ps = ps.rename(columns={'hourly_wape': label})
    if merged_combos is None:
        merged_combos = ps
    else:
        merged_combos = merged_combos.merge(ps, on=['store_id', 'product_id'], how='inner')

merged_combos_clean = merged_combos.dropna()
n_ser = len(merged_combos_clean)
print(f'  Serie allineate: {n_ser:,}')

labels_only = [x[0] for x in combo_labels]

print(f'\n  {"Combinazione":<45} {"WAPE med":>9} {"Δmed":>8} {"r_rb":>8} {"effect":>10} {"% best wins":>12}')
print('  ' + '-' * 96)

combo_wilcoxon = []
best_vals = merged_combos_clean[best_label].values

for label, imp, fc, wape_med in combo_labels:
    if label == best_label:
        print(f'  {label:<45} {wape_med:>9.4f}     —        —          —            —')
        continue

    other_vals = merged_combos_clean[label].values
    diff = best_vals - other_vals  # negative = best wins
    nz = diff != 0
    res = stats.wilcoxon(best_vals[nz], other_vals[nz], alternative='two-sided')
    w, p = res.statistic, res.pvalue
    n_nz = int(nz.sum())
    r_rb = 1.0 - (2.0 * w) / (n_nz * (n_nz + 1) / 2)

    med_diff = np.median(diff)
    abs_r = abs(r_rb)
    eff = 'large' if abs_r >= 0.5 else ('medium' if abs_r >= 0.3 else (
          'small' if abs_r >= 0.1 else 'negligible'))
    # diff = best - other. Best has lower WAPE, so diff < 0.
    # Wilcoxon on (best, other): r_rb > 0 means best tends to have lower values = best wins.
    # pct_best_wins = (1 + r_rb) / 2
    pct_best_wins = 100.0 * (1 + r_rb) / 2

    combo_wilcoxon.append({'other': label, 'med_diff': med_diff, 'r_rb': r_rb,
                            'effect': eff, 'pct_best_wins': pct_best_wins, 'p': p})

    print(f'  {label:<45} {wape_med:>9.4f} {med_diff:>8.4f} {r_rb:>8.4f} {eff:>10} {pct_best_wins:>11.1f}%')

# Save
combo_df = pd.DataFrame(combo_wilcoxon)
combo_df.to_parquet(os.path.join(RESULTS_DIR, 'wilcoxon_best_vs_all.parquet'), index=False)

# Bar chart: best model vs others
print('\n  Generazione grafico...')
others = [x for x in combo_labels if x[0] != best_label]
fig, ax = plt.subplots(figsize=(12, 6))

y_pos = np.arange(len(others))
deltas = [-r['med_diff'] for r in combo_wilcoxon]  # positive = best wins by this much
effects = [r['r_rb'] for r in combo_wilcoxon]
other_labels = [r['other'] for r in combo_wilcoxon]
pct_wins = [r['pct_best_wins'] for r in combo_wilcoxon]

bar_colors = [COLORS[i % len(COLORS)] for i in range(len(others))]
bars = ax.barh(y_pos, deltas, color=bar_colors, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(other_labels, fontsize=10)
ax.set_xlabel('ΔWAPE mediana (positivo = miglior modello vince)', fontsize=11)
ax.set_title(f'Vantaggio di {best_label}\nrispetto alle altre combinazioni', fontsize=13)
ax.invert_yaxis()

for i, (d, pw, eff) in enumerate(zip(deltas, pct_wins, [r['effect'] for r in combo_wilcoxon])):
    ax.text(d + 0.001, i, f'Δ={d:.3f}, r={effects[i]:.2f}, {pw:.0f}% wins ({eff})',
            va='center', fontsize=9)

ax.axvline(0, color='black', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig06_best_vs_all.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig06_best_vs_all.png')

# ===========================================================================
# 10. Summary tables
# ===========================================================================
print('\n' + '=' * 72)
print('  11. CONCLUSIONI')
print('=' * 72)

# --- 11a. Full matrix table (WAPE median) ---
print('\n11a. MATRICE COMPLETA — WAPE mediana per-serie (imputer × forecaster)')
print('-' * 100)
mat_df = pd.DataFrame(
    [[matrix_pooled.get((imp,fc), {}).get('hourly_wape', np.nan)
      for fc in FORECASTERS_ORDER] for imp in IMPUTERS_ORDER],
    index=IMPUTERS_ORDER, columns=FORECASTERS_ORDER)
print(mat_df.round(4).to_string())

print('\n11b. MATRICE COMPLETA — WPE mediana per-serie (imputer × forecaster)')
print('-' * 100)
mat_wpe = pd.DataFrame(
    [[matrix_pooled.get((imp,fc), {}).get('hourly_wpe', np.nan)
      for fc in FORECASTERS_ORDER] for imp in IMPUTERS_ORDER],
    index=IMPUTERS_ORDER, columns=FORECASTERS_ORDER)
print(mat_wpe.round(4).to_string())

# --- 11c. Ranked best combinations ---
print('\n11c. COMBINAZIONI ORDINATE PER WAPE MEDIANA (ascendente)')
print('-' * 80)
ranked = []
for (imp, fc), met in matrix_pooled.items():
    w = met.get('hourly_wape', np.nan)
    wpe = met.get('hourly_wpe', np.nan)
    if not np.isnan(w):
        ranked.append((imp, fc, w, wpe))
ranked.sort(key=lambda x: x[2])

df_ranked = pd.DataFrame(ranked, columns=['imputer','forecaster','wape_med','wpe_med'])
df_ranked['rank'] = range(1, len(df_ranked)+1)
df_ranked = df_ranked[['rank','imputer','forecaster','wape_med','wpe_med']]
df_ranked.to_parquet(os.path.join(RESULTS_DIR, 'ranked_combinations.parquet'), index=False)

print(f'\n  {"Rank":<5}{"Imputer":<24}{"Forecaster":<20}{"WAPE med":>10}{"WPE med":>10}')
print('  ' + '-'*69)
for r, (imp, fc, w, wpe) in enumerate(ranked, 1):
    mark = ' ★' if r == 1 else ''
    print(f'  {r:<5}{imp:<24}{fc:<20}{w:>10.4f}{wpe:>10.4f}{mark}')

# Find overall best cell
best_key = min(matrix_pooled, key=lambda k: matrix_pooled[k]['hourly_wape'])
best_v = matrix_pooled[best_key]['hourly_wape']
print(f'\n  Miglior cella (WAPE mediana): {best_key[0]} × {best_key[1]} = {best_v:.4f}')

# Find best per forecaster
print(f'\n  Miglior imputer per forecaster:')
for fc in FORECASTERS_ORDER:
    cells = [(imp, matrix_pooled[(imp,fc)]['hourly_wape'])
             for imp in IMPUTERS_ORDER if (imp,fc) in matrix_pooled]
    if cells:
        best = min(cells, key=lambda x: x[1])
        print(f'    {fc:<20} → {best[0]} ({best[1]:.4f})')

# Find best forecaster per imputer
print(f'\n  Miglior forecaster per imputer:')
for imp in IMPUTERS_ORDER:
    cells = [(fc, matrix_pooled[(imp,fc)]['hourly_wape'])
             for fc in FORECASTERS_ORDER if (imp,fc) in matrix_pooled]
    if cells:
        best = min(cells, key=lambda x: x[1])
        print(f'    {imp:<24} → {best[0]} ({best[1]:.4f})')

print('\n' + '=' * 72)
print('  DONE — 09_analysis.py')
print('=' * 72)
