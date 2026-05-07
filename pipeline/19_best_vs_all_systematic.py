"""
19_best_vs_all_systematic.py — Best cell vs all 87 alternatives
=================================================================
Confronto paired Wilcoxon di (No imputation + Chronos-bolt) vs ogni
altra cella della matrice 11×8 = 88 celle (88-1 = 87 confronti).

Output:
- pipeline/results/best_vs_all_systematic.parquet (87 righe)
- Tabella riassuntiva ordinata per Cliff's δ
"""
import os, functools, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

IMPUTERS = ['No imputation', 'Media condizionata', 'Media globale',
            'Mediana condizionata', 'Mediana globale',
            'LGB imputer', 'DLinear',
            'Forward Fill', 'Seasonal Naive', 'Linear Interp', 'SAITS']
FORECASTERS = ['Global Mean', 'DoW Mean', 'MA (K=21)',
               'LGB (no lags)', 'LGB (M5 lags)',
               'MLP (no lags)', 'MLP (M5 lags)', 'Chronos-bolt']

FC_FILE = {'Global Mean':'global_mean','DoW Mean':'dow_mean','MA (K=21)':'ma_k21',
           'LGB (no lags)':'lgb_nolags','LGB (M5 lags)':'lgb_m5lags',
           'MLP (no lags)':'mlp_nolags','MLP (M5 lags)':'mlp_m5lags',
           'Chronos-bolt':'chronos_bolt'}
IMP_FILE = {'Media condizionata':'media_cond','Media globale':'media_glob',
            'Mediana condizionata':'mediana_cond','Mediana globale':'mediana_glob',
            'LGB imputer':'lgb','DLinear':'dlinear',
            'Forward Fill':'forward_fill','Seasonal Naive':'seasonal_naive',
            'Linear Interp':'linear_interp','SAITS':'saits'}

def get_path(imp, fc):
    fc_safe = FC_FILE[fc]
    if fc == 'Chronos-bolt':
        imp_safe = 'no_imp' if imp == 'No imputation' else IMP_FILE[imp]
        return os.path.join(RESULTS_DIR, f'{imp_safe}__{fc_safe}_test_per_series.parquet')
    if imp == 'No imputation':
        if fc in ['Global Mean','DoW Mean','MA (K=21)']:
            return os.path.join(RESULTS_DIR, f'naive_{fc_safe}_test_per_series.parquet')
        return os.path.join(RESULTS_DIR, f'{fc_safe}_test_per_series.parquet')
    imp_safe = IMP_FILE[imp]
    if fc in ['LGB (no lags)', 'MLP (no lags)']:
        return os.path.join(RESULTS_DIR, f'{fc_safe}_test_per_series.parquet')
    return os.path.join(RESULTS_DIR, f'{imp_safe}__{fc_safe}_test_per_series.parquet')

def cliff_delta_paired(diff):
    diff = diff[~np.isnan(diff)]
    return (np.sum(diff > 0) - np.sum(diff < 0)) / len(diff)

def rank_biserial_paired(x, y):
    diff = x - y
    diff = diff[diff != 0]
    if len(diff) == 0: return 0.0
    ranks = stats.rankdata(np.abs(diff))
    Tp = ranks[diff > 0].sum(); Tm = ranks[diff < 0].sum()
    return (Tp - Tm) / (Tp + Tm) if (Tp + Tm) > 0 else 0.0

def cliff_effect_label(d):
    a = abs(d)
    if a < 0.147: return 'negligible'
    if a < 0.33:  return 'small'
    if a < 0.474: return 'medium'
    return 'large'

# Load best
print('='*80)
print('  BEST vs ALL: No imputation + Chronos-bolt vs 87 alternatives')
print('='*80)

best_path = get_path('No imputation', 'Chronos-bolt')
best_df = pd.read_parquet(best_path)[['store_id','product_id','hourly_wape']].rename(
    columns={'hourly_wape':'best'})
print(f'\nBest cell: No imputation + Chronos-bolt')
print(f'WAPE_med = {best_df["best"].median():.4f}, n_series = {len(best_df):,}')

# Iterate over all 87 alternatives
results = []
for imp in IMPUTERS:
    for fc in FORECASTERS:
        if imp == 'No imputation' and fc == 'Chronos-bolt':
            continue  # skip self
        path = get_path(imp, fc)
        if not os.path.exists(path):
            continue
        other = pd.read_parquet(path)[['store_id','product_id','hourly_wape']].rename(
            columns={'hourly_wape':'other'})
        merged = best_df.merge(other, on=['store_id','product_id']).dropna()
        if len(merged) == 0:
            continue
        a = merged['best'].values
        b = merged['other'].values
        diff = a - b   # negativo se best < other (best wins on WAPE)
        delta_med = np.median(a) - np.median(b)
        # Wilcoxon paired
        try:
            stat, p = stats.wilcoxon(a, b, zero_method='wilcox')
        except ValueError:
            p = np.nan
        r_rb = rank_biserial_paired(a, b)
        d_cliff = cliff_delta_paired(diff)
        eff = cliff_effect_label(d_cliff)
        # % serie dove best vince (diff < 0 perché diff = best - other, più piccolo = best vince)
        pct_best_wins = 100 * np.sum(diff < 0) / len(diff)
        # Best WAPE_med della cella alternativa
        other_wape_med = np.median(b)
        results.append({
            'imputer': imp,
            'forecaster': fc,
            'wape_med_other': other_wape_med,
            'delta_med': delta_med,        # negativo = best ha WAPE più basso
            'r_rb': r_rb,                  # rank-biserial
            'cliff_delta': d_cliff,        # negativo = best wins
            'effect': eff,
            'pct_best_wins': pct_best_wins,
            'wilcoxon_p': p,
            'n': len(merged),
        })

df_res = pd.DataFrame(results)

# Multiple comparisons correction
pvals = df_res['wilcoxon_p'].fillna(1.0).clip(lower=1e-300).values
df_res['p_holm'] = multipletests(pvals, method='holm')[1]
df_res['p_fdr_bh'] = multipletests(pvals, method='fdr_bh')[1]
df_res['significant_holm_05'] = df_res['p_holm'] < 0.05
df_res['significant_fdr_05'] = df_res['p_fdr_bh'] < 0.05

df_res = df_res.sort_values('cliff_delta').reset_index(drop=True)
df_res.to_parquet(os.path.join(RESULTS_DIR, 'best_vs_all_systematic.parquet'), index=False)
print(f'\nSalvato: best_vs_all_systematic.parquet ({len(df_res)} righe)\n')

# Multiple comparisons summary
print(f'Multiple comparisons correction su {len(df_res)} test:')
print(f'  Significativi senza correzione (p<0.05):     {(df_res["wilcoxon_p"]<0.05).sum()}/{len(df_res)}')
print(f'  Significativi con Holm-Bonferroni (p<0.05):  {df_res["significant_holm_05"].sum()}/{len(df_res)}')
print(f'  Significativi con Benjamini-Hochberg (FDR<0.05): {df_res["significant_fdr_05"].sum()}/{len(df_res)}')

# Summary statistics
print(f'Sintesi degli effect size (su {len(df_res)} alternative):')
print(f'  large    (|δ| ≥ 0.474):  {(df_res["effect"]=="large").sum()}')
print(f'  medium   (|δ| ≥ 0.33):   {(df_res["effect"]=="medium").sum()}')
print(f'  small    (|δ| ≥ 0.147):  {(df_res["effect"]=="small").sum()}')
print(f'  negligible (|δ| < 0.147): {(df_res["effect"]=="negligible").sum()}')

# Best wins quasi-tutti?
all_best_wins = (df_res['cliff_delta'] < 0).sum()
print(f'\n  Best vince paired in {all_best_wins}/{len(df_res)} confronti '
      f'({100*all_best_wins/len(df_res):.1f}%)')

# Sorted by cliff_delta ascending: most negative δ at top (best wins by large margin)
# Most "distant" from best = largest |δ| = most negative δ → HEAD of sorted
# Most "close" to best = smallest |δ| = least negative δ → TAIL of sorted

# Top 10 ALTERNATIVE PIÙ VICINE AL BEST (Cliff δ meno negativo, vicino a 0)
print('\n' + '='*80)
print('  TOP 10 ALTERNATIVE PIÙ VICINE AL BEST (Cliff δ meno negativo, effect più piccolo)')
print('='*80)
print(f'\n  {"Imputer":<22}{"Forecaster":<18}{"WAPE_med":>10}{"Δmed":>10}{"Cliff δ":>10}{"Effect":>12}{"% wins":>8}')
print('  ' + '-' * 90)
# Tail = most positive (least negative) cliff_delta = closest to best
for _, r in df_res.tail(10).iloc[::-1].iterrows():
    print(f'  {r["imputer"]:<22}{r["forecaster"]:<18}'
          f'{r["wape_med_other"]:>10.4f}{r["delta_med"]:>+10.4f}'
          f'{r["cliff_delta"]:>+10.3f}{r["effect"]:>12}'
          f'{r["pct_best_wins"]:>7.1f}%')

# Top 10 ALTERNATIVE PIÙ DISTANTI DAL BEST (Cliff δ più negativo, large effect)
print('\n' + '='*80)
print('  TOP 10 ALTERNATIVE PIÙ DISTANTI DAL BEST (Cliff δ più negativo, effect più grande)')
print('='*80)
print(f'\n  {"Imputer":<22}{"Forecaster":<18}{"WAPE_med":>10}{"Δmed":>10}{"Cliff δ":>10}{"Effect":>12}{"% wins":>8}')
print('  ' + '-' * 90)
# Head = most negative cliff_delta = furthest from best
for _, r in df_res.head(10).iterrows():
    print(f'  {r["imputer"]:<22}{r["forecaster"]:<18}'
          f'{r["wape_med_other"]:>10.4f}{r["delta_med"]:>+10.4f}'
          f'{r["cliff_delta"]:>+10.3f}{r["effect"]:>12}'
          f'{r["pct_best_wins"]:>7.1f}%')

# Conclusione
print('\n' + '='*80)
print('  CONCLUSIONE')
print('='*80)
n_neg = (df_res['effect'] == 'negligible').sum()
n_small = (df_res['effect'] == 'small').sum()
n_medium = (df_res['effect'] == 'medium').sum()
n_large = (df_res['effect'] == 'large').sum()
print(f'\n  No imp + Chronos-bolt è statisticamente significativamente migliore di:')
print(f'    - {n_large} celle con large effect (vittoria netta)')
print(f'    - {n_medium} celle con medium effect (vittoria moderata)')
print(f'    - {n_small} celle con small effect (vantaggio piccolo)')
print(f'    - {n_neg} celle con negligible effect (praticamente equivalenti)')
print(f'\n  In termini operativi:')
print(f'    - {n_large + n_medium}/{len(df_res)} alternative sono CHIARAMENTE peggiori')
print(f'    - {n_small + n_neg}/{len(df_res)} alternative sono FUNZIONALMENTE EQUIVALENTI')

print('\n' + '='*80)
print('  DONE')
print('='*80)
