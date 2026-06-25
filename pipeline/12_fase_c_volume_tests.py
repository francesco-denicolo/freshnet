"""
12_fase_c_volume_tests.py — Test statistici sul Finding 1 (effetto volume)
==========================================================================
Per ogni combinazione imputer × forecaster della matrice, testa:
- Trend monotono WAPE vs volume (Jonckheere-Terpstra)
- Correlazione Spearman volume_quartile vs WAPE
- Kruskal-Wallis (controllo generale)
- TOST per scala-invarianza (ε = 0.05)
"""
import os, functools, numpy as np, pandas as pd
from scipy import stats
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPS = 0.05  # soglia di equivalenza per TOST

# ===========================================================================
# 1. Load stratification and all per-series results
# ===========================================================================
print('='*72)
print('  FASE C — TEST STATISTICI SUL FINDING 1 (effetto volume)')
print('='*72)

strat = pd.read_parquet(os.path.join(RESULTS_DIR, 'stratification.parquet'))
vol_rank = {'Q1':1, 'Q2':2, 'Q3':3, 'Q4':4}
strat['vol_rank'] = strat['vol_bin'].map(vol_rank)

IMPUTERS = ['No imputation', 'Media condizionata', 'Media globale',
            'Mediana condizionata', 'LGB imputer', 'DLinear']
FORECASTERS = ['Global Mean', 'DoW Mean', 'MA (K=56)',
               'LGB (no lags)', 'LGB (M5 lags)',
               'MLP (no lags)', 'MLP (M5 lags)', 'Chronos-bolt']

FC_FILE = {'Global Mean':'global_mean','DoW Mean':'dow_mean','MA (K=56)':'ma_k56',
           'LGB (no lags)':'lgb_nolags','LGB (M5 lags)':'lgb_m5lags',
           'MLP (no lags)':'mlp_nolags','MLP (M5 lags)':'mlp_m5lags',
           'Chronos-bolt':'chronos_bolt'}
IMP_FILE = {'Media condizionata':'media_cond','Media globale':'media_glob',
            'Mediana condizionata':'mediana_cond','LGB imputer':'lgb',
            'DLinear':'dlinear'}

def get_path(imp, fc):
    fc_safe = FC_FILE[fc]
    if fc == 'Chronos-bolt':
        imp_safe = 'no_imp' if imp == 'No imputation' else IMP_FILE[imp]
        return os.path.join(RESULTS_DIR, f'{imp_safe}__{fc_safe}_test_per_series.parquet')
    if imp == 'No imputation':
        if fc in ['Global Mean','DoW Mean','MA (K=56)']:
            return os.path.join(RESULTS_DIR, f'naive_{fc_safe}_test_per_series.parquet')
        else:
            return os.path.join(RESULTS_DIR, f'{fc_safe}_test_per_series.parquet')
    imp_safe = IMP_FILE[imp]
    if fc in ['LGB (no lags)', 'MLP (no lags)']:
        return os.path.join(RESULTS_DIR, f'{fc_safe}_test_per_series.parquet')
    return os.path.join(RESULTS_DIR, f'{imp_safe}__{fc_safe}_test_per_series.parquet')

print('\n1. Loading cells...')
cells = {}
for imp in IMPUTERS:
    for fc in FORECASTERS:
        path = get_path(imp, fc)
        if os.path.exists(path):
            ps = pd.read_parquet(path)
            # Merge with stratification
            merged = ps.merge(strat[['store_id','product_id','vol_rank']],
                              on=['store_id','product_id'], how='inner')
            cells[(imp, fc)] = merged
print(f'  Loaded {len(cells)} cells')

# ===========================================================================
# 2. Run tests for each cell
# ===========================================================================
print('\n2. Running tests...')

def jonckheere_p_value(groups):
    """Approximate Jonckheere-Terpstra via normal approximation."""
    k = len(groups)
    n = [len(g) for g in groups]
    N = sum(n)
    # Statistic J: number of pairs (i<j) where x_i < x_j
    J = 0
    for i in range(k):
        for j in range(i+1, k):
            # Count pairs where group_i < group_j
            xi = groups[i]; xj = groups[j]
            # Vectorized count
            # J += sum over xi < xj
            J += np.sum(xi[:, None] < xj[None, :]) + 0.5 * np.sum(xi[:, None] == xj[None, :])
    # Expected value and variance under H0
    mean_J = (N**2 - sum(ni**2 for ni in n)) / 4.0
    var_J = (N**2 * (2*N + 3) - sum(ni**2 * (2*ni + 3) for ni in n)) / 72.0
    z = (J - mean_J) / np.sqrt(var_J)
    # One-sided p-value for decreasing trend: J should be SMALL if decreasing
    # (fewer pairs where x_i < x_j when groups are sorted by increasing index and values decrease)
    p_decreasing = stats.norm.cdf(z)
    p_increasing = stats.norm.sf(z)
    return z, p_decreasing, p_increasing

def tost_test(x1, x2, eps):
    """Two one-sided t-tests for equivalence.
    H0 (composite): |mu1 - mu2| >= eps.   H1: |mu1 - mu2| < eps.
    Returns p-value (max of two one-sided tests); p<alpha means equivalent.
    """
    t1, p1 = stats.ttest_ind(x1, x2 - eps, equal_var=False, alternative='greater')
    t2, p2 = stats.ttest_ind(x1, x2 + eps, equal_var=False, alternative='less')
    return max(p1, p2)

results = []
for (imp, fc), df in cells.items():
    df_clean = df.dropna(subset=['hourly_wape', 'vol_rank'])
    if len(df_clean) < 100:
        continue

    # Spearman correlation (vol_rank vs WAPE)
    rho, p_rho = stats.spearmanr(df_clean['vol_rank'], df_clean['hourly_wape'])

    # Kruskal-Wallis across quartiles
    groups = [df_clean[df_clean['vol_rank'] == q]['hourly_wape'].values for q in [1,2,3,4]]
    kw_stat, kw_p = stats.kruskal(*groups)

    # Jonckheere-Terpstra (trend decrescente nel rank)
    # Sottocampiono per rendere fattibile il test O(n^2)
    rng = np.random.default_rng(42)
    max_per_group = 500
    groups_sub = [rng.choice(g, size=min(len(g), max_per_group), replace=False) for g in groups]
    try:
        jt_z, jt_p_decr, jt_p_incr = jonckheere_p_value(groups_sub)
    except Exception as e:
        jt_z, jt_p_decr, jt_p_incr = np.nan, np.nan, np.nan

    # TOST per Q1 vs Q4 (scala-invarianza)
    q1 = df_clean[df_clean['vol_rank'] == 1]['hourly_wape'].values
    q4 = df_clean[df_clean['vol_rank'] == 4]['hourly_wape'].values
    tost_p = tost_test(q1, q4, eps=EPS)

    # Medians per quartile
    medians = {f'Q{q}_median': np.median(g) for q, g in zip([1,2,3,4], groups)}

    results.append({
        'imputer': imp, 'forecaster': fc,
        **medians,
        'delta_Q1_Q4': medians['Q1_median'] - medians['Q4_median'],
        'spearman_rho': rho, 'spearman_p': p_rho,
        'kw_H': kw_stat, 'kw_p': kw_p,
        'jt_z': jt_z, 'jt_p_decreasing': jt_p_decr,
        'tost_p_eq': tost_p,
        'n': len(df_clean),
    })

df_res = pd.DataFrame(results)
df_res.to_parquet(os.path.join(RESULTS_DIR, 'volume_tests.parquet'), index=False)
print(f'  Salvato: volume_tests.parquet ({len(df_res)} righe)')

# ===========================================================================
# 3. Summary tables
# ===========================================================================
print('\n3. Tabella Spearman ρ (volume quartile vs WAPE):')
print('   (valore molto negativo = volume aiuta molto; ~0 = scala-invariante)\n')

pivot_rho = df_res.pivot(index='imputer', columns='forecaster', values='spearman_rho')
pivot_rho = pivot_rho.reindex(index=IMPUTERS, columns=FORECASTERS)
print('  ', pivot_rho.round(4).to_string())

print('\n4. Tabella Δ(Q1 - Q4) mediana:')
print('   (grande = il volume aiuta; piccolo/negativo = scala-invariante o peggio)\n')
pivot_delta = df_res.pivot(index='imputer', columns='forecaster', values='delta_Q1_Q4')
pivot_delta = pivot_delta.reindex(index=IMPUTERS, columns=FORECASTERS)
print('  ', pivot_delta.round(4).to_string())

print('\n5. P-value Jonckheere-Terpstra (trend decrescente):')
print('   (p<0.01 = trend decrescente significativo)\n')
pivot_jt = df_res.pivot(index='imputer', columns='forecaster', values='jt_p_decreasing')
pivot_jt = pivot_jt.reindex(index=IMPUTERS, columns=FORECASTERS)
print('  ', pivot_jt.round(6).to_string())

print(f'\n6. P-value TOST (equivalenza Q1 ≡ Q4, ε={EPS}):')
print('   (p<0.05 = equivalenti, scala-invariante)\n')
pivot_tost = df_res.pivot(index='imputer', columns='forecaster', values='tost_p_eq')
pivot_tost = pivot_tost.reindex(index=IMPUTERS, columns=FORECASTERS)
print('  ', pivot_tost.round(6).to_string())

# ===========================================================================
# 4. Heatmap Spearman ρ
# ===========================================================================
print('\n7. Heatmap Spearman ρ...')

fig, ax = plt.subplots(figsize=(14, 6))
mat = pivot_rho.values
im = ax.imshow(mat, cmap='RdBu', aspect='auto', vmin=-0.5, vmax=0.5)
ax.set_xticks(range(len(FORECASTERS)))
ax.set_yticks(range(len(IMPUTERS)))
ax.set_xticklabels(FORECASTERS, rotation=30, ha='right', fontsize=10)
ax.set_yticklabels(IMPUTERS, fontsize=10)

for i in range(len(IMPUTERS)):
    for j in range(len(FORECASTERS)):
        v = mat[i, j]
        if np.isnan(v): continue
        color = 'white' if abs(v) > 0.3 else 'black'
        ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                fontsize=9, color=color,
                fontweight='bold' if abs(v) > 0.3 else 'normal')

for i in range(len(IMPUTERS)+1):
    ax.axhline(i - 0.5, color='white', linewidth=1.5)
for j in range(len(FORECASTERS)+1):
    ax.axvline(j - 0.5, color='white', linewidth=1.5)

fig.colorbar(im, ax=ax, label='Spearman ρ (volume rank vs WAPE)')
ax.set_title('Effetto del volume sul WAPE — Spearman ρ per ogni cella\n'
             '(ρ ≈ 0: scala-invariante; ρ negativo: volume migliora la predizione)',
             fontsize=12, pad=15)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig11_spearman_rho.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig11_spearman_rho.png')

# ===========================================================================
# 5. Findings summary
# ===========================================================================
print('\n'+'='*72)
print('  SINTESI DEI FINDINGS')
print('='*72)

# Per ogni forecaster, ρ medio sulla colonna
print('\n  Spearman ρ medio per forecaster (tra tutti gli imputer):')
for fc in FORECASTERS:
    rhos = df_res[df_res['forecaster'] == fc]['spearman_rho'].values
    mean_rho = np.mean(rhos)
    mean_delta = np.mean(df_res[df_res['forecaster'] == fc]['delta_Q1_Q4'].values)
    jt_mean_p = np.median(df_res[df_res['forecaster'] == fc]['jt_p_decreasing'].values)
    tost_mean_p = np.median(df_res[df_res['forecaster'] == fc]['tost_p_eq'].values)
    flag = '  '
    if abs(mean_rho) < 0.10:
        flag = '← scala-invariante' if tost_mean_p < 0.05 else '← effect piccolo'
    elif mean_rho < -0.20:
        flag = '← volume aiuta MOLTO'
    elif mean_rho < -0.10:
        flag = '← volume aiuta'
    print(f'    {fc:<18}  ρ_mean={mean_rho:+.4f}  Δ(Q1-Q4)={mean_delta:+.4f}  '
          f'JT p(med)={jt_mean_p:.2e}  TOST p(med)={tost_mean_p:.3f} {flag}')

print('\n'+'='*72)
print('  DONE — 12_fase_c_volume_tests.py')
print('='*72)
