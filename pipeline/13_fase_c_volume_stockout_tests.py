"""
13_fase_c_volume_stockout_tests.py — Test performance vs volume e stockout_rate
================================================================================
Per ogni cella (imputer × forecaster) della matrice, testa se il WAPE dipende
da volume e tasso di stockout, con effect size.

Per ogni cella e per ogni dimensione (volume, stockout_rate):
- Jonckheere-Terpstra: trend monotono significativo?
- Spearman ρ: forza della correlazione
- Cliff's δ Q1 vs Q4: effect size della differenza tra estremi
- Δ WAPE mediana Q1→Q4: magnitudine assoluta
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

# ===========================================================================
# 1. Load stratification and cells
# ===========================================================================
print('='*72)
print('  FASE C — TEST PERFORMANCE vs VOLUME e STOCKOUT RATE')
print('='*72)

strat = pd.read_parquet(os.path.join(RESULTS_DIR, 'stratification.parquet'))
rank_map = {'Q1':1, 'Q2':2, 'Q3':3, 'Q4':4}
strat['vol_rank'] = strat['vol_bin'].map(rank_map)
strat['so_rank'] = strat['so_bin'].map(rank_map)

IMPUTERS = ['No imputation', 'Media condizionata', 'Media globale',
            'Mediana condizionata', 'Mediana globale',
            'LGB imputer', 'DLinear',
            'Forward Fill', 'Seasonal Naive', 'Linear Interp', 'SAITS']
FORECASTERS = ['Global Mean', 'DoW Mean', 'MA (K=56)',
               'LGB (no lags)', 'LGB (M5 lags)',
               'MLP (no lags)', 'MLP (M5 lags)', 'Chronos-bolt']

FC_FILE = {'Global Mean':'global_mean','DoW Mean':'dow_mean','MA (K=56)':'ma_k56',
           'LGB (no lags)':'lgb_nolags','LGB (M5 lags)':'lgb_m5lags',
           'MLP (no lags)':'mlp_nolags','MLP (M5 lags)':'mlp_m5lags',
           'Chronos-bolt':'chronos_bolt'}
IMP_FILE = {'Media condizionata':'media_cond','Media globale':'media_glob',
            'Mediana condizionata':'mediana_cond','Mediana globale':'mediana_glob',
            'LGB imputer':'lgb',
            'DLinear':'dlinear',
            'Forward Fill':'forward_fill','Seasonal Naive':'seasonal_naive',
            'Linear Interp':'linear_interp','SAITS':'saits'}

def get_path(imp, fc):
    fc_safe = FC_FILE[fc]
    if fc == 'Chronos-bolt':
        imp_safe = 'no_imp' if imp == 'No imputation' else IMP_FILE[imp]
        return os.path.join(RESULTS_DIR, f'{imp_safe}__{fc_safe}_test_per_series.parquet')
    if imp == 'No imputation':
        if fc in ['Global Mean','DoW Mean','MA (K=56)']:
            return os.path.join(RESULTS_DIR, f'naive_{fc_safe}_test_per_series.parquet')
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
        if not os.path.exists(path):
            continue
        ps = pd.read_parquet(path)
        merged = ps.merge(strat[['store_id','product_id','vol_rank','so_rank']],
                          on=['store_id','product_id'], how='inner')
        cells[(imp, fc)] = merged
print(f'  Loaded {len(cells)} cells')

# ===========================================================================
# 2. Statistical tests
# ===========================================================================
def jonckheere_pvalue(groups):
    """Jonckheere-Terpstra via normal approximation. Returns (z, p_decreasing)."""
    k = len(groups)
    n_list = [len(g) for g in groups]
    N = sum(n_list)
    J = 0.0
    for i in range(k):
        for j in range(i+1, k):
            xi = groups[i]; xj = groups[j]
            J += np.sum(xi[:, None] < xj[None, :]) + 0.5 * np.sum(xi[:, None] == xj[None, :])
    mean_J = (N**2 - sum(ni**2 for ni in n_list)) / 4.0
    var_J = (N**2 * (2*N + 3) - sum(ni**2 * (2*ni + 3) for ni in n_list)) / 72.0
    z = (J - mean_J) / np.sqrt(var_J)
    p_decr = stats.norm.cdf(z)  # H1: trend decrescente nel rank
    return z, p_decr

def cliffs_delta(x, y):
    """Cliff's delta = P(X > Y) - P(X < Y). Positive = x tends to be larger."""
    x = np.asarray(x); y = np.asarray(y)
    n1, n2 = len(x), len(y)
    greater = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return (greater - less) / (n1 * n2)

def cliffs_delta_subsample(x, y, max_n=500, seed=42):
    """Cliff's delta su sottocampioni per efficienza."""
    rng = np.random.default_rng(seed)
    x_sub = rng.choice(x, size=min(len(x), max_n), replace=False)
    y_sub = rng.choice(y, size=min(len(y), max_n), replace=False)
    return cliffs_delta(x_sub, y_sub)

def cliff_effect(d):
    a = abs(d)
    if a < 0.147: return 'negligible'
    if a < 0.33:  return 'small'
    if a < 0.474: return 'medium'
    return 'large'

print('\n2. Running tests for each cell × each dimension...')

results = []
rng = np.random.default_rng(42)
for (imp, fc), df in cells.items():
    df_clean = df.dropna(subset=['hourly_wape'])
    for dim_name, rank_col in [('volume', 'vol_rank'), ('stockout', 'so_rank')]:
        df_d = df_clean.dropna(subset=[rank_col])
        # Groups per quartile
        groups = [df_d[df_d[rank_col] == q]['hourly_wape'].values for q in [1,2,3,4]]
        # Subsample for JT efficiency
        groups_sub = [rng.choice(g, size=min(len(g), 500), replace=False) for g in groups]

        # Spearman ρ
        rho, p_rho = stats.spearmanr(df_d[rank_col], df_d['hourly_wape'])

        # Jonckheere-Terpstra
        try:
            jt_z, jt_p_decr = jonckheere_pvalue(groups_sub)
        except Exception:
            jt_z, jt_p_decr = np.nan, np.nan

        # Cliff's delta Q1 vs Q4
        d_q1q4 = cliffs_delta_subsample(groups[0], groups[3], max_n=500)
        eff = cliff_effect(d_q1q4)

        # Medians
        meds = {f'Q{q}': np.median(g) for q, g in zip([1,2,3,4], groups)}
        delta_q1q4 = meds['Q1'] - meds['Q4']

        results.append({
            'imputer': imp, 'forecaster': fc, 'dimension': dim_name,
            **{f'med_{k}':v for k, v in meds.items()},
            'delta_q1_q4_median': delta_q1q4,
            'spearman_rho': rho, 'spearman_p': p_rho,
            'jt_z': jt_z, 'jt_p_decreasing': jt_p_decr,
            'cliff_delta_q1q4': d_q1q4, 'cliff_effect': eff,
            'n': len(df_d),
        })

df_res = pd.DataFrame(results)
df_res.to_parquet(os.path.join(RESULTS_DIR, 'volume_stockout_tests.parquet'), index=False)
print(f'  Salvato: volume_stockout_tests.parquet ({len(df_res)} righe)')

# ===========================================================================
# 3. Summary tables — una per dimensione
# ===========================================================================
for dim in ['volume', 'stockout']:
    print(f'\n{"="*72}')
    print(f'  DIMENSIONE: {dim.upper()}')
    print(f'{"="*72}')

    sub = df_res[df_res['dimension'] == dim]

    # Heatmap values
    pivot_cliff = sub.pivot(index='imputer', columns='forecaster', values='cliff_delta_q1q4')
    pivot_cliff = pivot_cliff.reindex(index=IMPUTERS, columns=FORECASTERS)

    pivot_delta = sub.pivot(index='imputer', columns='forecaster', values='delta_q1_q4_median')
    pivot_delta = pivot_delta.reindex(index=IMPUTERS, columns=FORECASTERS)

    pivot_rho = sub.pivot(index='imputer', columns='forecaster', values='spearman_rho')
    pivot_rho = pivot_rho.reindex(index=IMPUTERS, columns=FORECASTERS)

    pivot_jt_p = sub.pivot(index='imputer', columns='forecaster', values='jt_p_decreasing')
    pivot_jt_p = pivot_jt_p.reindex(index=IMPUTERS, columns=FORECASTERS)

    print(f'\n  Cliff\'s δ (Q1 vs Q4):')
    print('  ', pivot_cliff.round(3).to_string())

    print(f'\n  Δ WAPE mediana (Q1 - Q4):')
    print('  ', pivot_delta.round(3).to_string())

    print(f'\n  Spearman ρ (rank_{dim} vs WAPE):')
    print('  ', pivot_rho.round(3).to_string())

    # Count significance
    n_sig = (pivot_jt_p < 0.01).sum().sum()
    print(f'\n  Jonckheere-Terpstra: {n_sig}/48 celle con p<0.01 (trend decrescente significativo)')

    # Effect size summary
    print(f'\n  Distribuzione di Cliff\'s δ per forecaster:')
    for fc in FORECASTERS:
        vals = sub[sub['forecaster'] == fc]['cliff_delta_q1q4'].values
        effs = sub[sub['forecaster'] == fc]['cliff_effect'].values
        print(f'    {fc:<18}  δ_mean={np.mean(vals):+.3f}  δ_range=[{np.min(vals):+.3f}, {np.max(vals):+.3f}]  '
              f'effects={[e for e in effs]}')

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    mat = pivot_cliff.values
    im = ax.imshow(mat, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(FORECASTERS)))
    ax.set_yticks(range(len(IMPUTERS)))
    ax.set_xticklabels(FORECASTERS, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(IMPUTERS, fontsize=10)

    for i in range(len(IMPUTERS)):
        for j in range(len(FORECASTERS)):
            v = mat[i, j]
            if np.isnan(v): continue
            color = 'white' if abs(v) > 0.5 else 'black'
            eff = cliff_effect(v)
            fw = 'bold' if abs(v) >= 0.474 else 'normal'
            ax.text(j, i, f'{v:+.2f}\n({eff[:3]})', ha='center', va='center',
                    fontsize=8, color=color, fontweight=fw)

    for i in range(len(IMPUTERS)+1):
        ax.axhline(i - 0.5, color='white', linewidth=1.5)
    for j in range(len(FORECASTERS)+1):
        ax.axvline(j - 0.5, color='white', linewidth=1.5)

    fig.colorbar(im, ax=ax, label=f'Cliff\'s δ (WAPE Q1 vs Q4, dimensione {dim})')
    sign_note = 'δ > 0: Q1 WAPE > Q4 WAPE = il volume aiuta' if dim == 'volume' \
        else 'δ > 0: Q1 WAPE > Q4 WAPE = stockout basso aiuta'
    ax.set_title(f'Effect size (Cliff\'s δ) — sensibilità WAPE al quartile di {dim}\n{sign_note}',
                 fontsize=11, pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig12_cliff_delta_{dim}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Salvata: fig12_cliff_delta_{dim}.png')

# ===========================================================================
# 4. Cross-dimension comparison
# ===========================================================================
print(f'\n{"="*72}')
print('  RIASSUNTO CROSS-DIMENSIONE')
print(f'{"="*72}')

print('\n  Cliff\'s δ medio per forecaster (su tutti gli imputer):')
print('  Forecaster          δ_volume    δ_stockout    Interpretazione')
print('  ' + '-'*74)
for fc in FORECASTERS:
    d_vol = df_res[(df_res['forecaster']==fc) & (df_res['dimension']=='volume')]['cliff_delta_q1q4'].mean()
    d_so = df_res[(df_res['forecaster']==fc) & (df_res['dimension']=='stockout')]['cliff_delta_q1q4'].mean()
    eff_vol = cliff_effect(d_vol)
    eff_so = cliff_effect(d_so)
    print(f'  {fc:<18}  {d_vol:+.3f}({eff_vol[:3]})   {d_so:+.3f}({eff_so[:3]})')

print('\n'+'='*72)
print('  DONE — 13_fase_c_volume_stockout_tests.py')
print('='*72)
