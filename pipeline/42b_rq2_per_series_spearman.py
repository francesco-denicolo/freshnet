"""
42b_rq2_per_series_spearman.py — RQ2 extended: per-series Spearman ρ
====================================================================
Per ciascuno dei 4 forecaster (MLP_M5, LGB_M5, TFT, Chronos):
  Per ogni serie i (i=1..50K):
    1. Estrai 9 valori (recovery_imp, WAPE_forecasting_series_i_under_imp)
    2. Calcola Spearman ρ_i sui 9 punti
  Aggrega: mediana, IQR, distribuzione, test Wilcoxon "mediana ρ ≠ 0".

Risponde a: "per una serie tipica, l'imputer con miglior recovery dà
miglior forecasting?"
"""
import os, glob, functools, numpy as np, pandas as pd
from scipy import stats
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Recovery values (same as RQ2 base)
recovery = {
    'mediana_glob': 0.8090, 'imputeformer': 0.8666, 'itransformer': 0.9302,
    'saits': 0.9431, 'dlinear': 0.9513, 'timesnet': 1.0405,
    'linear_interp': 1.0473, 'seasonal_naive': 1.0638, 'forward_fill': 1.1878,
}
imputers = list(recovery.keys())
recovery_vec = np.array([recovery[i] for i in imputers])
print(f'9 imputer con WAPE_recovery')

# ---------------------------------------------------------------------
# Helper: load WAPE per series for (imputer, forecaster) cell
# ---------------------------------------------------------------------
NON_HPO_FC = {'chronos_bolt','global_mean','dow_mean','ma_k21'}

def find_parquet(imp, fc):
    p_hpo = f'{RESULTS_DIR}/{imp}__{fc}_hpo_test_per_series.parquet'
    p_plain = f'{RESULTS_DIR}/{imp}__{fc}_test_per_series.parquet'
    if os.path.exists(p_hpo): return p_hpo
    if os.path.exists(p_plain): return p_plain
    return None

# ---------------------------------------------------------------------
# Per-series Spearman per forecaster
# ---------------------------------------------------------------------
panels = ['mlp_m5lags','lgb_m5lags','tft','chronos_bolt']
panel_titles = {'mlp_m5lags':'MLP_M5','lgb_m5lags':'LGB_M5','tft':'TFT','chronos_bolt':'Chronos-bolt'}

results_per_fc = {}
for fc in panels:
    print(f'\n=== {panel_titles[fc]} ===')
    # Build WAPE matrix [series × imputer]
    dfs = {}
    for imp in imputers:
        p = find_parquet(imp, fc)
        if p is None:
            print(f'  WARN: missing {imp}__{fc}')
            dfs[imp] = None
            continue
        dfs[imp] = pd.read_parquet(p).set_index(['store_id','product_id'])['hourly_wape']
    # Common series index = intersection
    avail = [imp for imp in imputers if dfs[imp] is not None]
    if len(avail) < 5:
        print(f'  SKIP: only {len(avail)} imputers available'); continue
    common = dfs[avail[0]].index
    for imp in avail[1:]:
        common = common.intersection(dfs[imp].index)
    print(f'  Common series across {len(avail)} imputers: {len(common):,}')

    # Build matrix [n_common × len(avail)]
    M = np.full((len(common), len(avail)), np.nan, dtype=np.float64)
    for j, imp in enumerate(avail):
        M[:, j] = dfs[imp].loc[common].values
    rec_avail = np.array([recovery[i] for i in avail])

    # Per-series Spearman: rank recovery + rank WAPE per row
    # Spearman = Pearson on ranks
    print(f'  Computing {len(common):,} per-series Spearman ρ...')
    # Rank recovery (constant across series)
    rank_rec = stats.rankdata(rec_avail)
    # Rank WAPE per series (along columns = imputers axis)
    rank_M = np.argsort(np.argsort(M, axis=1), axis=1).astype(np.float64) + 1
    # If a row has NaN in M, set those to NaN in rank too
    nan_mask = np.isnan(M)
    rank_M[nan_mask] = np.nan
    # Compute correlation per row (Pearson on ranks)
    # ρ = cov(x, y) / (std(x) * std(y))
    valid_per_row = (~nan_mask).sum(axis=1)
    rho_per_series = np.full(len(common), np.nan)
    for i in range(len(common)):
        if valid_per_row[i] < 5:
            continue
        x = rank_rec[~nan_mask[i]]
        y = rank_M[i, ~nan_mask[i]]
        # Re-rank y (drop NaN may break ordering)
        y = stats.rankdata(M[i, ~nan_mask[i]])
        # Pearson on ranks = Spearman
        if x.std() == 0 or y.std() == 0:
            continue
        rho_per_series[i] = np.corrcoef(x, y)[0, 1]
    print(f'  Valid ρ: {(~np.isnan(rho_per_series)).sum():,}/{len(common):,}')

    rho_valid = rho_per_series[~np.isnan(rho_per_series)]
    med = np.median(rho_valid)
    q25, q75 = np.percentile(rho_valid, [25, 75])
    # Test: median ρ ≠ 0
    w_stat, w_pval = stats.wilcoxon(rho_valid - 0.0, alternative='two-sided')
    # Cliff's δ vs 0: P(ρ > 0) - P(ρ < 0)
    cliff_d = ((rho_valid > 0).sum() - (rho_valid < 0).sum()) / len(rho_valid)
    # Compare with median-based Spearman (the original n=9 analysis)
    median_wape = np.array([np.nanmedian(M[:, j]) for j in range(len(avail))])
    rho_median, p_median = stats.spearmanr(rec_avail, median_wape)
    results_per_fc[fc] = {
        'rho_dist': rho_valid,
        'median_rho': med, 'q25': q25, 'q75': q75,
        'wilcoxon_p': w_pval, 'cliffs_delta': cliff_d,
        'n_series': len(rho_valid),
        'rho_median_n9': rho_median,  # confronto con n=9 approach
    }
    print(f'  median ρ = {med:+.4f} (IQR [{q25:+.4f}, {q75:+.4f}])')
    print(f'  Cliff δ vs 0: {cliff_d:+.4f}  Wilcoxon p={w_pval:.2e}')
    print(f'  Recovery → forecasting at median ranking (n=9): ρ = {rho_median:+.4f}')

# ---------------------------------------------------------------------
# Plot: violin per forecaster
# ---------------------------------------------------------------------
print('\nBuilding violin plot...')
fig, ax = plt.subplots(figsize=(13, 8))
data = []
labels = []
for fc in panels:
    if fc not in results_per_fc: continue
    data.append(results_per_fc[fc]['rho_dist'])
    labels.append(panel_titles[fc])

positions = np.arange(len(data)) + 1
parts = ax.violinplot(data, positions=positions, widths=0.7, showmeans=False,
                      showmedians=True, showextrema=False)
# Color violins
colors = ['#4575b4','#f46d43','#7b3294','#d73027']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.6)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.2)
parts['cmedians'].set_color('white')
parts['cmedians'].set_linewidth(2.5)

# Zero reference line
ax.axhline(0, color='gray', linestyle='--', alpha=0.6, linewidth=1)

# Annotate median + p-value + cliff δ
for i, fc in enumerate([f for f in panels if f in results_per_fc]):
    r = results_per_fc[fc]
    txt = (f'median ρ = {r["median_rho"]:+.3f}\n'
           f'Cliff δ = {r["cliffs_delta"]:+.3f}\n'
           f'(n={r["n_series"]:,})')
    ax.text(positions[i], -0.95, txt, ha='center', va='top',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9,
                      edgecolor='gray'))

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=13)
ax.set_ylim(-1.05, 1.05)
ax.set_ylabel('Per-series Spearman ρ (recovery vs forecasting)', fontsize=13)
ax.set_title('RQ2 extended — Per-series Spearman ρ distribution\n'
             '(positive median = imputer recovery quality predicts forecasting for typical series)',
             fontsize=13, pad=12)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.tick_params(axis='y', labelsize=11)

plt.tight_layout()
out_fig = f'{FIG_DIR}/fig_rq2_per_series_spearman.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'Saved: {out_fig}')

# Save summary
summary = pd.DataFrame([
    {'forecaster': panel_titles[fc],
     'n_series_valid': results_per_fc[fc]['n_series'],
     'median_rho': results_per_fc[fc]['median_rho'],
     'q25_rho': results_per_fc[fc]['q25'],
     'q75_rho': results_per_fc[fc]['q75'],
     'cliffs_delta_vs_zero': results_per_fc[fc]['cliffs_delta'],
     'wilcoxon_p': results_per_fc[fc]['wilcoxon_p'],
     'rho_n9_median_approach': results_per_fc[fc]['rho_median_n9']}
    for fc in panels if fc in results_per_fc
])
summary.to_parquet(f'{RESULTS_DIR}/rq2_per_series_spearman_summary.parquet', index=False)
print('\nSummary saved.')

print('\nDONE')
