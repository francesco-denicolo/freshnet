"""
42_rq2_recovery_vs_forecasting.py — RQ2 recovery quality vs downstream forecasting
====================================================================================
Hypothesis: WAPE_recovery (Traccia A, imputation quality on MNAR-masked ground truth)
does NOT predict WAPE_forecasting (Traccia B, downstream forecasting accuracy).

Approach:
  - 9 imputers with WAPE_recovery available
  - For each forecaster (MLP_M5, LGB_M5, TFT, Chronos): scatter recovery vs forecasting
  - Spearman ρ to quantify
  - Highlight outliers (e.g., TimesNet: bad recovery 1.04, but best forecasting cell)
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

# ---------------------------------------------------------------------
# 1. Collect WAPE_recovery per imputer
# ---------------------------------------------------------------------
print('1. Collecting WAPE_recovery per imputer...')
recovery = {}
ta_files = glob.glob(f'{RESULTS_DIR}/traccia_a_*.parquet')
# Also include traccia_a.parquet (no underscore-suffix; contains the 4 naive/lgb imputer entries)
_ta_main = f'{RESULTS_DIR}/traccia_a.parquet'
if os.path.exists(_ta_main):
    ta_files.append(_ta_main)
imp_name_map = {
    'Forward Fill':'forward_fill','Seasonal Naive':'seasonal_naive',
    'Linear Interp':'linear_interp','DLinear':'dlinear','SAITS':'saits',
    'iTransformer':'itransformer','TimesNet':'timesnet',
    'ImputeFormer':'imputeformer','Mediana globale':'mediana_glob',
    'Media globale':'media_glob','Media condizionata':'media_cond',
    'Mediana condizionata':'mediana_cond','LGB imputer':'lgb',
    'CSDI':'csdi',  # CSDI was removed but if file lingers, map it
}
for f in ta_files:
    df = pd.read_parquet(f)
    for _, r in df.iterrows():
        key = imp_name_map.get(r['imputer'], None)
        if key is None:
            print(f'  WARN: unknown imputer label {r["imputer"]}')
            continue
        recovery[key] = float(r['wape_recovery'])

# Drop CSDI if present (was removed from matrix)
recovery.pop('csdi', None)
print(f'   {len(recovery)} imputers with WAPE_recovery:')
for k, v in sorted(recovery.items(), key=lambda x: x[1]):
    print(f'     {k}: {v:.4f}')

# ---------------------------------------------------------------------
# 2. Compute WAPE_forecasting median per (imputer, forecaster)
# ---------------------------------------------------------------------
print('\n2. Loading forecasting cells...')
NON_HPO_FC = {'chronos_bolt','timesfm','global_mean','dow_mean','ma_k56'}

def parse_name(name):
    if '__' in name: return name.split('__', 1)
    return 'no_imp', name

forecasting = {}  # (imputer, forecaster) -> WAPE_h_med
for f in sorted(glob.glob(f'{RESULTS_DIR}/*_hpo_test_per_series.parquet')):
    name = os.path.basename(f).replace('_hpo_test_per_series.parquet','')
    imp, fc = parse_name(name)
    forecasting[(imp, fc)] = pd.read_parquet(f)['hourly_wape'].median()
for f in sorted(glob.glob(f'{RESULTS_DIR}/*_test_per_series.parquet')):
    fn = os.path.basename(f)
    if '_hpo_test_per_series' in fn: continue
    name = fn.replace('_test_per_series.parquet','')
    if name.startswith('naive_'):
        base = name.replace('naive_','',1)
        if base in NON_HPO_FC and ('no_imp', base) not in forecasting:
            forecasting[('no_imp', base)] = pd.read_parquet(f)['hourly_wape'].median()
        continue
    imp, fc = parse_name(name)
    if fc in NON_HPO_FC and (imp, fc) not in forecasting:
        forecasting[(imp, fc)] = pd.read_parquet(f)['hourly_wape'].median()

print(f'   {len(forecasting)} forecasting cells loaded')

# ---------------------------------------------------------------------
# 3. Scatter plots: 4 panels (MLP_M5, LGB_M5, TFT, Chronos)
# ---------------------------------------------------------------------
print('\n3. Building scatter plots...')
panels = ['mlp_m5lags','lgb_m5lags','tft','chronos_bolt','timesfm']
panel_titles = {'mlp_m5lags':'MLP_M5','lgb_m5lags':'LGB_M5','tft':'TFT',
                'chronos_bolt':'Chronos-bolt','timesfm':'TimesFM'}

fig, axes = plt.subplots(2, 3, figsize=(22, 11))
spearman_results = {}
for ax, fc in zip(axes.flat, panels):
    xs, ys, labels = [], [], []
    for imp, wr in recovery.items():
        if (imp, fc) in forecasting:
            xs.append(wr); ys.append(forecasting[(imp, fc)]); labels.append(imp)
    if len(xs) < 3:
        ax.text(0.5, 0.5, f'Not enough data for {fc}', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        continue
    xs, ys = np.array(xs), np.array(ys)
    rho, pval = stats.spearmanr(xs, ys)
    spearman_results[fc] = {'rho': rho, 'p': pval, 'n': len(xs)}
    # Scatter
    ax.scatter(xs, ys, s=180, c='#1f77b4', edgecolor='black', linewidth=1.5, zorder=3)
    # Label points
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    # Trend line
    if len(xs) >= 3:
        z = np.polyfit(xs, ys, 1)
        xline = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xline, np.polyval(z, xline), '--', color='gray', alpha=0.6, lw=1.5, zorder=2)
    # Highlight TimesNet outlier (best forecasting cell despite worst recovery)
    if 'timesnet' in labels:
        i = labels.index('timesnet')
        ax.scatter([xs[i]], [ys[i]], s=350, facecolors='none',
                   edgecolor='red', linewidth=2.8, zorder=4, label='TimesNet (outlier)')
    ax.set_xlabel('WAPE_recovery (lower = better imputation)', fontsize=12)
    ax.set_ylabel(f'WAPE_forecasting med (× {panel_titles[fc]})', fontsize=12)
    ax.set_title(f'{panel_titles[fc]}   |   Spearman ρ = {rho:+.3f}, p = {pval:.3f}',
                 fontsize=13, pad=8)
    ax.grid(True, alpha=0.3, linestyle='--')

# Hide unused axes (last cell of 2x3 grid)
for i in range(len(panels), len(axes.flat)):
    axes.flat[i].axis('off')

fig.suptitle('RQ2 — Imputer recovery quality vs downstream forecasting quality\n'
             '(if recovery predicts forecasting, ρ would be strongly positive)',
             fontsize=14, y=1.005)
plt.tight_layout()
out_fig = f'{FIG_DIR}/fig_rq2_recovery_vs_forecasting.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'   Saved: {out_fig}')

# Summary
print('\n4. Spearman ρ summary:')
print(f'{"Forecaster":<15} {"Spearman ρ":<14} {"p-value":<12} {"n":<5}')
print('-' * 55)
for fc, r in spearman_results.items():
    print(f'{panel_titles[fc]:<15} {r["rho"]:<+14.4f} {r["p"]:<12.4f} {r["n"]:<5}')

print('\nDONE')
