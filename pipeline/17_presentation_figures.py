"""
17_presentation_figures.py — 3 grafici ad-hoc per la presentazione
====================================================================
1. Trade-off WAPE × |WPE| scatter (slide trade-off)
2. Line chart: WAPE vs quartile volume per forecaster (slide crossover)
3. Bar chart: Cliff's δ per dimension (volume vs stockout) (slide effect size)
"""
import os, functools, numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

# Color per forecaster
FC_COLORS = {
    'Global Mean':    '#bcbcbc',
    'DoW Mean':       '#969696',
    'MA (K=56)':      '#737373',
    'LGB (no lags)':  '#fdae61',
    'LGB (M5 lags)':  '#f46d43',
    'MLP (no lags)':  '#74add1',
    'MLP (M5 lags)':  '#4575b4',
    'Chronos-bolt':   '#d73027',
}

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

print('='*72)
print('  GENERAZIONE GRAFICI PER PRESENTAZIONE')
print('='*72)

# ===========================================================================
# Load all 80 cells
# ===========================================================================
print('\n1. Caricamento celle...')
cells = {}
for imp in IMPUTERS:
    for fc in FORECASTERS:
        p = get_path(imp, fc)
        if os.path.exists(p):
            ps = pd.read_parquet(p)
            wape_med = ps['hourly_wape'].dropna().median()
            wpe_med = ps['hourly_wpe'].dropna().median()
            cells[(imp, fc)] = {'wape_med': wape_med, 'wpe_med': wpe_med, 'ps': ps}
print(f'  Loaded {len(cells)} cells')

# ===========================================================================
# FIGURA A: Trade-off WAPE × |WPE| scatter
# ===========================================================================
print('\n2. Figura A — Trade-off WAPE × |WPE|...')

fig, ax = plt.subplots(figsize=(11, 7))

# Group points by forecaster
for fc in FORECASTERS:
    xs, ys, labels = [], [], []
    for imp in IMPUTERS:
        if (imp, fc) not in cells: continue
        c = cells[(imp, fc)]
        xs.append(c['wape_med'])
        ys.append(abs(c['wpe_med']))
        labels.append(imp)
    ax.scatter(xs, ys, s=80, color=FC_COLORS[fc], alpha=0.75,
               edgecolor='black', linewidth=0.5, label=fc, zorder=3)

# Highlight best WAPE cell
best_key = min(cells, key=lambda k: cells[k]['wape_med'])
b = cells[best_key]
ax.scatter([b['wape_med']], [abs(b['wpe_med'])], s=350, marker='*',
           color='gold', edgecolor='black', linewidth=1.5, zorder=5,
           label=f'Best WAPE: {best_key[1]} + {best_key[0]}')
ax.annotate(f"{best_key[1]}\n+ {best_key[0]}",
            (b['wape_med'], abs(b['wpe_med'])),
            xytext=(15, 0), textcoords='offset points', fontsize=9,
            fontweight='bold', color='black')

# Highlight best WPE (lowest |WPE|)
best_wpe_key = min(cells, key=lambda k: abs(cells[k]['wpe_med']))
b2 = cells[best_wpe_key]
ax.scatter([b2['wape_med']], [abs(b2['wpe_med'])], s=350, marker='D',
           facecolor='none', edgecolor='green', linewidth=2.5, zorder=4,
           label=f'Best |WPE|: {best_wpe_key[1]} + {best_wpe_key[0]}')
ax.annotate(f"{best_wpe_key[1]}\n+ {best_wpe_key[0]}",
            (b2['wape_med'], abs(b2['wpe_med'])),
            xytext=(-110, 0), textcoords='offset points', fontsize=9,
            fontweight='bold', color='darkgreen')

ax.set_xlabel('WAPE mediana per-serie (accuratezza)', fontsize=12)
ax.set_ylabel('|WPE| mediana per-serie (bias)', fontsize=12)
ax.set_title('Trade-off accuratezza vs bias — 80 celle (10 imputer × 8 forecaster)',
             fontsize=13, pad=12)
ax.grid(alpha=0.3, zorder=0)
ax.legend(loc='upper right', fontsize=9, ncol=2)

# Annotate ideal corner
ax.annotate('IDEALE\n(basso WAPE, basso |WPE|)',
            xy=(ax.get_xlim()[0] + 0.01, ax.get_ylim()[0] + 0.02),
            xytext=(ax.get_xlim()[0] + 0.05, ax.get_ylim()[0] + 0.15),
            fontsize=10, color='gray', style='italic',
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig15_tradeoff_wape_wpe.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig15_tradeoff_wape_wpe.png')

# ===========================================================================
# FIGURA B: Line chart WAPE vs quartile volume per forecaster
# ===========================================================================
print('\n3. Figura B — Line chart WAPE vs quartile volume...')

# Load stratification
strat = pd.read_parquet(os.path.join(RESULTS_DIR, 'stratification.parquet'))
strat_keys = strat[['store_id','product_id','vol_bin']]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Per ogni forecaster, prendi il "best imputer" e quello "no_imp" (default)
for ax_idx, mode in enumerate(['no_imp', 'best_imp']):
    ax = axes[ax_idx]
    for fc in FORECASTERS:
        if mode == 'no_imp':
            imp = 'No imputation'
        else:
            # Best imputer per quel forecaster (su WAPE med globale)
            scores = {imp: cells[(imp, fc)]['wape_med']
                       for imp in IMPUTERS if (imp, fc) in cells}
            imp = min(scores, key=scores.get)
        if (imp, fc) not in cells: continue
        ps = cells[(imp, fc)]['ps']
        merged = strat_keys.merge(ps[['store_id','product_id','hourly_wape']],
                                   on=['store_id','product_id'], how='inner').dropna()
        wape_per_q = []
        for q in ['Q1','Q2','Q3','Q4']:
            sub = merged[merged['vol_bin'] == q]
            wape_per_q.append(sub['hourly_wape'].median())
        label = f'{fc}' if mode == 'no_imp' else f'{fc} + {imp}'
        ax.plot([1,2,3,4], wape_per_q, marker='o', color=FC_COLORS[fc],
                linewidth=2.5, markersize=10, label=label, alpha=0.85)

    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(['Q1\n(basso)','Q2','Q3','Q4\n(alto)'], fontsize=11)
    ax.set_xlabel('Quartile di volume', fontsize=12)
    ax.set_ylabel('WAPE mediana per-serie', fontsize=12)
    title_mode = 'No imputation' if mode == 'no_imp' else 'Best imputer per forecaster'
    ax.set_title(f'{title_mode}', fontsize=12, pad=10)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0.7, 1.35)

# Annotation: crossover
axes[0].annotate('CROSSOVER\n(Chronos vince a sinistra,\nML vince a destra)',
                 xy=(3, 1.0), xytext=(2.0, 1.20),
                 fontsize=10, color='darkred', fontweight='bold', ha='center',
                 arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.7))

fig.suptitle('Effetto del volume sul WAPE per forecaster (sinistra: no imputation, destra: best imputer)',
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig16_crossover_volume.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig16_crossover_volume.png')

# ===========================================================================
# FIGURA C: Bar chart Cliff's δ per dimensione (volume vs stockout)
# ===========================================================================
print('\n4. Figura C — Bar chart Cliff\'s δ per dimensione...')

# Load test results
df_tests = pd.read_parquet(os.path.join(RESULTS_DIR, 'volume_stockout_tests.parquet'))

# Average Cliff's delta per forecaster per dimension
delta_summary = []
for fc in FORECASTERS:
    for dim in ['volume', 'stockout']:
        sub = df_tests[(df_tests['forecaster'] == fc) & (df_tests['dimension'] == dim)]
        delta_summary.append({
            'forecaster': fc, 'dimension': dim,
            'cliff_delta': sub['cliff_delta_q1q4'].mean()
        })
df_summary = pd.DataFrame(delta_summary)

fig, ax = plt.subplots(figsize=(13, 6))

x = np.arange(len(FORECASTERS))
width = 0.4

vol_vals = df_summary[df_summary['dimension'] == 'volume'].set_index('forecaster').reindex(FORECASTERS)['cliff_delta'].values
so_vals = df_summary[df_summary['dimension'] == 'stockout'].set_index('forecaster').reindex(FORECASTERS)['cliff_delta'].values

bars1 = ax.bar(x - width/2, vol_vals, width, label='Volume', color='#3690c0', edgecolor='black')
bars2 = ax.bar(x + width/2, so_vals, width, label='Stockout rate', color='#e6550d', edgecolor='black')

# Annotate bars
for i, (v, s) in enumerate(zip(vol_vals, so_vals)):
    ax.text(i - width/2, v + 0.015, f'{v:.2f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
    ax.text(i + width/2, s + 0.015, f'{s:.2f}', ha='center', va='bottom',
            fontsize=9)

# Effect size thresholds
ax.axhline(0.474, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(0.33, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.axhline(0.147, color='gray', linestyle=':', alpha=0.3, linewidth=1)
ax.text(7.7, 0.474, ' large (0.474)', fontsize=8, color='gray', va='center')
ax.text(7.7, 0.33, ' medium (0.33)', fontsize=8, color='gray', va='center')
ax.text(7.7, 0.147, ' small (0.147)', fontsize=8, color='gray', va='center')

ax.set_xticks(x)
ax.set_xticklabels(FORECASTERS, rotation=20, ha='right', fontsize=10)
ax.set_ylabel("Cliff's δ medio (Q1 vs Q4)", fontsize=12)
ax.set_title("Sensibilità del WAPE a volume e stockout — Cliff's δ medio sui 10 imputer",
             fontsize=13, pad=12)
ax.legend(loc='upper left', fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.95)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig17_cliff_delta_dim_bars.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig17_cliff_delta_dim_bars.png')

print('\n' + '='*72)
print('  DONE — 3 figure salvate per la presentazione')
print('='*72)
print('\n  fig15_tradeoff_wape_wpe.png        (slide trade-off)')
print('  fig16_crossover_volume.png         (slide crossover)')
print('  fig17_cliff_delta_dim_bars.png     (slide effect size)')
