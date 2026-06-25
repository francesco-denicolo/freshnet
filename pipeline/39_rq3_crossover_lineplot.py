"""
39_rq3_crossover_lineplot.py — RQ3 crossover visualization across volume quartiles
==================================================================================
Line chart that visualizes the crossover: for each selected cell, plot WAPE_h_med
across Q1-Q4. Lines that cross each other expose the regime-dependence of the
"best cell".

Selection of cells to plot:
  - top-3 globali (from general matrix)
  - best per quartile (Q1, Q2, Q3, Q4)
  - plus a few "interesting" cells for context (worst-on-Q1 cell that wins Q4 etc.)
"""
import os, functools, numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load matrices
strat = pd.read_parquet(f'{RESULTS_DIR}/hpo_stratified_quartile.parquet')
matgen = pd.read_parquet(f'{RESULTS_DIR}/hpo_matrix_pareto.parquet')

# ---------------------------------------------------------------------
# Two complementary views of "best cell":
#   (a) Fixed cell: best globally per family, plot across quartiles
#   (b) Per-quartile best: choose the best cell per family per quartile (cells change)
# ---------------------------------------------------------------------
families = ['mlp_m5lags','lgb_m5lags','tft','chronos_bolt','timesfm',
            'global_mean','dow_mean','ma_k56']

# (a) Best globally per family
family_best_global = {}
for fc in families:
    sub = matgen[matgen.forecaster==fc].sort_values('wape_h_med')
    if len(sub) == 0: continue
    family_best_global[fc] = sub.iloc[0]['cell']

# (b) Best per family PER QUARTILE
family_best_per_q = {fc: {} for fc in families}
for fc in families:
    for q in ['Q1','Q2','Q3','Q4']:
        sub = strat[(strat.forecaster==fc) & (strat.quartile==q)].sort_values('wape_h_med')
        if len(sub) == 0: continue
        family_best_per_q[fc][q] = (sub.iloc[0]['cell'], float(sub.iloc[0]['wape_h_med']))

# (legacy) selected cells = best globals (for the original single plot)
selected = list(family_best_global.values())
best_per_q = {}
for q in ['Q1','Q2','Q3','Q4']:
    best_per_q[q] = strat[strat.quartile==q].sort_values('wape_h_med').iloc[0]['cell']
print(f'Selected {len(selected)} cells:')
for c in selected:
    g = matgen[matgen.cell==c]
    wape = g.wape_h_med.iloc[0] if len(g) else np.nan
    print(f'  {c}: global WAPE={wape:.4f}')

# Build long-format dataframe: cell, quartile, wape_h_med
long_df = strat[strat.cell.isin(selected)].copy()
pivot = long_df.pivot(index='cell', columns='quartile', values='wape_h_med')
pivot = pivot[['Q1','Q2','Q3','Q4']]
print('\nWAPE_h_med per quartile (selected cells):')
print(pivot.to_string(float_format='%.4f'))

# ---------------------------------------------------------------------
# V2 plot: 2 panels side-by-side
#   (a) Fixed cell (best globally per family) plotted across quartiles
#   (b) Best cell per quartile per family (cells change)
# Same color per family across both panels for direct comparison.
# ---------------------------------------------------------------------
FC_COLORS = {'mlp_m5lags':'#4575b4','lgb_m5lags':'#f46d43','tft':'#7b3294',
             'chronos_bolt':'#d73027','timesfm':'#b15928','global_mean':'#2ca02c',
             'dow_mean':'#bcbd22','ma_k56':'#17becf'}
FC_LABELS = {'mlp_m5lags':'MLP_M5','lgb_m5lags':'LGB_M5','tft':'TFT',
             'chronos_bolt':'Chronos','timesfm':'TimesFM','global_mean':'GlobalMean',
             'dow_mean':'DoWMean','ma_k56':'MA56'}
markers = ['o','D','s','v','^','P','X']

quartiles = ['Q1','Q2','Q3','Q4']
x = np.arange(4)

# ---------- Figure A: fixed cell (best globally) ----------
fig_a, ax = plt.subplots(figsize=(13, 9))
for i, fc in enumerate(families):
    if fc not in family_best_global: continue
    cell = family_best_global[fc]
    if cell not in pivot.index: continue
    vals = pivot.loc[cell].values
    label = f'{FC_LABELS[fc]} ({cell.split("__")[0]})'
    ax.plot(x, vals, marker=markers[i % len(markers)], markersize=14,
            linewidth=2.5, color=FC_COLORS[fc], label=label,
            markerfacecolor=FC_COLORS[fc], markeredgecolor='black',
            markeredgewidth=1.2, zorder=4)
ax.set_xticks(x)
ax.set_xticklabels(['Q1\n(basso vol)', 'Q2', 'Q3', 'Q4\n(alto vol)'], fontsize=13)
ax.set_ylabel('WAPE_h median (lower = better)', fontsize=14)
ax.set_xlabel('Volume quartile', fontsize=14)
ax.set_title('RQ3 — Crossover (a) Fixed cell per family (best globally)\n'
             'Same imputer across all quartiles',
             fontsize=15, pad=12)
ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
out_a = f'{FIG_DIR}/fig_rq3_crossover_fixed_global.png'
fig_a.savefig(out_a, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out_a}')

# ---------- Figure B: best per quartile per family (no annotations) ----------
fig_b, ax = plt.subplots(figsize=(13, 9))
for i, fc in enumerate(families):
    bpq = family_best_per_q.get(fc, {})
    if len(bpq) == 0: continue
    vals = []
    for q in quartiles:
        vals.append(bpq[q][1] if q in bpq else np.nan)
    label = f'{FC_LABELS[fc]} (best/Q)'
    ax.plot(x, vals, marker=markers[i % len(markers)], markersize=14,
            linewidth=2.5, color=FC_COLORS[fc], label=label,
            markerfacecolor=FC_COLORS[fc], markeredgecolor='black',
            markeredgewidth=1.2, zorder=4)
ax.set_xticks(x)
ax.set_xticklabels(['Q1\n(basso vol)', 'Q2', 'Q3', 'Q4\n(alto vol)'], fontsize=13)
ax.set_ylabel('WAPE_h median (lower = better)', fontsize=14)
ax.set_xlabel('Volume quartile', fontsize=14)
ax.set_title('RQ3 — Crossover (b) Best cell per family PER QUARTILE\n'
             'Imputer may change across quartiles',
             fontsize=15, pad=12)
ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
out_b = f'{FIG_DIR}/fig_rq3_crossover_perq.png'
fig_b.savefig(out_b, dpi=150, bbox_inches='tight')
print(f'Saved: {out_b}')

# Save table for paper
pivot.to_parquet(f'{RESULTS_DIR}/rq3_crossover_table.parquet')
print('Saved: rq3_crossover_table.parquet')
print('\nDONE')
