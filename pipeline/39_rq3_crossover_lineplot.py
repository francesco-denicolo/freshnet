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
# Build selection: 1 best cell per "family" to expose true crossover
# Families: MLP_M5 (ML), LGB_M5 (boosted ML), TFT (DL), Chronos (foundation),
#           Global Mean / DoW Mean / MA (naive)
# Pick best cell per family on GLOBAL matrix.
# ---------------------------------------------------------------------
families = ['mlp_m5lags','lgb_m5lags','tft','chronos_bolt',
            'global_mean','dow_mean','ma_k21']
selected = []
family_best = {}
for fc in families:
    sub = matgen[matgen.forecaster==fc].sort_values('wape_h_med')
    if len(sub) == 0:
        continue
    family_best[fc] = sub.iloc[0]['cell']
    selected.append(sub.iloc[0]['cell'])

# Also track best per quartile for halos
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
# Plot 1: line chart with all selected cells
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 9))

# Assign distinct colors and markers
cmap = plt.get_cmap('tab10')
colors = {cell: cmap(i % 10) for i, cell in enumerate(selected)}
markers = ['o','D','s','v','^','P','X','*','d','<']

quartiles = ['Q1','Q2','Q3','Q4']
x = np.arange(4)

# Mark best per quartile with halos
best_cells_per_q = set(best_per_q.values())
for q_idx, q in enumerate(quartiles):
    bp = best_per_q[q]
    if bp in pivot.index:
        ax.scatter(x[q_idx], pivot.loc[bp, q], s=600, facecolors='none',
                   edgecolor='gold', linewidth=3, zorder=2)

for i, cell in enumerate(selected):
    vals = pivot.loc[cell].values
    ax.plot(x, vals, marker=markers[i % len(markers)], markersize=14,
            linewidth=2.5, color=colors[cell], label=cell,
            markerfacecolor=colors[cell], markeredgecolor='black',
            markeredgewidth=1.2, zorder=4)

ax.set_xticks(x)
ax.set_xticklabels(['Q1\n(basso vol)', 'Q2', 'Q3', 'Q4\n(alto vol)'], fontsize=13)
ax.set_ylabel('WAPE_h median (lower = better)', fontsize=14)
ax.set_xlabel('Volume quartile', fontsize=14)
ax.set_title('RQ3 — Crossover: best cell depends on volume regime',
             fontsize=15, pad=12)
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=11, framealpha=0.95,
          title='Selected cells\n(gold = best in that quartile)', title_fontsize=11)
ax.tick_params(axis='y', labelsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
out_fig = f'{FIG_DIR}/fig_rq3_crossover.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out_fig}')

# Save table for paper
pivot.to_parquet(f'{RESULTS_DIR}/rq3_crossover_table.parquet')
print('Saved: rq3_crossover_table.parquet')
print('\nDONE')
