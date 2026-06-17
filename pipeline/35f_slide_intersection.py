"""
35f_slide_intersection.py — Slide-ready figure: 11 cells doubly Pareto-optimal.

Layout slide 16:9:
  - Main panel: scatter (mean rank, |WPE|) con highlight delle 11 cells intersection
  - Side panel: tabella riassuntiva con i 3 archetypes + decision tree

Output: fig_slide_intersection.png (1600x900 px, slide-ready)
"""
import os, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patheffects as pe

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# Load matrix + Friedman ranks
print('Loading data...')
mat = pd.read_parquet(os.path.join(RESULTS_DIR, 'hpo_matrix_pareto.parquet'))
fr = pd.read_parquet(os.path.join(RESULTS_DIR, 'friedman_nemenyi_ranks.parquet'))
mat = mat.merge(fr[['cell', 'mean_rank', 'cd_indistinguishable']], on='cell', how='left')


def pareto_mask(x, y):
    n = len(x); is_par = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                is_par[i] = False; break
    return is_par


mat['pareto_wape'] = pareto_mask(mat['wape_h_med'].values, mat['abs_wpe_med'].values)
mat['pareto_rank'] = pareto_mask(mat['mean_rank'].values, mat['abs_wpe_med'].values)

intersection = mat[mat.pareto_wape & mat.pareto_rank].sort_values('mean_rank').reset_index(drop=True)
print(f'Intersection: {len(intersection)} cells')

# Friedman best (gold star)
friedman_best_cell = fr.iloc[0]['cell']
friedman_best = mat[mat.cell == friedman_best_cell].iloc[0]

# Archetype classification by family
def archetype(row):
    fc = row.forecaster
    if fc == 'tft':
        return ('TFT', '#7b3294')
    if fc in ('global_mean', 'dow_mean', 'ma_k21'):
        return ('Naive aggregato', '#2ca02c')
    if fc == 'timesfm':
        return ('Foundation', '#b15928')
    if fc == 'mlp_m5lags':
        return ('MLP_M5', '#4575b4')
    return ('Other', 'gray')


intersection['archetype'] = intersection.apply(lambda r: archetype(r)[0], axis=1)
intersection['color'] = intersection.apply(lambda r: archetype(r)[1], axis=1)

# Short label for plotting
intersection['short'] = intersection['cell'].apply(
    lambda c: c.split('__')[0][:8] + '__' + c.split('__')[1][:4]
)

# Build figure 16:9
fig = plt.figure(figsize=(20, 11.25))
gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.18)

# ============================================================================
# LEFT panel: scatter plot
# ============================================================================
ax = fig.add_subplot(gs[0])

# All cells faded
others = mat[~(mat.pareto_wape & mat.pareto_rank)]
ax.scatter(others.mean_rank, others.abs_wpe_med,
           c='lightgray', s=50, alpha=0.5, edgecolor='white', linewidth=0.4, zorder=2,
           label='Dominated / single-Pareto-only')

# Friedman best (also annotate as gold star for reference, even if not in intersection)
ax.scatter([friedman_best.mean_rank], [friedman_best.abs_wpe_med], marker='*',
           s=550, c='gold', edgecolor='black', linewidth=1.6, zorder=8,
           label=f'★ Friedman best ({friedman_best.cell.split("__")[0]}__MLP_M5)')

# Intersection cells, colored by archetype, big markers
for _, r in intersection.iterrows():
    ax.scatter([r.mean_rank], [r.abs_wpe_med],
               s=380, c=r.color, alpha=0.9, edgecolor='black', linewidth=1.5, zorder=6)

# Frontier line through intersection (Pareto sorted by mean rank)
inter_sorted = intersection.sort_values('mean_rank')
ax.plot(inter_sorted.mean_rank, inter_sorted.abs_wpe_med, '-', c='black',
        lw=1.5, alpha=0.5, zorder=3)

# Labels on intersection cells
for _, r in intersection.iterrows():
    txt = r.cell.replace('mlp_m5lags', 'MLP_M5').replace('lgb_m5lags', 'LGB_M5')
    txt = txt.replace('chronos_bolt', 'Chronos').replace('global_mean', 'GlobalMean')
    txt = txt.replace('dow_mean', 'DoWMean').replace('ma_k21', 'MA_K21')
    txt = txt.replace('timesfm', 'TimesFM').replace('tft', 'TFT')
    ax.annotate(txt, (r.mean_rank, r.abs_wpe_med),
                xytext=(6, 6), textcoords='offset points',
                fontsize=8.5, fontweight='bold', color='black',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

ax.set_xlabel('Friedman mean rank (lower = better accuracy)', fontsize=13)
ax.set_ylabel('|WPE_h median| (lower = less bias)', fontsize=13)
ax.set_title(f'11 cells doubly Pareto-optimal (intersezione delle due frontiere)',
             fontsize=14, pad=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(15, 110)
ax.set_ylim(-0.03, 1.0)

# Custom legend for archetypes
import matplotlib.lines as mlines
arch_handles = [
    mlines.Line2D([], [], color='#7b3294', marker='o', linestyle='None',
                  markeredgecolor='black', markersize=12, label='TFT (4 cells)'),
    mlines.Line2D([], [], color='#2ca02c', marker='o', linestyle='None',
                  markeredgecolor='black', markersize=12, label='Naive aggregato (6 cells)'),
    mlines.Line2D([], [], color='#b15928', marker='o', linestyle='None',
                  markeredgecolor='black', markersize=12, label='Foundation/TimesFM (1 cell)'),
    mlines.Line2D([], [], color='gold', marker='*', linestyle='None',
                  markeredgecolor='black', markersize=16,
                  label='Friedman best (NOT in intersection)'),
    mlines.Line2D([], [], color='lightgray', marker='o', linestyle='None',
                  markersize=8, label='Other cells (dominated)'),
]
ax.legend(handles=arch_handles, loc='upper right', fontsize=11, framealpha=0.95)

# ============================================================================
# RIGHT panel: archetype summary + decision tree
# ============================================================================
ax2 = fig.add_subplot(gs[1])
ax2.axis('off')

# Title on the right
ax2.text(0.5, 0.96, 'Decision Tree per il deployment',
         ha='center', va='top', fontsize=15, fontweight='bold',
         transform=ax2.transAxes)
ax2.text(0.5, 0.92, '(safest = doppia ottimalità Pareto)',
         ha='center', va='top', fontsize=11, style='italic', color='gray',
         transform=ax2.transAxes)

# 3 archetype boxes
archetype_data = [
    ('ACCURACY-extreme', '#7b3294',
     ['dlinear__TFT', 'seasonal_naive__TFT'],
     ['WAPE ~ 0.98', '|WPE| ~ 0.85', 'rank 29-33'],
     'Per accuracy KPI con bias control medio'),
    ('KNEE (balanced)', '#2ca02c',
     ['mediana_glob__MA_K21', 'mediana_cond__MA_K21'],
     ['WAPE ~ 1.11-1.12', '|WPE| ~ 0.13-0.15', 'rank 55-58'],
     'Trade-off bilanciato accuracy/bias'),
    ('BIAS-extreme', '#b15928',
     ['media_cond__MA_K21', 'linear_interp__TimesFM'],
     ['WAPE ~ 1.14-1.29', '|WPE| ~ 0.06-0.07', 'rank 66-98'],
     'Per inventory critica e bias-bound SLA'),
]

box_height = 0.21
for i, (title, color, cells, vals, use_case) in enumerate(archetype_data):
    y_top = 0.83 - i * 0.27
    # Header box
    box = FancyBboxPatch((0.02, y_top - 0.04), 0.96, 0.05,
                          boxstyle='round,pad=0.01',
                          facecolor=color, alpha=0.6, edgecolor='black',
                          transform=ax2.transAxes, linewidth=1.2)
    ax2.add_patch(box)
    ax2.text(0.5, y_top - 0.015, title, ha='center', va='center',
             fontsize=13, fontweight='bold', color='white',
             transform=ax2.transAxes)
    # Content
    cells_txt = '\n'.join(f'  • {c}' for c in cells)
    ax2.text(0.04, y_top - 0.07, 'Cells esempio:', fontsize=10, fontweight='bold',
             color='black', transform=ax2.transAxes)
    ax2.text(0.04, y_top - 0.105, cells_txt, fontsize=10,
             family='monospace', color='black', transform=ax2.transAxes,
             va='top')
    # Numbers
    vals_txt = '   '.join(vals)
    ax2.text(0.04, y_top - 0.165, f'  {vals_txt}', fontsize=9,
             color='#444444', transform=ax2.transAxes, va='top')
    # Use case
    ax2.text(0.04, y_top - 0.205, f'→ {use_case}', fontsize=10.5,
             style='italic', color='black', transform=ax2.transAxes, va='top')

# Bottom finding
ax2.text(0.5, 0.04,
         'ATTENZIONE: MLP_M5 (Friedman best) NON e\' nell\'intersezione ->\n'
         'massima accuracy ma bias estremo |WPE| >= 0.86',
         ha='center', va='bottom', fontsize=11, fontweight='bold', color='#d73027',
         transform=ax2.transAxes,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#fff5e6',
                   edgecolor='#d73027', linewidth=1.5))

# Big title at top
fig.suptitle('Doubly Pareto-optimal cells: 11 safest choices per il deployment',
             fontsize=18, fontweight='bold', y=0.985)

plt.tight_layout(rect=[0, 0, 1, 0.965])
out = os.path.join(FIG_DIR, 'fig_slide_intersection.png')
plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
