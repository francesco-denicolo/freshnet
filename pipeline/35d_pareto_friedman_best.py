"""
35d_pareto_friedman_best.py — Pareto frontier con "best" = Friedman (non median WAPE).

Variante di fig_pareto_hpo.png che:
  - Mantiene gli assi (WAPE_h_med, |WPE_h_med|) — interpretabili per practitioner.
  - Sostituisce la gold star da "median-WAPE best" a "Friedman best".
  - Mantiene equiv-set (Nemenyi CD), knee point, min |WPE| come ring colorati.

Output: fig_pareto_friedman.png (figura originale fig_pareto_hpo.png invariata).
"""
import os, functools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# Same per-forecaster styling as script 35
FC_COLORS = {
    'lgb_nolags':'#fdae61', 'lgb_m5lags':'#f46d43',
    'mlp_nolags':'#74add1', 'mlp_m5lags':'#4575b4',
    'tft':'#7b3294', 'chronos_bolt':'#d73027', 'timesfm':'#b15928',
    'global_mean':'#2ca02c', 'dow_mean':'#bcbd22', 'ma_k56':'#17becf',
}
FC_MARKERS = {
    'lgb_nolags':'^', 'lgb_m5lags':'^',
    'mlp_nolags':'D', 'mlp_m5lags':'D',
    'tft':'o', 'chronos_bolt':'P', 'timesfm':'*',
    'global_mean':'s', 'dow_mean':'X', 'ma_k56':'v',
}
FC_LABELS = {
    'lgb_nolags':'LGB_nolags', 'lgb_m5lags':'LGB_M5',
    'mlp_nolags':'MLP_nolags', 'mlp_m5lags':'MLP_M5',
    'tft':'TFT', 'chronos_bolt':'Chronos-bolt', 'timesfm':'TimesFM',
    'global_mean':'Global Mean', 'dow_mean':'DoW Mean', 'ma_k56':'MA (K=56)',
}

# ----------------------------------------------------------------------
# Load matrix + Friedman ranks
# ----------------------------------------------------------------------
print('Loading data...')
mat = pd.read_parquet(os.path.join(RESULTS_DIR, 'hpo_matrix_pareto.parquet'))
fr = pd.read_parquet(os.path.join(RESULTS_DIR, 'friedman_nemenyi_ranks.parquet'))
mat = mat.merge(fr[['cell', 'mean_rank', 'cd_indistinguishable']], on='cell', how='left')

# Friedman best (not median best!)
friedman_best_cell = fr.iloc[0]['cell']
friedman_best = mat[mat.cell == friedman_best_cell].iloc[0]
median_best = mat.sort_values('wape_h_med').iloc[0]
print(f'  Friedman best (mean rank {friedman_best.mean_rank:.2f}): {friedman_best.cell}')
print(f'  Median-WAPE best (WAPE_h_med {median_best.wape_h_med:.4f}): {median_best.cell}')

cd_equiv = set(mat[mat.cd_indistinguishable]['cell'])

# Pareto frontier (on WAPE, |WPE|) — invariata
def pareto_mask(x, y):
    n = len(x); is_par = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if x[j] <= x[i] and y[j] <= y[i] and (x[j] < x[i] or y[j] < y[i]):
                is_par[i] = False; break
    return is_par

mat['pareto'] = pareto_mask(mat['wape_h_med'].values, mat['abs_wpe_med'].values)
par = mat[mat.pareto].sort_values('wape_h_med')
print(f'  Pareto-optimal: {mat.pareto.sum()}/{len(mat)}')

# Knee + min |WPE|
par_n = par.copy()
par_n['x_n'] = (par_n.wape_h_med - par_n.wape_h_med.min()) / (par_n.wape_h_med.max() - par_n.wape_h_med.min() + 1e-9)
par_n['y_n'] = (par_n.abs_wpe_med - par_n.abs_wpe_med.min()) / (par_n.abs_wpe_med.max() - par_n.abs_wpe_med.min() + 1e-9)
par_n['dist'] = np.sqrt(par_n.x_n**2 + par_n.y_n**2)
knee = par_n.sort_values('dist').iloc[0]
min_wpe = par.sort_values('abs_wpe_med').iloc[0]
print(f'  Knee: {knee.cell}, Min |WPE|: {min_wpe.cell}')

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(20, 12))
xmin, xmax = mat.wape_h_med.min() - 0.005, mat.wape_h_med.max() + 0.012
ymin, ymax = mat.abs_wpe_med.min() - 0.03, mat.abs_wpe_med.max() + 0.03

ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

# Dominated (pale)
dom = mat[~mat.pareto]
for _, r in dom.iterrows():
    ax.scatter([r.wape_h_med], [r.abs_wpe_med],
               c=FC_COLORS.get(r.forecaster, '#cccccc'),
               marker=FC_MARKERS.get(r.forecaster, 'o'),
               s=70, alpha=0.55, edgecolor='gray', linewidth=0.4, zorder=2)

# Pareto (saturated)
for _, r in par.iterrows():
    ax.scatter([r.wape_h_med], [r.abs_wpe_med],
               c=FC_COLORS.get(r.forecaster, '#000000'),
               marker=FC_MARKERS.get(r.forecaster, 'o'),
               s=200, edgecolor='black', linewidth=1.8, zorder=4)

# Frontier line
ax.plot(par.wape_h_med, par.abs_wpe_med, '-', c='black', lw=1.5, alpha=0.5, zorder=3)

# CD-equiv set: orange ring around each
for _, r in mat[mat.cd_indistinguishable].iterrows():
    ax.scatter([r.wape_h_med], [r.abs_wpe_med], marker='o',
               s=420, facecolors='none', edgecolor='#ff7f0e',
               linewidth=2.5, linestyle='--', zorder=5)

# HIGHLIGHTS (Friedman best replaces median best!)
# Gold star: Friedman best
ax.scatter([friedman_best.wape_h_med], [friedman_best.abs_wpe_med], marker='*',
           s=550, c='gold', edgecolor='black', linewidth=1.8, zorder=7)
# Knee
ax.scatter([knee.wape_h_med], [knee.abs_wpe_med], marker='o',
           s=380, facecolors='none', edgecolor='#2ca02c', linewidth=3, zorder=6)
# Min |WPE|
ax.scatter([min_wpe.wape_h_med], [min_wpe.abs_wpe_med], marker='o',
           s=380, facecolors='none', edgecolor='#1f77b4', linewidth=3, zorder=6)
# Median-WAPE best (secondary annotation, smaller, gray)
ax.scatter([median_best.wape_h_med], [median_best.abs_wpe_med], marker='*',
           s=200, c='lightgray', edgecolor='black', linewidth=1.2, zorder=6)

ax.set_xlabel('WAPE_h median (lower = better accuracy)', fontsize=15)
ax.set_ylabel('|WPE_h median| (lower = less bias)', fontsize=15)
ax.set_title(
    f'Pareto frontier WAPE × |WPE|  —  '
    f'{len(mat)} cells, {mat.pareto.sum()} Pareto-optimal\n'
    f'★ gold = Friedman best  ·  ★ light = lowest median WAPE  ·  '
    f'⊙ orange = CD-equivalent (Nemenyi)',
    fontsize=14, pad=14)
ax.grid(True, alpha=0.25, linestyle='--')

# Legend
legend_handles = []
for fc in ['lgb_nolags','lgb_m5lags','mlp_nolags','mlp_m5lags','tft',
           'chronos_bolt','timesfm','global_mean','dow_mean','ma_k56']:
    if fc in mat.forecaster.unique():
        legend_handles.append(
            mlines.Line2D([], [], color=FC_COLORS[fc], marker=FC_MARKERS[fc],
                          linestyle='None', markeredgecolor='black', markersize=10,
                          label=FC_LABELS[fc])
        )

legend_handles += [
    mlines.Line2D([], [], color='gold', marker='*', linestyle='None',
                  markeredgecolor='black', markersize=16,
                  label=f'★ Friedman best: {friedman_best.imputer}__{FC_LABELS.get(friedman_best.forecaster, friedman_best.forecaster)}'),
    mlines.Line2D([], [], color='lightgray', marker='*', linestyle='None',
                  markeredgecolor='black', markersize=10,
                  label=f'★ Median-WAPE best (reference): {median_best.imputer}__{FC_LABELS.get(median_best.forecaster, median_best.forecaster)}'),
    mlines.Line2D([], [], color='#ff7f0e', marker='o', linestyle='--',
                  markerfacecolor='none', markeredgewidth=2.5, markersize=12,
                  label=f'⊙ Nemenyi CD-equivalent to Friedman best (n={len(cd_equiv)})'),
    mlines.Line2D([], [], color='#2ca02c', marker='o', linestyle='None',
                  markerfacecolor='none', markeredgewidth=2.5, markersize=12,
                  label=f'● Knee: {knee.imputer}__{FC_LABELS.get(knee.forecaster, knee.forecaster)}'),
    mlines.Line2D([], [], color='#1f77b4', marker='o', linestyle='None',
                  markerfacecolor='none', markeredgewidth=2.5, markersize=12,
                  label=f'● Min |WPE|: {min_wpe.imputer}__{FC_LABELS.get(min_wpe.forecaster, min_wpe.forecaster)}'),
]

ax.legend(handles=legend_handles, loc='center left',
          bbox_to_anchor=(1.02, 0.5), fontsize=12, framealpha=0.95)
ax.tick_params(axis='both', labelsize=13)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
out_path = f'{FIG_DIR}/fig_pareto_friedman.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out_path}')
print('Original fig_pareto_hpo.png untouched.')
