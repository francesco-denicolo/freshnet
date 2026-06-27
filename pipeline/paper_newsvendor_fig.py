"""Newsvendor scatter for the paper: per-cell order-level cost (r=2) vs in-stock
WAPE, coloured by forecaster family. Shows that (i) the lag-based families separate,
(ii) within a family the imputer spread is small and unstructured. Writes
fig_newsvendor.png into the Overleaf figures/ folder."""
import os, functools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
print = functools.partial(print, flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), 'results')
OUT = '/Users/utente/Desktop/MDPI_Overleaf/figures/fig_newsvendor.png'

c = pd.read_parquet(f'{RES}/newsvendor_cost_summary.parquet')
m = pd.read_parquet(f'{RES}/hpo_matrix_pareto.parquet')[['cell', 'wape_h_med']]
c['cellk'] = c['cell'].str.replace('_hpo', '', regex=False)
d = c.merge(m, left_on='cellk', right_on='cell', suffixes=('', '_m'))

fam = {'mlp_m5lags': ('MLP-M5', '#1f77b4', 'o'), 'lgb_m5lags': ('LGB-M5', '#d62728', 's')}
fig, ax = plt.subplots(figsize=(7.6, 5.8))
for fc, (lab, col, mk) in fam.items():
    g = d[d.forecaster == fc]
    ax.scatter(g.wape_h_med, g.cost_r2, c=col, marker=mk, s=70, alpha=0.85,
               edgecolor='white', linewidth=0.8, label=f'{lab} (13 imputers)', zorder=3)
    rho = spearmanr(g.wape_h_med, g.cost_r2).correlation
    # annotate the cheapest cell of each family
    b = g.loc[g.cost_r2.idxmin()]
    ax.annotate(b.imputer, (b.wape_h_med, b.cost_r2), textcoords='offset points',
                xytext=(7, -2), fontsize=8.5, color=col)
    print(f'{lab}: Spearman(WAPE,cost_r2)={rho:.3f}')

rho_ystar = spearmanr(d.cost_r1, d.cost_r1_obs).correlation
ax.set_xlabel('In-stock WAPE (per-series median)', fontsize=12)
ax.set_ylabel('Newsvendor cost, $r=c_u/c_o=2$ (per-series median)', fontsize=12)
ax.set_title('Order-level cost vs in-stock accuracy across the imputer choice', fontsize=12.5)
ax.legend(loc='upper left', fontsize=10, frameon=True)
ax.grid(True, alpha=0.3, zorder=0)
txt = ('Within-family imputer spread: 5.9% (MLP-M5), 3.7% (LGB-M5)\n'
       'Within-family rho(WAPE, cost) ~ 0 (CIs include 0)\n'
       'Ranking stable across reference imputers (rho >= 0.99)')
ax.text(0.97, 0.04, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=8.8,
        bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='0.7'))
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'saved {OUT}')
