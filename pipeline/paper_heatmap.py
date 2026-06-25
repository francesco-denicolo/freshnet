"""Publication-quality full-matrix WAPE heatmap for the paper (Section RQ3).
Reads hpo_matrix_pareto.parquet (113 cells) + friedman_nemenyi_ranks.parquet
(best cell + CD equivalence set). Writes fig_results_heatmap.png into the
Overleaf figures/ folder. Friedman-best outlined in blue, equiv set dashed."""
import os, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RES = os.path.join(os.path.dirname(__file__), 'results')
OUT = '/Users/utente/Desktop/MDPI_Overleaf/figures/fig_results_heatmap.png'

# Imputer rows grouped by family; forecaster cols ordered naive -> lag-free -> lag ML -> deep -> foundation
IMP_ORDER = ['no_imp', 'media_glob', 'media_cond', 'mediana_glob', 'mediana_cond',
             'forward_fill', 'seasonal_naive', 'linear_interp',
             'lgb', 'dlinear', 'saits', 'itransformer', 'timesnet', 'imputeformer']
IMP_LABEL = {'no_imp': 'No imputation', 'media_glob': 'Mean (global)',
             'media_cond': 'Mean (cond.)', 'mediana_glob': 'Median (global)',
             'mediana_cond': 'Median (cond.)', 'forward_fill': 'Forward fill',
             'seasonal_naive': 'Seasonal naive', 'linear_interp': 'Linear interp.',
             'lgb': 'LGB imputer', 'dlinear': 'DLinear', 'saits': 'SAITS',
             'itransformer': 'iTransformer', 'timesnet': 'TimesNet',
             'imputeformer': 'ImputeFormer'}
FC_ORDER = ['global_mean', 'dow_mean', 'ma_k56', 'lgb_nolags', 'mlp_nolags',
            'lgb_m5lags', 'mlp_m5lags', 'tft', 'chronos_bolt', 'timesfm']
FC_SHORT = {'global_mean': 'Global\nMean', 'dow_mean': 'DoW\nMean', 'ma_k56': 'MA\n(K=56)',
            'lgb_nolags': 'LGB\n(no lag)', 'mlp_nolags': 'MLP\n(no lag)',
            'lgb_m5lags': 'LGB\n(M5)', 'mlp_m5lags': 'MLP\n(M5)', 'tft': 'TFT',
            'chronos_bolt': 'Chronos', 'timesfm': 'TimesFM'}

mat = pd.read_parquet(f'{RES}/hpo_matrix_pareto.parquet')
fr = pd.read_parquet(f'{RES}/friedman_nemenyi_ranks.parquet')
best_cell = fr.iloc[0]['cell']
equiv = set(fr[fr.cd_indistinguishable]['cell'])
print(f'Friedman best: {best_cell} | equiv set: {sorted(equiv)} | cells: {len(mat)}')

pivot = mat.pivot(index='imputer', columns='forecaster', values='wape_h_med')
pivot = pivot.reindex(index=IMP_ORDER, columns=FC_ORDER)
vals = pivot.values.astype(float)
vmin, vmax = 0.97, 1.08

fig, ax = plt.subplots(figsize=(11.5, 9.0))
im = ax.imshow(np.clip(vals, vmin, vmax), cmap='RdYlGn_r', aspect='auto', vmin=vmin, vmax=vmax)

for i, imp in enumerate(IMP_ORDER):
    for j, fc in enumerate(FC_ORDER):
        v = vals[i, j]
        if np.isnan(v):
            ax.text(j, i, '—', ha='center', va='center', fontsize=10, color='0.5')
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, facecolor='0.92', edgecolor='white', zorder=1))
            continue
        cell = f'{imp}__{fc}'
        is_best, is_equiv = cell == best_cell, cell in equiv
        txtcol = 'white' if (v > 1.045 or v < 0.978) else 'black'
        ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=9.5,
                color=txtcol, fontweight='bold' if (is_best or is_equiv) else 'normal')
        if is_best:
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='#0033cc', lw=3.2, zorder=4))
        elif is_equiv:
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='#0033cc', lw=2.0, ls='--', zorder=3))

ax.set_xticks(range(len(FC_ORDER)))
ax.set_xticklabels([FC_SHORT[c] for c in FC_ORDER], fontsize=10.5)
ax.set_yticks(range(len(IMP_ORDER)))
ax.set_yticklabels([IMP_LABEL[c] for c in IMP_ORDER], fontsize=10.5)
ax.set_xlabel('Forecaster', fontsize=12, labelpad=8)
ax.set_ylabel('Imputer', fontsize=12)
# family separators on the forecaster axis (after naive | after lag-free | after lag-ML | after deep)
for x in (2.5, 4.5, 6.5, 7.5):
    ax.axvline(x, color='0.25', lw=1.2)
ax.set_xticks(np.arange(-0.5, len(FC_ORDER), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(IMP_ORDER), 1), minor=True)
ax.grid(which='minor', color='white', lw=0.8)
ax.tick_params(which='minor', length=0)

cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02, extend='both')
cbar.set_label('Median WAPE (in-stock test hours)', fontsize=11)
ax.set_title('Forecasting accuracy across the imputer $\\times$ forecaster matrix',
             fontsize=13, pad=12)
# legend for the outlines
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], color='#0033cc', lw=3.2, label='Best cell (Friedman)'),
       Line2D([0], [0], color='#0033cc', lw=2.0, ls='--', label='Within critical difference')]
ax.legend(handles=leg, loc='upper left', bbox_to_anchor=(0.0, -0.09), ncol=2,
          frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'saved {OUT}')
