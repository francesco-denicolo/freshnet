"""
35b_heatmap_no_imputeformer.py — Variante di fig_heatmap_general.png senza riga imputeformer.

Carica hpo_matrix_pareto.parquet e rigenera la heatmap escludendo l'imputer
'imputeformer'. La figura originale non viene modificata.
"""
import os, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# Same ordering as script 35 but without 'imputeformer'
IMP_ORDER = ['no_imp', 'media_glob', 'media_cond', 'mediana_glob', 'mediana_cond',
             'forward_fill', 'seasonal_naive', 'linear_interp', 'lgb',
             'dlinear', 'saits', 'itransformer', 'timesnet']
FC_ORDER = ['lgb_nolags', 'lgb_m5lags', 'mlp_nolags', 'mlp_m5lags', 'tft',
            'chronos_bolt', 'timesfm', 'global_mean', 'dow_mean', 'ma_k21']
FC_SHORT = {
    'lgb_nolags': 'LGB_nl', 'lgb_m5lags': 'LGB_M5', 'mlp_nolags': 'MLP_nl',
    'mlp_m5lags': 'MLP_M5', 'tft': 'TFT', 'chronos_bolt': 'Chron',
    'timesfm': 'TimesFM', 'global_mean': 'GM', 'dow_mean': 'DoW', 'ma_k21': 'MA21',
}

print('Loading hpo_matrix_pareto.parquet...')
mat = pd.read_parquet(os.path.join(RESULTS_DIR, 'hpo_matrix_pareto.parquet'))
mat_no = mat[mat.imputer != 'imputeformer'].copy()
print(f'  Total cells: {len(mat)}, after removing imputeformer: {len(mat_no)}')

# Best cell (WAPE_h_med) — keep using global best from full matrix for reference
best_row = mat.sort_values('wape_h_med').iloc[0]
print(f'  Reference best (full matrix): {best_row.cell} (WAPE_h_med={best_row.wape_h_med:.4f})')

# Re-derive equiv set from Friedman if available
fr_path = f'{RESULTS_DIR}/friedman_nemenyi_ranks.parquet'
if os.path.exists(fr_path):
    fr = pd.read_parquet(fr_path)
    cd_equiv = set(fr[fr.cd_indistinguishable]['cell'])
else:
    cd_equiv = set()

# Pivot
pivot = mat_no.pivot(index='imputer', columns='forecaster', values='wape_h_med')
pivot = pivot.reindex(index=IMP_ORDER, columns=FC_ORDER)

# Heatmap
fig, ax = plt.subplots(figsize=(13, 9.5))
im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0.95, vmax=1.20)
for i, imp in enumerate(IMP_ORDER):
    for j, fc in enumerate(FC_ORDER):
        v = pivot.iloc[i, j]
        if np.isnan(v): continue
        cell = f'{imp}__{fc}'
        is_best = (cell == best_row.cell)
        is_equiv = cell in cd_equiv
        text = f'{v:.3f}'
        color = 'white' if v > 1.10 or v < 0.98 else 'black'
        ax.text(j, i, text, ha='center', va='center', fontsize=11,
                color=color, fontweight='bold' if (is_best or is_equiv) else 'normal')
        if is_best:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor='blue', lw=3.5, zorder=3))
        elif is_equiv:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor='orange', lw=2.2, linestyle='--', zorder=2))

ax.set_xticks(range(len(FC_ORDER)))
ax.set_xticklabels([FC_SHORT[c] for c in FC_ORDER], fontsize=12, rotation=15)
ax.set_yticks(range(len(IMP_ORDER)))
ax.set_yticklabels(IMP_ORDER, fontsize=12)
ax.set_title(f'General heatmap WAPE_h_med (NO imputeformer) — '
             f'blue=best ({best_row.cell}), '
             f'orange dashed=Nemenyi CD-equivalent to Friedman best (n={len(cd_equiv)})',
             fontsize=12, pad=14)
ax.set_xlabel('Forecaster', fontsize=13)
ax.set_ylabel('Imputer', fontsize=13)
plt.colorbar(im, ax=ax, label='WAPE_h_med')
plt.tight_layout()

out = f'{FIG_DIR}/fig_heatmap_general_no_imputeformer.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
