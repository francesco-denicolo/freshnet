"""
35c_heatmap_mean_rank.py — Heatmap by Friedman mean rank (Options B & C).

Per coerenza con il framework decisionale del paper (Friedman+W+CD,
Sez. 1.1), colora le celle per *mean rank* invece che per WAPE_h_med.

Genera due figure:
  - Option B: fig_heatmap_general_no_imputeformer_meanrank.png
              colore + testo = mean rank Friedman
  - Option C: fig_heatmap_general_no_imputeformer_meanrank_wape.png
              colore = mean rank, testo a 2 righe = rank + WAPE_h_med

Riga 'imputeformer' esclusa (come la heatmap precedente).
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

# Load matrix + Friedman ranks
print('Loading data...')
mat = pd.read_parquet(os.path.join(RESULTS_DIR, 'hpo_matrix_pareto.parquet'))
fr = pd.read_parquet(os.path.join(RESULTS_DIR, 'friedman_nemenyi_ranks.parquet'))
mat = mat[mat.imputer != 'imputeformer'].copy()

# Merge mean_rank into mat
mat = mat.merge(fr[['cell', 'mean_rank', 'cd_indistinguishable']], on='cell', how='left')

# Best = lowest mean rank
best_row = mat.sort_values('mean_rank').iloc[0]
cd_equiv = set(mat[mat.cd_indistinguishable]['cell'])
print(f'  Friedman best: {best_row.cell}, mean_rank={best_row.mean_rank:.2f}')
print(f'  CD-equiv set: {len(cd_equiv)} cells')

# Pivots
pivot_rank = mat.pivot(index='imputer', columns='forecaster', values='mean_rank').reindex(
    index=IMP_ORDER, columns=FC_ORDER)
pivot_wape = mat.pivot(index='imputer', columns='forecaster', values='wape_h_med').reindex(
    index=IMP_ORDER, columns=FC_ORDER)

# Colormap calibration based on rank range
vmin = float(pivot_rank.min().min())
vmax = float(pivot_rank.max().max())
print(f'  Mean rank range: [{vmin:.2f}, {vmax:.2f}]')


def draw_heatmap(use_wape_text, out_name, title):
    """Draw heatmap colored by mean rank. Text = rank only (B) or rank+WAPE (C)."""
    fig, ax = plt.subplots(figsize=(13, 9.5))
    # Color: low rank = good = green; high rank = bad = red
    im = ax.imshow(pivot_rank.values, cmap='RdYlGn_r', aspect='auto',
                   vmin=vmin, vmax=vmax)
    for i, imp in enumerate(IMP_ORDER):
        for j, fc in enumerate(FC_ORDER):
            v = pivot_rank.iloc[i, j]
            if np.isnan(v): continue
            cell = f'{imp}__{fc}'
            is_best = (cell == best_row.cell)
            is_equiv = cell in cd_equiv

            color = 'white' if (v - vmin) / (vmax - vmin) > 0.75 else 'black'
            fw = 'bold' if (is_best or is_equiv) else 'normal'

            if use_wape_text:
                wape_val = pivot_wape.iloc[i, j]
                txt = f'{v:.1f}\n({wape_val:.3f})'
                fontsize = 8
            else:
                txt = f'{v:.1f}'
                fontsize = 11

            ax.text(j, i, txt, ha='center', va='center', fontsize=fontsize,
                    color=color, fontweight=fw)
            if is_best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            edgecolor='blue', lw=3.5, zorder=3))
            elif is_equiv:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            edgecolor='orange', lw=2.2,
                                            linestyle='--', zorder=2))

    ax.set_xticks(range(len(FC_ORDER)))
    ax.set_xticklabels([FC_SHORT[c] for c in FC_ORDER], fontsize=12, rotation=15)
    ax.set_yticks(range(len(IMP_ORDER)))
    ax.set_yticklabels(IMP_ORDER, fontsize=12)
    ax.set_title(title, fontsize=12, pad=14)
    ax.set_xlabel('Forecaster', fontsize=13)
    ax.set_ylabel('Imputer', fontsize=13)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Friedman mean rank (lower = better)', fontsize=11)
    plt.tight_layout()
    out_path = f'{FIG_DIR}/{out_name}'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


# Option B: only mean rank
print('\nOption B — mean rank coloring + mean rank text...')
draw_heatmap(
    use_wape_text=False,
    out_name='fig_heatmap_general_no_imputeformer_meanrank.png',
    title=(f'Heatmap Friedman mean rank (NO imputeformer) — '
           f'blue=Friedman best ({best_row.cell}), '
           f'orange dashed=CD-equivalent set (n={len(cd_equiv)})')
)

# Option C: mean rank + WAPE
print('\nOption C — mean rank coloring + (rank, WAPE) double text...')
draw_heatmap(
    use_wape_text=True,
    out_name='fig_heatmap_general_no_imputeformer_meanrank_wape.png',
    title=(f'Heatmap Friedman mean rank + WAPE_h_med (NO imputeformer) — '
           f'blue=Friedman best, orange dashed=CD-equiv (n={len(cd_equiv)})')
)

print('\nDONE')
