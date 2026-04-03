"""
08c_mlp_instock_plots.py — Boxplot WAPE/WPE in-stock per val e test
====================================================================
Confronto tutti i baseline, metriche solo su ore in-stock.

Eseguire con: freshnet/bin/python notebooks/08c_mlp_instock_plots.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

all_baselines = {
    'Naive': 'naive_direct',
    'MA K=14': 'ma_direct_K14',
    'Global Mean': 'global_mean',
    'DoW Mean': 'dow_mean',
    'LGB (A)': 'lgb_a',
    'LGB (F)': 'lgb_f',
    '2-Stage LGB': 'twostage_lgb',
    'MLP (A)': 'mlp',
    'MLP (F)': 'mlp_f',
    '2-Stage MLP': 'twostage_mlp',
    'PINN (F)': 'pinn',
    'PINN (A)': 'pinn_a',
}
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974',
          '#64B5CD', '#DD8452', '#937860', '#DA8BC3', '#E24A33', '#348ABD', '#8B4513']

# ---------------------------------------------------------------------------
# Figure: 2×2 boxplot (WAPE_instock + WPE_instock) × (val + test)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(26, 10))
fig.suptitle('Confronto Tutti i Modelli — Metriche In-Stock (per-serie)', fontsize=15, y=0.98)

for j, split in enumerate(['val', 'test']):
    # --- WAPE instock ---
    ax = axes[0, j]
    box_data = []
    box_labels = []
    box_colors = []
    medians = []

    for k, (label, prefix) in enumerate(all_baselines.items()):
        path = os.path.join(RESULTS_DIR, f'{prefix}_{split}_per_series.parquet')
        if not os.path.exists(path):
            continue
        ps = pd.read_parquet(path)
        vals = ps['wape_instock'].dropna()
        q99 = vals.quantile(0.99)
        box_data.append(vals.clip(upper=q99).values)
        box_labels.append(label)
        box_colors.append(colors[k])
        medians.append(vals.median())

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for ml in bp['medians']:
        ml.set_color('red')
        ml.set_linewidth(2)
    for k, med in enumerate(medians):
        ax.text(k + 1, med + 0.01, f'{med:.3f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='red')

    ax.set_title(f'WAPE in-stock — {split}', fontsize=13)
    ax.set_ylabel('WAPE in-stock')
    ax.tick_params(axis='x', rotation=25)

    # --- WPE instock ---
    ax = axes[1, j]
    box_data = []
    medians = []

    for k, (label, prefix) in enumerate(all_baselines.items()):
        path = os.path.join(RESULTS_DIR, f'{prefix}_{split}_per_series.parquet')
        if not os.path.exists(path):
            continue
        ps = pd.read_parquet(path)
        vals = ps['wpe_instock'].dropna()
        q01, q99 = vals.quantile(0.01), vals.quantile(0.99)
        box_data.append(vals.clip(lower=q01, upper=q99).values)
        medians.append(vals.median())

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for ml in bp['medians']:
        ml.set_color('red')
        ml.set_linewidth(2)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    for k, med in enumerate(medians):
        offset = 0.005 if med >= 0 else -0.005
        va = 'bottom' if med >= 0 else 'top'
        ax.text(k + 1, med + offset, f'{med:.4f}', ha='center', va=va,
                fontsize=8, fontweight='bold', color='red')

    ax.set_title(f'WPE in-stock — {split}', fontsize=13)
    ax.set_ylabel('WPE in-stock')
    ax.tick_params(axis='x', rotation=25)

fig.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(FIG_DIR, 'fig38_compare_instock_all_models.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Salvata: {out_path}')

# ---------------------------------------------------------------------------
# Tabella riepilogativa
# ---------------------------------------------------------------------------
print(f'\n{"Model":<24} {"Split":<6} {"WAPE_in med":>12} {"WPE_in med":>12}')
print('-' * 58)

for split in ['val', 'test']:
    for label, prefix in all_baselines.items():
        path = os.path.join(RESULTS_DIR, f'{prefix}_{split}_per_series.parquet')
        if not os.path.exists(path):
            continue
        ps = pd.read_parquet(path)
        wape_med = ps['wape_instock'].median()
        wpe_med = ps['wpe_instock'].median()
        print(f'{label:<24} {split:<6} {wape_med:>12.4f} {wpe_med:>12.4f}')
    print()
