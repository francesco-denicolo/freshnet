"""
38_rq5_loss_alignment.py — RQ5: MSE training loss vs MAE training loss
======================================================================
Compare WAPE_h_med across 24 (imputer, forecaster) cells trained with:
  - MSE training loss (legacy, pipeline/results/_mse_backup/)
  - MAE training loss (current, pipeline/results/)

Hypothesis: with MAE (aligned with WAPE metric), cross-imputer spread is
much smaller (the imputer choice "collapses" in importance).

Outputs:
  - pipeline/results/rq5_mae_vs_mse.parquet
  - pipeline/figures/fig_rq5_mae_vs_mse.png
"""
import os, functools, numpy as np, pandas as pd
from scipy import stats
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
MSE_DIR = os.path.join(RESULTS_DIR, '_mse_backup')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Load MSE backup + current (MAE)
# ---------------------------------------------------------------------
def parse_name(name):
    if '__' in name: return name.split('__', 1)
    return 'no_imp', name

def cell_metric(path):
    df = pd.read_parquet(path)
    return df['hourly_wape'].median(), df['hourly_wpe'].median(), df

print('1. Loading 24 MSE backup cells + current MAE...')
rows = []
import glob
for f in sorted(glob.glob(f'{MSE_DIR}/*_test_per_series.parquet')):
    name = os.path.basename(f).replace('_test_per_series.parquet','')
    imp, fc = parse_name(name)
    mse_wape, mse_wpe, _ = cell_metric(f)
    # Counterpart MAE: prefer _hpo_ if exists, else baseline _test_per_series
    mae_path_hpo = f'{RESULTS_DIR}/{name}_hpo_test_per_series.parquet'
    mae_path_plain = f'{RESULTS_DIR}/{name}_test_per_series.parquet'
    if os.path.exists(mae_path_hpo):
        mae_wape, mae_wpe, _ = cell_metric(mae_path_hpo)
        mae_kind = 'HPO'
    elif os.path.exists(mae_path_plain):
        mae_wape, mae_wpe, _ = cell_metric(mae_path_plain)
        mae_kind = 'baseline'
    else:
        continue
    rows.append({
        'cell': name, 'imputer': imp, 'forecaster': fc,
        'mse_wape_med': mse_wape, 'mse_wpe_med': mse_wpe,
        'mae_wape_med': mae_wape, 'mae_wpe_med': mae_wpe,
        'mae_kind': mae_kind,
        'delta_wape': mae_wape - mse_wape,
        'delta_wpe': mae_wpe - mse_wpe,
    })

mat = pd.DataFrame(rows).sort_values(['forecaster','imputer'])
mat.to_parquet(f'{RESULTS_DIR}/rq5_mae_vs_mse.parquet', index=False)
print(f'   {len(mat)} cells paired (MSE backup + MAE)')

# Forecasters covered
print(f'   Forecasters: {sorted(mat.forecaster.unique())}')

# ---------------------------------------------------------------------
# 2. Spread analysis per forecaster
# ---------------------------------------------------------------------
print('\n2. Cross-imputer spread per forecaster (lower = more equivalent imputers):')
spread_rows = []
for fc in mat.forecaster.unique():
    sub = mat[mat.forecaster == fc]
    sp_mse = sub.mse_wape_med.max() - sub.mse_wape_med.min()
    sp_mae = sub.mae_wape_med.max() - sub.mae_wape_med.min()
    print(f'   {fc:<15} MSE spread={sp_mse:.4f}, MAE spread={sp_mae:.4f}, compression={(1-sp_mae/sp_mse)*100:.1f}%')
    spread_rows.append({'forecaster': fc, 'mse_spread': sp_mse, 'mae_spread': sp_mae,
                        'compression_pct': (1-sp_mae/sp_mse)*100 if sp_mse > 0 else 0})
spread = pd.DataFrame(spread_rows)

# ---------------------------------------------------------------------
# 3. Figure: 2-panel (WAPE side-by-side + spread bars)
# ---------------------------------------------------------------------
print('\n3. Building figure...')
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Panel 1: scatter MSE_wape vs MAE_wape per cell, colored by forecaster
ax = axes[0]
fc_colors = {'lgb_m5lags':'#f46d43','mlp_m5lags':'#4575b4',
             'lgb_nolags':'#fdae61','mlp_nolags':'#74add1'}
for fc in mat.forecaster.unique():
    sub = mat[mat.forecaster == fc]
    ax.scatter(sub.mse_wape_med, sub.mae_wape_med,
               c=fc_colors.get(fc, 'gray'), s=180, edgecolor='black',
               linewidth=1.5, label=fc, alpha=0.85)
# Diagonal y=x
lo = min(mat.mse_wape_med.min(), mat.mae_wape_med.min()) - 0.02
hi = max(mat.mse_wape_med.max(), mat.mae_wape_med.max()) + 0.02
ax.plot([lo, hi], [lo, hi], '--', color='gray', alpha=0.5, lw=1, label='MSE = MAE')
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel('WAPE_h_med (MSE training loss)', fontsize=13)
ax.set_ylabel('WAPE_h_med (MAE training loss)', fontsize=13)
ax.set_title('(a) Per-cell shift MSE → MAE\n(points below diagonal = MAE improves WAPE)',
             fontsize=13, pad=10)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

# Panel 2: bar chart of spread per forecaster
ax = axes[1]
fc_labels = spread.forecaster.tolist()
x = np.arange(len(fc_labels))
w = 0.35
ax.bar(x - w/2, spread.mse_spread.values, w, label='MSE training', color='#d73027', edgecolor='black')
ax.bar(x + w/2, spread.mae_spread.values, w, label='MAE training', color='#1f77b4', edgecolor='black')
for i, comp in enumerate(spread.compression_pct.values):
    ax.text(x[i], max(spread.mse_spread.iloc[i], spread.mae_spread.iloc[i]) + 0.005,
            f'−{comp:.0f}%', ha='center', fontsize=11, fontweight='bold', color='#444')
ax.set_xticks(x)
ax.set_xticklabels(fc_labels, fontsize=12, rotation=15)
ax.set_ylabel('Cross-imputer spread (max − min WAPE_h_med)', fontsize=13)
ax.set_title('(b) Imputer choice impact: MAE compresses spread\n(lower bars = imputer choice less impactful)',
             fontsize=13, pad=10)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

fig.suptitle('RQ5 — Loss alignment: MAE training collapses imputer effect on WAPE',
             fontsize=15, y=1.005)
plt.tight_layout()
out_fig = f'{FIG_DIR}/fig_rq5_mae_vs_mse.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'   Saved: {out_fig}')

# ---------------------------------------------------------------------
# 4. Paired test MSE vs MAE
# ---------------------------------------------------------------------
print('\n4. Paired test: median |WAPE_MAE − WAPE_MSE| per forecaster')
for fc in mat.forecaster.unique():
    sub = mat[mat.forecaster == fc]
    delta = sub.delta_wape.values
    print(f'   {fc}: median(MAE-MSE) = {np.median(delta):+.4f}, '
          f'WAPE typically {"improves" if np.median(delta) < 0 else "worsens"} with MAE')

print('\nDONE')
