"""
36_stratified_quartile_analysis.py — Stratified analysis by series volume quartile
====================================================================================
Per ogni cella (imputer, forecaster):
  1. Calcola WAPE_h_med stratificato per quartile di volume (Q1-Q4 su sales training).
  2. Identifica best per quartile + equivalence set (Cliff δ < 0.147 vs best Q).
  3. Heatmap 4 sub-figure (Q1-Q4) imputer × forecaster.
Output: pipeline/results/hpo_stratified_quartile.parquet, fig_stratified_4grid.png
"""
import os, glob, functools, numpy as np, pandas as pd
from scipy import stats
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EQUIV_THRESHOLD = 0.147  # |Cliff δ| < 0.147 = negligible effect

# ============================================================================
# 1. Compute series volume on training (days 1-83)
# ============================================================================
print('1. Computing series volume on training (days 1-83)...')
df = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df['dt_parsed'] = pd.to_datetime(df['dt'])
df = df.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
df['day_num'] = df.groupby(['store_id','product_id']).cumcount() + 1

# Training period: days 1-83
df_train = df[df.day_num <= 83]
vol = df_train.groupby(['store_id','product_id'])['sale_amount'].sum().reset_index()
vol.columns = ['store_id','product_id','volume']

# Assign quartiles
vol['quartile'] = pd.qcut(vol['volume'], q=4, labels=['Q1','Q2','Q3','Q4']).astype(str)
print(f'   {len(vol):,} series, quartile sizes: '
      f'{vol.quartile.value_counts().sort_index().to_dict()}')
print(f'   Volume thresholds (Q1/Q2/Q3 medians): '
      f'{vol[vol.quartile=="Q1"].volume.max():.1f} | '
      f'{vol[vol.quartile=="Q2"].volume.max():.1f} | '
      f'{vol[vol.quartile=="Q3"].volume.max():.1f}')

# ============================================================================
# 2. Load all 95 cells and join with quartile
# ============================================================================
print('\n2. Loading 95 cells and joining quartile...')

NON_HPO_FC = {'chronos_bolt','global_mean','dow_mean','ma_k21'}

def parse_name(name):
    if '__' in name:
        return name.split('__', 1)
    return 'no_imp', name

per_series = {}
for f in sorted(glob.glob(f'{RESULTS_DIR}/*_hpo_test_per_series.parquet')):
    name = os.path.basename(f).replace('_hpo_test_per_series.parquet','')
    imp, fc = parse_name(name)
    per_series[(imp, fc)] = pd.read_parquet(f).merge(vol[['store_id','product_id','quartile']],
                                                     on=['store_id','product_id'], how='inner')

for f in sorted(glob.glob(f'{RESULTS_DIR}/*_test_per_series.parquet')):
    fn = os.path.basename(f)
    if '_hpo_test_per_series' in fn: continue
    name = fn.replace('_test_per_series.parquet','')
    # Map naive_<fc> -> no_imp__<fc>
    if name.startswith('naive_'):
        base = name.replace('naive_','',1)
        if base in NON_HPO_FC:
            imp, fc = 'no_imp', base
        else:
            continue
    else:
        imp, fc = parse_name(name)
    if fc not in NON_HPO_FC: continue
    if (imp, fc) in per_series: continue
    per_series[(imp, fc)] = pd.read_parquet(f).merge(vol[['store_id','product_id','quartile']],
                                                     on=['store_id','product_id'], how='inner')

print(f'   Loaded {len(per_series)} cells')

# ============================================================================
# 3. Compute WAPE_h_med per (cell, quartile)
# ============================================================================
print('\n3. Computing stratified WAPE_h_med per (cell, quartile)...')
strat_rows = []
for (imp, fc), df_cell in per_series.items():
    for q in ['Q1','Q2','Q3','Q4']:
        sub = df_cell[df_cell.quartile == q]
        if len(sub) == 0: continue
        strat_rows.append({
            'imputer': imp, 'forecaster': fc, 'cell': f'{imp}__{fc}',
            'quartile': q, 'n_series': len(sub),
            'wape_h_med': sub.hourly_wape.median(),
            'abs_wpe_med': abs(sub.hourly_wpe.median()),
        })
strat = pd.DataFrame(strat_rows)
print(f'   {len(strat)} stratified observations')

# ============================================================================
# 4. Best per quartile + equivalence set
# ============================================================================
print('\n4. Best + equivalence set per quartile...')

def cliff_delta(a, b):
    """Compute Cliff's delta on paired series. Drop NaN pairs."""
    a, b = np.asarray(a), np.asarray(b)
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) == 0: return np.nan
    gt = (a < b).sum(); lt = (a > b).sum()
    return (gt - lt) / len(a)

best_per_q = {}
equiv_per_q = {}
for q in ['Q1','Q2','Q3','Q4']:
    sub = strat[strat.quartile == q].sort_values('wape_h_med')
    best_cell = sub.iloc[0]['cell']
    best_imp, best_fc = parse_name(best_cell)
    best_per_q[q] = sub.iloc[0]
    # Equivalence: Cliff δ best vs other cells on series in this quartile
    best_df = per_series[(best_imp, best_fc)]
    best_q = best_df[best_df.quartile == q].set_index(['store_id','product_id'])['hourly_wape']
    equiv = []
    for _, r in sub.iterrows():
        if r.cell == best_cell:
            equiv.append((r.cell, 0.0))
            continue
        other_imp, other_fc = parse_name(r.cell)
        other_df = per_series[(other_imp, other_fc)]
        other_q = other_df[other_df.quartile == q].set_index(['store_id','product_id'])['hourly_wape']
        common = best_q.index.intersection(other_q.index)
        d = cliff_delta(best_q.loc[common].values, other_q.loc[common].values)
        equiv.append((r.cell, d))
    eq_df = pd.DataFrame(equiv, columns=['cell','cliffs_delta'])
    eq_df['equiv'] = eq_df.cliffs_delta.abs() < EQUIV_THRESHOLD
    equiv_per_q[q] = eq_df
    n_eq = eq_df.equiv.sum()
    print(f'   {q}: best={best_cell} WAPE={sub.iloc[0].wape_h_med:.4f}, equivalence set: {n_eq} cells')

# Mark equivalence in stratified df
strat['equiv'] = False
for q, eq_df in equiv_per_q.items():
    eq_cells = set(eq_df[eq_df.equiv].cell)
    strat.loc[(strat.quartile==q) & (strat.cell.isin(eq_cells)), 'equiv'] = True

# Save
strat.to_parquet(f'{RESULTS_DIR}/hpo_stratified_quartile.parquet', index=False)
pd.concat([eq_df.assign(quartile=q) for q, eq_df in equiv_per_q.items()]).to_parquet(
    f'{RESULTS_DIR}/hpo_equivalence_per_quartile.parquet', index=False)

# ============================================================================
# 5. 4-grid heatmap (Q1-Q4)
# ============================================================================
print('\n5. Generating 4-grid heatmap...')

IMP_ORDER = ['no_imp','media_glob','media_cond','mediana_glob','mediana_cond',
             'forward_fill','seasonal_naive','linear_interp','lgb',
             'dlinear','saits','itransformer','timesnet','imputeformer']
FC_ORDER = ['lgb_nolags','lgb_m5lags','mlp_nolags','mlp_m5lags','tft',
            'chronos_bolt','global_mean','dow_mean','ma_k21']
FC_LABEL = {'lgb_nolags':'LGB_nl','lgb_m5lags':'LGB_M5','mlp_nolags':'MLP_nl','mlp_m5lags':'MLP_M5',
            'tft':'TFT','chronos_bolt':'Chron','global_mean':'GM','dow_mean':'DoW','ma_k21':'MA21'}

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
for ax, q in zip(axes.flat, ['Q1','Q2','Q3','Q4']):
    sub = strat[strat.quartile == q]
    pivot = sub.pivot(index='imputer', columns='forecaster', values='wape_h_med')
    pivot = pivot.reindex(index=IMP_ORDER, columns=FC_ORDER)
    eq_mask = sub.pivot(index='imputer', columns='forecaster', values='equiv').reindex(
              index=IMP_ORDER, columns=FC_ORDER).fillna(False)
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto',
                   vmin=0.95, vmax=1.20)
    # Annotate values + mark equivalence/best with edge
    best_cell = best_per_q[q].cell
    for i, imp in enumerate(IMP_ORDER):
        for j, fc in enumerate(FC_ORDER):
            v = pivot.iloc[i, j]
            if np.isnan(v): continue
            cell = f'{imp}__{fc}'
            is_best = (cell == best_cell)
            is_equiv = eq_mask.iloc[i, j]
            text = f'{v:.3f}'
            color = 'white' if v > 1.10 or v < 0.98 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=8.5,
                    color=color, fontweight='bold' if is_best else 'normal')
            if is_best:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                            edgecolor='blue', lw=3, zorder=3))
            elif is_equiv:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                            edgecolor='orange', lw=1.8, linestyle='--', zorder=2))
    ax.set_xticks(range(len(FC_ORDER)))
    ax.set_xticklabels([FC_LABEL[c] for c in FC_ORDER], rotation=30, fontsize=9)
    ax.set_yticks(range(len(IMP_ORDER)))
    ax.set_yticklabels(IMP_ORDER, fontsize=9)
    bp = best_per_q[q]
    n_eq = equiv_per_q[q].equiv.sum()
    vol_min = vol[vol.quartile==q].volume.min()
    vol_max = vol[vol.quartile==q].volume.max()
    ax.set_title(f'{q}: vol ∈ [{vol_min:.0f}, {vol_max:.0f}]\n'
                 f'best={bp.cell} (WAPE={bp.wape_h_med:.3f}), equiv n={n_eq}',
                 fontsize=11)
    plt.colorbar(im, ax=ax, label='WAPE_h_med')

fig.suptitle('Stratified WAPE_h_med per volume quartile — blue=best, orange dashed=equivalence set',
             fontsize=14, y=1.005)
plt.tight_layout()
out_fig = f'{FIG_DIR}/fig_stratified_4grid.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'   Saved: {out_fig}')

# ============================================================================
# 6. Crossover summary
# ============================================================================
print('\n6. Crossover: best changes between quartiles?')
for q in ['Q1','Q2','Q3','Q4']:
    bp = best_per_q[q]
    print(f'   {q}: {bp.cell} (WAPE={bp.wape_h_med:.4f})')

print('\nDONE')
