"""
46_friedman_nemenyi_per_quartile.py — Friedman + Nemenyi per volume quartile
============================================================================
Same methodology as script 45 (Demšar 2006) but stratified by volume quartile.
For each quartile Q1..Q4:
  1. Filter series belonging to the quartile.
  2. Build wide WAPE_h matrix [series_Q × 103 cells].
  3. Friedman test on per-series ranks.
  4. Nemenyi post-hoc: CD = q_α(k,∞)/√2 · √(k(k+1)/(6·N_Q)).
  5. CD-indistinguishable-from-best set.
  6. Cross-check with TOST per-quartile equivalence set (script 43).

Output:
  - friedman_nemenyi_ranks_per_quartile.parquet  (full ranking per Q + CD flag)
  - fig_cd_diagram_per_quartile.png              (4-panel CD diagram, top-15 per Q)
"""
import os, glob, functools
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import studentized_range
print = functools.partial(print, flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

NON_HPO_FC = {'chronos_bolt', 'timesfm', 'global_mean', 'dow_mean', 'ma_k21'}

def parse_name(n):
    return n.split('__', 1) if '__' in n else ('no_imp', n)

# ----------------------------------------------------------------------
# 1. Load cells
# ----------------------------------------------------------------------
print('1. Loading cells...')
per_series = {}
seen = set()
for f in sorted(glob.glob(f'{RESULTS_DIR}/*_hpo_test_per_series.parquet')):
    name = os.path.basename(f).replace('_hpo_test_per_series.parquet', '')
    per_series[name] = pd.read_parquet(f)
    seen.add(name)
for f in sorted(glob.glob(f'{RESULTS_DIR}/*_test_per_series.parquet')):
    fn = os.path.basename(f)
    if '_hpo_test_per_series' in fn:
        continue
    name = fn.replace('_test_per_series.parquet', '')
    if name in seen:
        continue
    if name.startswith('naive_'):
        base = name.replace('naive_', '', 1)
        if base in NON_HPO_FC:
            new_name = f'no_imp__{base}'
            if new_name not in seen:
                per_series[new_name] = pd.read_parquet(f)
                seen.add(new_name)
        continue
    imp, fc = parse_name(name)
    if fc not in NON_HPO_FC:
        continue
    per_series[name] = pd.read_parquet(f)
    seen.add(name)
print(f'   Loaded {len(per_series)} cells')

# ----------------------------------------------------------------------
# 2. Volume quartiles
# ----------------------------------------------------------------------
print('\n2. Computing volume quartiles from train days 1-83...')
df_tr = pd.read_parquet('/Users/utente/Desktop/FreshNetRetail/data/frn50k_train.parquet')
df_tr['dt_parsed'] = pd.to_datetime(df_tr['dt'])
df_tr = df_tr.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)
df_tr['day_num'] = df_tr.groupby(['store_id', 'product_id']).cumcount() + 1
vol = (df_tr[df_tr.day_num <= 83]
       .groupby(['store_id', 'product_id'])['sale_amount'].sum().reset_index())
vol['quartile'] = pd.qcut(vol['sale_amount'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']).astype(str)
quart_map = vol.set_index(['store_id', 'product_id'])['quartile']
del df_tr
print(f'   Quartile counts: {quart_map.value_counts().to_dict()}')

# ----------------------------------------------------------------------
# 3. Build wide WAPE matrix (global) and intersect with quartile map
# ----------------------------------------------------------------------
print('\n3. Building global wide WAPE matrix...')
cell_names = sorted(per_series.keys())
common_idx = None
for name in cell_names:
    idx = per_series[name].set_index(['store_id', 'product_id']).index
    common_idx = idx if common_idx is None else common_idx.intersection(idx)
print(f'   Common series: {len(common_idx):,}')

W = pd.DataFrame(index=common_idx, columns=cell_names, dtype=float)
for name in cell_names:
    s = per_series[name].set_index(['store_id', 'product_id'])['hourly_wape']
    W[name] = s.loc[common_idx].values
W = W.dropna()
print(f'   Complete-case rows (global): {len(W):,}')

# Attach quartile to each series (intersect index with quart_map)
W_quart = quart_map.reindex(W.index)
print(f'   Series with quartile label: {W_quart.notna().sum():,}')

# ----------------------------------------------------------------------
# 4. Friedman + Nemenyi per quartile
# ----------------------------------------------------------------------
all_rows = []
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
tost_path = f'{RESULTS_DIR}/tost_equivalence.parquet'
tost = pd.read_parquet(tost_path) if os.path.exists(tost_path) else None
k = len(cell_names)

for ax, q in zip(axes.flat, ['Q1', 'Q2', 'Q3', 'Q4']):
    print(f'\n=== Quartile {q} ===')
    mask_q = (W_quart == q).values
    W_q = W[mask_q]
    N_q = len(W_q)
    print(f'  Series in {q}: {N_q:,}')

    # Friedman
    chi2, p_fried = stats.friedmanchisquare(*[W_q[c].values for c in cell_names])
    df_fried = k - 1
    print(f'  Friedman χ²({df_fried}) = {chi2:.2f}, p = {p_fried:.2e}')

    # Kendall's W — effect size for Friedman (concordance across series in this Q)
    kendall_W = chi2 / (N_q * (k - 1))
    if kendall_W < 0.1: W_cat = 'negligible'
    elif kendall_W < 0.3: W_cat = 'small'
    elif kendall_W < 0.5: W_cat = 'moderate'
    else: W_cat = 'large'
    print(f"  Kendall's W = {kendall_W:.4f}  ({W_cat})")

    # Per-series ranks → mean rank per cell
    R_per_series = W_q.rank(axis=1, method='average')
    R_mean = R_per_series.mean(axis=0).sort_values()
    best_cell = R_mean.index[0]
    best_rank = R_mean.iloc[0]
    print(f'  Best by mean rank: {best_cell} ({best_rank:.3f})')
    print(f'  Mean rank range: [{R_mean.min():.2f}, {R_mean.max():.2f}]')

    # Critical Difference
    q_alpha = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N_q))
    print(f'  CD (α=0.05) = {CD:.4f} rank units')

    # CD-indistinguishable set
    cd_indist = R_mean[(R_mean - best_rank) <= CD]
    print(f'  CD-indistinguishable from best: {len(cd_indist)} cells')
    for c in cd_indist.index:
        rk = R_mean.loc[c]
        print(f'    - {c} (mean_rank={rk:.3f})')

    # Cross-check with TOST per-quartile equivalence
    cd_set = set(cd_indist.index)
    if tost is not None:
        tost_q = tost[tost.level == f'quartile_{q}']
        # tost_q best_cell is the per-quartile best by median WAPE; equivalent set is the EQUIVALENT decisions.
        tost_equiv = set(tost_q[tost_q.tost_decision == 'EQUIVALENT']['other_cell'])
        # Include each level's "best_cell" (a single cell per quartile is the TOST reference)
        tost_equiv.update(tost_q.best_cell.unique())
        both = cd_set & tost_equiv
        only_cd = cd_set - tost_equiv
        only_tost = tost_equiv - cd_set
        print(f'  CD ∩ TOST: {len(both)}   only CD: {len(only_cd)}   only TOST: {len(only_tost)}')

    # Save rows for parquet
    for pos, c in enumerate(R_mean.index, start=1):
        all_rows.append({
            'quartile': q, 'cell': c, 'mean_rank': float(R_mean.loc[c]),
            'rank_position': pos,
            'cd_indistinguishable': c in cd_set,
            'best_cell': best_cell, 'CD': CD,
            'N_series': N_q,
            'kendall_W': kendall_W,
        })

    # CD diagram top-15 per quartile
    TOP_N = 15
    top = R_mean.head(TOP_N)
    y = np.arange(TOP_N)
    colors = ['#4575b4' if (top.iloc[i] - best_rank) <= CD else '#cccccc'
              for i in range(TOP_N)]
    ax.barh(y, top.values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(top.index.tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean rank (lower = better)', fontsize=11)
    ax.set_title(f'{q}   N={N_q:,}   k={k}   χ²={chi2:.0f} (p<{p_fried:.0e})\n'
                 f'Best: {best_cell}   CD={CD:.3f}   '
                 f'{len(cd_indist)} cells indistinguishable',
                 fontsize=11, pad=8)
    ax.axvline(best_rank, color='red', linestyle='--', alpha=0.7, linewidth=1.3)
    ax.axvline(best_rank + CD, color='orange', linestyle=':', alpha=0.8, linewidth=1.5)
    ax.set_xlim(left=best_rank - 0.5)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

fig.suptitle('Friedman + Nemenyi per volume quartile — top-15 cells\n'
             'Blue = CD-indistinguishable from best (Nemenyi α=0.05)',
             fontsize=13, y=1.005)
plt.tight_layout()
out_fig = f'{FIG_DIR}/fig_cd_diagram_per_quartile.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out_fig}')

# Save full ranking
out = pd.DataFrame(all_rows)
out.to_parquet(f'{RESULTS_DIR}/friedman_nemenyi_ranks_per_quartile.parquet', index=False)
print(f'Saved: friedman_nemenyi_ranks_per_quartile.parquet ({len(out)} rows)')

# Cross-quartile best summary
print('\n5. Best cell by Friedman mean rank per quartile:')
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    sub = out[out.quartile == q].sort_values('mean_rank')
    n_cd = sub.cd_indistinguishable.sum()
    print(f'  {q}: {sub.iloc[0].cell} (mean_rank={sub.iloc[0].mean_rank:.3f}, CD={sub.iloc[0].CD:.3f}, '
          f'{n_cd} CD-equiv)')

print('\nDONE')
