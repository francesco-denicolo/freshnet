"""
45_friedman_nemenyi.py — Friedman test + Nemenyi post-hoc + Critical Difference diagram
========================================================================================
Standard benchmarking methodology (Demšar 2006, JMLR).

For each series, rank the 103 cells from 1 (best WAPE_h) to k (worst).
Friedman test on per-series ranks → reject H0 (all cells equivalent)?
Nemenyi post-hoc: two cells are statistically distinguishable iff
    |mean_rank_i - mean_rank_j| > CD,
where  CD = q_α(k,∞)/√2 · √(k(k+1)/(6N)).

Output:
  - friedman_nemenyi_ranks.parquet  full ranking + CD-indistinguishable-from-best flag
  - fig_cd_diagram.png              top-20 cells with CD reference line
  - Cross-check: CD-indistinguishable vs TOST-equivalent set (script 43)
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

NON_HPO_FC = {'chronos_bolt', 'timesfm', 'global_mean', 'dow_mean', 'ma_k56'}

def parse_name(n):
    return n.split('__', 1) if '__' in n else ('no_imp', n)

# ----------------------------------------------------------------------
# 1. Load all cells
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
# 2. Build wide WAPE matrix [series × cells]
# ----------------------------------------------------------------------
print('\n2. Building wide WAPE matrix (series × cells)...')
cell_names = sorted(per_series.keys())
common_idx = None
for name in cell_names:
    idx = per_series[name].set_index(['store_id', 'product_id']).index
    common_idx = idx if common_idx is None else common_idx.intersection(idx)
print(f'   Common series across all cells: {len(common_idx):,}')

W = pd.DataFrame(index=common_idx, columns=cell_names, dtype=float)
for name in cell_names:
    s = per_series[name].set_index(['store_id', 'product_id'])['hourly_wape']
    W[name] = s.loc[common_idx].values
n_before = len(W)
W = W.dropna()
print(f'   Complete-case rows: {len(W):,} (dropped {n_before - len(W):,})')

k = len(cell_names)
N = len(W)

# ----------------------------------------------------------------------
# 3. Friedman test (global H0: all cells equivalent)
# ----------------------------------------------------------------------
print(f'\n3. Friedman test on k={k} cells × N={N:,} series...')
chi2, p_fried = stats.friedmanchisquare(*[W[c].values for c in cell_names])
df_fried = k - 1
print(f'   χ²({df_fried}) = {chi2:.2f}')
print(f'   p = {p_fried:.4e}')
if p_fried < 0.05:
    print(f'   → Reject H0: at least one pair of cells differs in distribution.')
else:
    print(f'   → Fail to reject H0.')

# Kendall's W — overall effect size (coefficient of concordance)
# W = χ² / [N · (k-1)]  range [0,1].  Cohen-style: 0.1=small, 0.3=moderate, 0.5=large
kendall_W = chi2 / (N * (k - 1))
if kendall_W < 0.1:
    W_cat = 'negligible'
elif kendall_W < 0.3:
    W_cat = 'small'
elif kendall_W < 0.5:
    W_cat = 'moderate'
else:
    W_cat = 'large'
print(f"   Kendall's W = {kendall_W:.4f}  ({W_cat} concordance across series)")

# ----------------------------------------------------------------------
# 4. Per-series ranks + mean rank per cell
# ----------------------------------------------------------------------
print('\n4. Computing per-series ranks (1 = best WAPE, k = worst)...')
R_per_series = W.rank(axis=1, method='average')
R_mean = R_per_series.mean(axis=0).sort_values()
print(f'   Mean rank range: [{R_mean.min():.2f}, {R_mean.max():.2f}]')

# ----------------------------------------------------------------------
# 5. Critical Difference (Nemenyi)
# ----------------------------------------------------------------------
print('\n5. Critical Difference (Nemenyi post-hoc)...')
q_alpha = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
print(f'   q_α(k={k}, ∞)/√2 = {q_alpha:.3f}')
print(f'   CD = q_α · √(k(k+1)/(6N)) = {CD:.4f} rank units')
print(f'   → Two cells with |Δ mean_rank| ≤ {CD:.3f} are statistically indistinguishable.')

# ----------------------------------------------------------------------
# 6. Save full ranking + indistinguishable-from-best flag
# ----------------------------------------------------------------------
out = pd.DataFrame({
    'cell': R_mean.index,
    'mean_rank': R_mean.values,
})
out['rank_position'] = np.arange(1, len(out) + 1)
best_rank = out['mean_rank'].iloc[0]
out['cd_indistinguishable'] = (out['mean_rank'] - best_rank) <= CD
out.to_parquet(f'{RESULTS_DIR}/friedman_nemenyi_ranks.parquet', index=False)
print(f'\n6. CD-indistinguishable from best ({out["cell"].iloc[0]}): '
      f'{out.cd_indistinguishable.sum()} cells')
for c in out[out.cd_indistinguishable].cell:
    rk = out[out.cell == c]['mean_rank'].iloc[0]
    print(f'   - {c} (mean_rank={rk:.3f})')

# ----------------------------------------------------------------------
# 7. Cross-check with TOST equivalence set (script 43)
# ----------------------------------------------------------------------
print('\n7. Cross-check CD vs TOST equivalence set...')
tost_path = f'{RESULTS_DIR}/tost_equivalence.parquet'
if os.path.exists(tost_path):
    tost = pd.read_parquet(tost_path)
    tost_global = tost[tost.level == 'global']
    tost_equiv = set(tost_global[tost_global.tost_decision == 'EQUIVALENT']['other_cell'])
    tost_equiv.add(out['cell'].iloc[0])  # add best to TOST set
    cd_equiv = set(out[out.cd_indistinguishable].cell)
    both = tost_equiv & cd_equiv
    only_tost = tost_equiv - cd_equiv
    only_cd = cd_equiv - tost_equiv
    print(f'   TOST equivalent set:    {len(tost_equiv)} cells')
    print(f'   CD indistinguishable:   {len(cd_equiv)} cells')
    print(f'   Both:                   {len(both)}')
    print(f'   Only TOST (not CD):     {len(only_tost)}  {sorted(only_tost) if only_tost else ""}')
    print(f'   Only CD (not TOST):     {len(only_cd)}  {sorted(only_cd) if only_cd else ""}')

# ----------------------------------------------------------------------
# 8. CD diagram (top-N) — Demšar-style
# ----------------------------------------------------------------------
print('\n8. Generating CD diagram (top-20)...')
TOP_N = 20
top = out.head(TOP_N).copy()

fig, ax = plt.subplots(figsize=(14, 8))
y = np.arange(TOP_N)
ranks = top['mean_rank'].values
labels = top['cell'].tolist()
colors = ['#4575b4' if top['cd_indistinguishable'].iloc[i] else '#cccccc'
          for i in range(TOP_N)]
ax.barh(y, ranks, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Mean rank across series (lower = better)', fontsize=13)
ax.set_title(f'Top-{TOP_N} cells by Friedman mean rank   '
             f'(k={k}, N={N:,}, χ²={chi2:.0f}, p<{p_fried:.1e})\n'
             f'Blue = CD-indistinguishable from best (CD={CD:.3f})',
             fontsize=12, pad=10)
ax.axvline(best_rank, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
           label=f'best = {labels[0]} ({best_rank:.2f})')
ax.axvline(best_rank + CD, color='orange', linestyle=':', alpha=0.8, linewidth=1.8,
           label=f'best + CD ({best_rank + CD:.2f})')
ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(left=best_rank - 0.5)
plt.tight_layout()
out_fig = f'{FIG_DIR}/fig_cd_diagram.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'   Saved: {out_fig}')

print('\nDONE')
