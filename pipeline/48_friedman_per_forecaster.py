"""
48_friedman_per_forecaster.py — Friedman + Kendall W + Nemenyi CD per ogni forecaster
======================================================================================
Per ciascun forecaster fc, restringe la matrice ai k imputer di fc e applica
Friedman + Kendall W + Nemenyi CD per identificare:
  - best imputer (mean rank)
  - equivalence set (CD-indistinguishable)

Versioni:
  (a) GLOBAL: tutte le serie (~49.9K)
  (b) PER QUARTILE: Q1..Q4 stratificato

Output:
  - friedman_per_forecaster.parquet   ranking per (level, fc, imputer)
  - fig_cd_per_forecaster.png         mini CD diagram per ogni fc (global)
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

def kendall_category(W):
    if W < 0.1: return 'negligible'
    if W < 0.3: return 'small'
    if W < 0.5: return 'moderate'
    return 'large'

# ----------------------------------------------------------------------
# 1. Load cells (same pattern as 45)
# ----------------------------------------------------------------------
print('1. Loading cells...')
per_series = {}
seen = set()
for f in sorted(glob.glob(f'{RESULTS_DIR}/*_hpo_test_per_series.parquet')):
    name = os.path.basename(f).replace('_hpo_test_per_series.parquet', '')
    per_series[name] = pd.read_parquet(f); seen.add(name)
for f in sorted(glob.glob(f'{RESULTS_DIR}/*_test_per_series.parquet')):
    fn = os.path.basename(f)
    if '_hpo_test_per_series' in fn: continue
    name = fn.replace('_test_per_series.parquet', '')
    if name in seen: continue
    if name.startswith('naive_'):
        base = name.replace('naive_', '', 1)
        if base in NON_HPO_FC:
            new_name = f'no_imp__{base}'
            if new_name not in seen:
                per_series[new_name] = pd.read_parquet(f); seen.add(new_name)
        continue
    imp, fc = parse_name(name)
    if fc not in NON_HPO_FC: continue
    per_series[name] = pd.read_parquet(f); seen.add(name)
print(f'   Loaded {len(per_series)} cells')

# Build cell→(imp,fc) mapping
cells_by_fc = {}
for name in per_series:
    imp, fc = parse_name(name)
    cells_by_fc.setdefault(fc, []).append((imp, name))
fc_list = sorted(cells_by_fc.keys())
print(f'   Forecasters: {fc_list}')
for fc in fc_list:
    print(f'     {fc}: {len(cells_by_fc[fc])} imputers')

# ----------------------------------------------------------------------
# 2. Volume quartiles
# ----------------------------------------------------------------------
print('\n2. Computing volume quartiles...')
df_tr = pd.read_parquet('/Users/utente/Desktop/FreshNetRetail/data/frn50k_train.parquet')
df_tr['dt_parsed'] = pd.to_datetime(df_tr['dt'])
df_tr = df_tr.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
df_tr['day_num'] = df_tr.groupby(['store_id','product_id']).cumcount() + 1
vol = (df_tr[df_tr.day_num <= 83]
       .groupby(['store_id','product_id'])['sale_amount'].sum().reset_index())
vol['quartile'] = pd.qcut(vol['sale_amount'], q=4, labels=['Q1','Q2','Q3','Q4']).astype(str)
quart_map = vol.set_index(['store_id','product_id'])['quartile']
del df_tr

# ----------------------------------------------------------------------
# 3. For each forecaster, build wide matrix [series × imputers]
# ----------------------------------------------------------------------
def friedman_nemenyi(W_sub, level_label):
    """Run Friedman + Kendall W + Nemenyi CD on a wide matrix.
    Returns list of dicts (one per imputer)."""
    cell_cols = list(W_sub.columns)
    k = len(cell_cols)
    N = len(W_sub)
    if k < 3 or N < 50:
        return None
    chi2, p_fried = stats.friedmanchisquare(*[W_sub[c].values for c in cell_cols])
    kendall_W = chi2 / (N * (k - 1))
    q_alpha = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
    R_mean = W_sub.rank(axis=1, method='average').mean(axis=0).sort_values()
    best_cell = R_mean.index[0]
    best_rank = R_mean.iloc[0]
    rows = []
    for pos, c in enumerate(R_mean.index, start=1):
        imp = parse_name(c)[0]
        rows.append({
            'level': level_label,
            'forecaster': parse_name(c)[1],
            'imputer': imp,
            'cell': c,
            'mean_rank': float(R_mean.loc[c]),
            'rank_position': pos,
            'cd_indistinguishable': bool((R_mean.loc[c] - best_rank) <= CD),
            'best_cell': best_cell,
            'k': k, 'N': N, 'CD': CD,
            'chi2': chi2, 'p_friedman': p_fried,
            'kendall_W': kendall_W,
            'kendall_cat': kendall_category(kendall_W),
        })
    return rows

print('\n3. Running Friedman+Nemenyi per forecaster (global + per quartile)...')

all_rows = []
summary = []

# For each forecaster, do (global + Q1 + Q2 + Q3 + Q4)
for fc in fc_list:
    imp_cells = cells_by_fc[fc]
    cell_cols = [c for _, c in imp_cells]
    print(f'\n=== {fc} (k={len(cell_cols)} imputers) ===')

    # Build wide matrix W [series × imputers] for this forecaster
    common_idx = None
    for c in cell_cols:
        idx = per_series[c].set_index(['store_id','product_id']).index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    W = pd.DataFrame(index=common_idx, columns=cell_cols, dtype=float)
    for c in cell_cols:
        s = per_series[c].set_index(['store_id','product_id'])['hourly_wape']
        W[c] = s.loc[common_idx].values
    W = W.dropna()
    print(f'  Complete-case rows: {len(W):,}')

    # (a) GLOBAL
    rs = friedman_nemenyi(W, 'global')
    if rs is None:
        print('  SKIP global (k or N too small)')
    else:
        all_rows.extend(rs)
        sub = pd.DataFrame(rs).sort_values('mean_rank')
        best = sub.iloc[0]
        n_eq = sub.cd_indistinguishable.sum()
        print(f'  global   best={best.cell:<40s}  W={best.kendall_W:.3f} ({best.kendall_cat})  CD={best.CD:.3f}  equiv={n_eq}/{len(sub)}')
        summary.append({'level':'global','forecaster':fc,'best_imputer':best.imputer,
                        'k':best.k,'N':best.N,'kendall_W':best.kendall_W,
                        'CD':best.CD,'n_equiv':int(n_eq)})

    # (b) PER QUARTILE
    quart_series = quart_map.reindex(W.index)
    for q in ['Q1','Q2','Q3','Q4']:
        mask = (quart_series == q).values
        W_q = W[mask]
        rs = friedman_nemenyi(W_q, f'quartile_{q}')
        if rs is None:
            continue
        all_rows.extend(rs)
        sub = pd.DataFrame(rs).sort_values('mean_rank')
        best = sub.iloc[0]
        n_eq = sub.cd_indistinguishable.sum()
        print(f'  {q}       best={best.cell:<40s}  W={best.kendall_W:.3f} ({best.kendall_cat})  CD={best.CD:.3f}  equiv={n_eq}/{len(sub)}')
        summary.append({'level':q,'forecaster':fc,'best_imputer':best.imputer,
                        'k':best.k,'N':best.N,'kendall_W':best.kendall_W,
                        'CD':best.CD,'n_equiv':int(n_eq)})

# Save full ranking
out = pd.DataFrame(all_rows)
out.to_parquet(f'{RESULTS_DIR}/friedman_per_forecaster.parquet', index=False)
print(f'\n4. Saved: friedman_per_forecaster.parquet ({len(out)} rows)')

# Save summary
summary_df = pd.DataFrame(summary)
summary_df.to_parquet(f'{RESULTS_DIR}/friedman_per_forecaster_summary.parquet', index=False)

# ----------------------------------------------------------------------
# 4. Pretty summary tables
# ----------------------------------------------------------------------
print('\n5. Summary tables')
for level in ['global','Q1','Q2','Q3','Q4']:
    print(f'\n  --- {level} ---')
    s = summary_df[summary_df.level == level].sort_values('forecaster')
    if len(s) == 0:
        continue
    print(f'  {"forecaster":<14}  {"best_imputer":<16}  {"k":>3}  {"N":>7}  {"W":>5}  {"cat":>9}  {"CD":>5}  {"#eq":>4}')
    for _, r in s.iterrows():
        cat = kendall_category(r.kendall_W)
        print(f'  {r.forecaster:<14}  {r.best_imputer:<16}  {r.k:>3}  {r.N:>7,}  '
              f'{r.kendall_W:>5.3f}  {cat:>9}  {r.CD:>5.3f}  {r.n_equiv:>4}')

# ----------------------------------------------------------------------
# 5. Figure: CD diagram per forecaster (global level)
# ----------------------------------------------------------------------
print('\n6. Generating CD diagram per forecaster (global)...')
fc_to_plot = [fc for fc in fc_list if any(r['level']=='global' and r['forecaster']==fc for r in all_rows)]
n_fc = len(fc_to_plot)
ncols = 3
nrows = (n_fc + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4*nrows))
axes_flat = axes.flat if n_fc > 1 else [axes]
for ax, fc in zip(axes_flat, fc_to_plot):
    sub = pd.DataFrame([r for r in all_rows if r['level']=='global' and r['forecaster']==fc])
    sub = sub.sort_values('mean_rank').reset_index(drop=True)
    y = np.arange(len(sub))
    colors = ['#4575b4' if v else '#cccccc' for v in sub.cd_indistinguishable]
    ax.barh(y, sub.mean_rank.values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(sub.imputer.tolist(), fontsize=9)
    ax.invert_yaxis()
    best_rank = sub.mean_rank.iloc[0]
    CD = sub.CD.iloc[0]
    W = sub.kendall_W.iloc[0]
    ax.axvline(best_rank, color='red', linestyle='--', alpha=0.7, linewidth=1.2)
    ax.axvline(best_rank + CD, color='orange', linestyle=':', alpha=0.7, linewidth=1.4)
    ax.set_title(f'{fc}   k={sub.k.iloc[0]}  W={W:.3f}  CD={CD:.3f}', fontsize=10, pad=6)
    ax.set_xlabel('Mean rank (lower=better)', fontsize=9)
    ax.set_xlim(left=best_rank - 0.5)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
# Hide unused axes
for ax in list(axes_flat)[n_fc:]:
    ax.axis('off')
fig.suptitle('Friedman per forecaster (GLOBAL) — blue = CD-indistinguishable from best (Nemenyi α=0.05)',
             fontsize=13, y=1.005)
plt.tight_layout()
out_fig = f'{FIG_DIR}/fig_cd_per_forecaster.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'   Saved: {out_fig}')

print('\nDONE')
