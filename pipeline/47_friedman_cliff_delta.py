"""
47_friedman_cliff_delta.py — Pairwise Cliff's δ vs FRIEDMAN best (not median best)
==================================================================================
The Friedman "best" (lowest mean rank) can differ from the median-WAPE best.
This script complements script 43 (which uses median best) by computing:
  For each Friedman best (1 global + 4 per-quartile):
    pairwise Cliff's δ vs every other cell + 95% bootstrap CI + TOST decision.

Output:
  - friedman_cliff_delta.parquet     pairwise table
"""
import os, glob, functools, time
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

MARGIN = 0.147
N_BOOT = 300
SEED = 42
np.random.seed(SEED)

NON_HPO_FC = {'chronos_bolt', 'timesfm', 'global_mean', 'dow_mean', 'ma_k56', 'croston', 'sba', 'tsb'}

def parse_name(n):
    return n.split('__', 1) if '__' in n else ('no_imp', n)

def cliffs_delta(a, b):
    return ((a < b).sum() - (a > b).sum()) / len(a)

def bootstrap_cliffs(a, b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(a); deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[k] = ((a[idx] < b[idx]).sum() - (a[idx] > b[idx]).sum()) / n
    return deltas

def tost_decision(ci_lo, ci_hi, margin=MARGIN):
    if ci_lo > -margin and ci_hi < +margin: return 'EQUIVALENT'
    if ci_hi < -margin or ci_lo > +margin: return 'NOT_EQUIVALENT'
    return 'INCONCLUSIVE'

# ----------------------------------------------------------------------
# Load cells
# ----------------------------------------------------------------------
print('1. Loading cells...')
per_series = {}; seen = set()
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

# ----------------------------------------------------------------------
# Volume quartiles
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
# Read Friedman bests from script 45/46 outputs
# ----------------------------------------------------------------------
print('\n3. Reading Friedman bests (script 45/46 outputs)...')
fr_global = pd.read_parquet(f'{RESULTS_DIR}/friedman_nemenyi_ranks.parquet')
fr_quart = pd.read_parquet(f'{RESULTS_DIR}/friedman_nemenyi_ranks_per_quartile.parquet')

best_global = fr_global.iloc[0]['cell']
best_q = {}
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    sub = fr_quart[fr_quart.quartile == q].sort_values('mean_rank')
    best_q[q] = sub.iloc[0]['cell']
print(f'   Global Friedman best: {best_global}')
for q in ['Q1','Q2','Q3','Q4']:
    print(f'   {q} Friedman best:     {best_q[q]}')

# ----------------------------------------------------------------------
# 4. Pairwise Cliff's δ + bootstrap CI: each Friedman best vs every other cell
# ----------------------------------------------------------------------
def pairwise(best_cell, level, mask_idx=None):
    """Compute Cliff δ + CI95 + TOST decision for best_cell vs all others.

    mask_idx (optional): a pandas Index of (store_id, product_id) to restrict comparison.
    """
    best_imp, best_fc = parse_name(best_cell)
    best_df = per_series[best_cell].set_index(['store_id','product_id'])['hourly_wape']
    rows = []
    t0 = time.time()
    n_total = len(per_series)
    for i, (other_cell, df) in enumerate(per_series.items()):
        if other_cell == best_cell: continue
        other_s = df.set_index(['store_id','product_id'])['hourly_wape']
        common = best_df.index.intersection(other_s.index)
        if mask_idx is not None:
            common = common.intersection(mask_idx)
        a = best_df.loc[common].values; b = other_s.loc[common].values
        valid = ~(np.isnan(a) | np.isnan(b))
        a, b = a[valid], b[valid]
        if len(a) < 100: continue
        d_obs = cliffs_delta(a, b)
        boots = bootstrap_cliffs(a, b, n_boot=N_BOOT, seed=SEED + i)
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        decision = tost_decision(ci_lo, ci_hi)
        rows.append({
            'level': level, 'best_cell': best_cell,
            'other_cell': other_cell, 'n_paired': len(a),
            'cliffs_delta': d_obs, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'tost_decision': decision,
            'threshold_equiv': abs(d_obs) < MARGIN,
        })
        if (i + 1) % 25 == 0:
            print(f'    {i+1}/{n_total} | elapsed={time.time()-t0:.0f}s')
    return pd.DataFrame(rows)

print('\n4. Computing pairwise Cliff δ for Friedman bests...')
all_rows = []

print(f'\n  [global] best={best_global}')
r = pairwise(best_global, 'global')
all_rows.append(r)
print(f'    {len(r)} comparisons | TOST: '
      f'EQ={(r.tost_decision=="EQUIVALENT").sum()} '
      f'INC={(r.tost_decision=="INCONCLUSIVE").sum()} '
      f'NEQ={(r.tost_decision=="NOT_EQUIVALENT").sum()}')

for q in ['Q1','Q2','Q3','Q4']:
    mask_idx = quart_map[quart_map == q].index
    print(f'\n  [{q}] best={best_q[q]} (N_series={len(mask_idx):,})')
    r = pairwise(best_q[q], f'quartile_{q}', mask_idx=mask_idx)
    all_rows.append(r)
    print(f'    {len(r)} comparisons | TOST: '
          f'EQ={(r.tost_decision=="EQUIVALENT").sum()} '
          f'INC={(r.tost_decision=="INCONCLUSIVE").sum()} '
          f'NEQ={(r.tost_decision=="NOT_EQUIVALENT").sum()}')

out = pd.concat(all_rows, ignore_index=True)
out.to_parquet(f'{RESULTS_DIR}/friedman_cliff_delta.parquet', index=False)
print(f'\n5. Saved: friedman_cliff_delta.parquet ({len(out)} rows)')

# ----------------------------------------------------------------------
# 6. Summary: per level, list TOST-EQUIVALENT cells (sorted by |δ|)
# ----------------------------------------------------------------------
print('\n6. TOST-EQUIVALENT to Friedman best (per level):')
for level in out.level.unique():
    sub = out[(out.level == level) & (out.tost_decision == 'EQUIVALENT')]
    sub = sub.assign(abs_d=lambda d: d.cliffs_delta.abs()).sort_values('abs_d')
    print(f'\n  --- {level} (best={sub.best_cell.iloc[0] if len(sub) else "?"}) ---')
    print(f'  {len(sub)} cells EQUIVALENT (Cliff δ near 0)')
    for _, r in sub.head(15).iterrows():
        print(f'    {r.other_cell:<40s}  δ={r.cliffs_delta:+.3f}  CI=[{r.ci_lo:+.3f},{r.ci_hi:+.3f}]')

print('\nDONE')
