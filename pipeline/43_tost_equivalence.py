"""
43_tost_equivalence.py — TOST equivalence testing on key comparisons
====================================================================
For each "best vs other" comparison, runs bootstrap-based TOST equivalence test:
  H0: |Cliff δ| ≥ 0.147 (the margin)
  H1: |Cliff δ| < 0.147

Methodology:
  - 1000 paired-bootstrap iterations
  - Compute Cliff δ on each bootstrap sample
  - 95% bootstrap CI [ci_lo, ci_hi]
  - Decision:
      EQUIVALENT      if ci_lo > -0.147 AND ci_hi < +0.147
      NOT EQUIVALENT  if ci_hi < -0.147 OR ci_lo > +0.147
      INCONCLUSIVE    otherwise (CI crosses margin)

Compared to threshold-only equivalence (|δ̂|<0.147):
  - threshold uses POINT estimate → ignores uncertainty
  - TOST uses INTERVAL → formal Type-I error control
  - Cells classified "EQUIVALENT" by both → strongest evidence
  - Cells classified "EQUIVALENT" by threshold but "INCONCLUSIVE" by TOST →
    weaker claim (small N could change classification)
"""
import os, glob, functools, time, numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

MARGIN = 0.147     # Romano/Cliff "negligible" cutoff
N_BOOT = 1000      # bootstrap iterations
SEED = 42
np.random.seed(SEED)

NON_HPO_FC = {'chronos_bolt','timesfm','global_mean','dow_mean','ma_k56'}

def parse_name(n):
    return n.split('__', 1) if '__' in n else ('no_imp', n)

# ---------------------------------------------------------------------
# Load all cells
# ---------------------------------------------------------------------
print('1. Loading cells...')
per_series = {}; seen = set()
for f in sorted(glob.glob(f'{RESULTS_DIR}/*_hpo_test_per_series.parquet')):
    name = os.path.basename(f).replace('_hpo_test_per_series.parquet','')
    imp, fc = parse_name(name); per_series[(imp,fc)] = pd.read_parquet(f); seen.add(name)
for f in sorted(glob.glob(f'{RESULTS_DIR}/*_test_per_series.parquet')):
    fn = os.path.basename(f)
    if '_hpo_test_per_series' in fn: continue
    name = fn.replace('_test_per_series.parquet','')
    if name in seen: continue
    if name.startswith('naive_'):
        base = name.replace('naive_','',1)
        if base in NON_HPO_FC:
            per_series[('no_imp', base)] = pd.read_parquet(f); seen.add(f'no_imp__{base}')
            continue
    imp, fc = parse_name(name)
    if fc not in NON_HPO_FC: continue
    per_series[(imp, fc)] = pd.read_parquet(f); seen.add(name)
print(f'   Loaded {len(per_series)} cells')

# Quartile assignment
print('2. Computing volume quartiles...')
df_tr = pd.read_parquet('/Users/utente/Desktop/FreshNetRetail/data/frn50k_train.parquet')
df_tr['dt_parsed'] = pd.to_datetime(df_tr['dt'])
df_tr = df_tr.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
df_tr['day_num'] = df_tr.groupby(['store_id','product_id']).cumcount() + 1
vol = df_tr[df_tr.day_num <= 83].groupby(['store_id','product_id'])['sale_amount'].sum().reset_index()
vol.columns = ['store_id','product_id','volume']
vol['quartile'] = pd.qcut(vol['volume'], q=4, labels=['Q1','Q2','Q3','Q4']).astype(str)
quart_map = vol.set_index(['store_id','product_id'])['quartile']
del df_tr

def cliffs_delta(a, b):
    """Paired Cliff's delta (best better when δ > 0)."""
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) == 0: return np.nan
    return ((a < b).sum() - (a > b).sum()) / len(a)

def bootstrap_cliffs(a, b, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap of Cliff's δ; returns array of n_boot deltas."""
    rng = np.random.default_rng(seed)
    n = len(a)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        a_b, b_b = a[idx], b[idx]
        deltas[k] = ((a_b < b_b).sum() - (a_b > b_b).sum()) / n
    return deltas

def tost_decision(ci_lo, ci_hi, margin=MARGIN):
    if ci_lo > -margin and ci_hi < +margin:
        return 'EQUIVALENT'
    if ci_hi < -margin or ci_lo > +margin:
        return 'NOT_EQUIVALENT'
    return 'INCONCLUSIVE'

# ---------------------------------------------------------------------
# 3. Global best vs all other cells (filter: equivalent by point estimate)
# ---------------------------------------------------------------------
print('\n3. TOST: GLOBAL best vs all other cells...')
matgen = pd.read_parquet(f'{RESULTS_DIR}/hpo_matrix_pareto.parquet')
best_global = matgen.sort_values('wape_h_med').iloc[0]
best_imp, best_fc = best_global.imputer, best_global.forecaster
print(f'   Best global: {best_global.cell} (WAPE={best_global.wape_h_med:.4f})')

# Reduce N_BOOT for global comparison to keep runtime tractable
n_boot_global = 300
print(f'   Bootstrap iterations: {n_boot_global}')

best_df = per_series[(best_imp, best_fc)]
best_s = best_df.set_index(['store_id','product_id'])['hourly_wape']

rows_global = []
t0 = time.time()
for i, ((imp, fc), df) in enumerate(per_series.items()):
    if (imp, fc) == (best_imp, best_fc): continue
    other_s = df.set_index(['store_id','product_id'])['hourly_wape']
    common = best_s.index.intersection(other_s.index)
    a = best_s.loc[common].values; b = other_s.loc[common].values
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) < 100: continue
    d_obs = cliffs_delta(a, b)
    boots = bootstrap_cliffs(a, b, n_boot=n_boot_global, seed=SEED+i)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    decision = tost_decision(ci_lo, ci_hi)
    rows_global.append({
        'level': 'global', 'best_cell': best_global.cell,
        'other_cell': f'{imp}__{fc}',
        'cliffs_delta_obs': d_obs, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'tost_decision': decision,
        'threshold_equiv': abs(d_obs) < MARGIN,
    })
    if (i+1) % 20 == 0:
        print(f'     {i+1}/{len(per_series)} cells | elapsed={time.time()-t0:.0f}s')

rg = pd.DataFrame(rows_global)
print(f'   GLOBAL TOST summary:')
print(f'     EQUIVALENT (TOST):     {(rg.tost_decision=="EQUIVALENT").sum()}')
print(f'     INCONCLUSIVE:          {(rg.tost_decision=="INCONCLUSIVE").sum()}')
print(f'     NOT_EQUIVALENT:        {(rg.tost_decision=="NOT_EQUIVALENT").sum()}')
print(f'     threshold-equiv (|δ|<{MARGIN}):  {rg.threshold_equiv.sum()}')

# Show details for equivalent cells
eq_cells = rg[rg.tost_decision=='EQUIVALENT'].sort_values('cliffs_delta_obs', key=abs)
print(f'\n   TOST-equivalent to best global ({len(eq_cells)} cells):')
print(eq_cells[['other_cell','cliffs_delta_obs','ci_lo','ci_hi']].to_string(index=False))

# ---------------------------------------------------------------------
# 4. Per-quartile best vs all other cells in same quartile
# ---------------------------------------------------------------------
print('\n4. TOST: PER-QUARTILE best vs other cells...')
strat = pd.read_parquet(f'{RESULTS_DIR}/hpo_stratified_quartile.parquet')
rows_pq = []
n_boot_q = 300
for q in ['Q1','Q2','Q3','Q4']:
    print(f'\n   Quartile {q}:')
    sub = strat[strat.quartile==q].sort_values('wape_h_med')
    if len(sub) == 0: continue
    best_q_cell = sub.iloc[0]['cell']
    best_q_imp, best_q_fc = parse_name(best_q_cell)
    print(f'     best={best_q_cell} (WAPE={sub.iloc[0].wape_h_med:.4f})')
    if (best_q_imp, best_q_fc) not in per_series:
        print(f'     SKIP: cell data not loaded')
        continue
    best_df_q = per_series[(best_q_imp, best_q_fc)]
    best_df_q = best_df_q.merge(quart_map.reset_index(), on=['store_id','product_id'])
    best_q_sub = best_df_q[best_df_q.quartile == q]
    best_s_q = best_q_sub.set_index(['store_id','product_id'])['hourly_wape']

    # Filter candidates: only point-estimate equivalents (for speed)
    cands = sub[sub.cell.apply(lambda c: c != best_q_cell)]
    n_eq, n_ne, n_in = 0, 0, 0
    for _, r in cands.iterrows():
        imp, fc = parse_name(r.cell)
        if (imp, fc) not in per_series: continue
        df_other = per_series[(imp, fc)]
        df_other = df_other.merge(quart_map.reset_index(), on=['store_id','product_id'])
        sub_other = df_other[df_other.quartile == q]
        other_s = sub_other.set_index(['store_id','product_id'])['hourly_wape']
        common = best_s_q.index.intersection(other_s.index)
        a = best_s_q.loc[common].values; b = other_s.loc[common].values
        valid = ~(np.isnan(a) | np.isnan(b))
        a, b = a[valid], b[valid]
        if len(a) < 100: continue
        d_obs = cliffs_delta(a, b)
        boots = bootstrap_cliffs(a, b, n_boot=n_boot_q, seed=SEED+hash(r.cell)%10000)
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        decision = tost_decision(ci_lo, ci_hi)
        rows_pq.append({
            'level': f'quartile_{q}', 'best_cell': best_q_cell,
            'other_cell': r.cell,
            'cliffs_delta_obs': d_obs, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'tost_decision': decision,
            'threshold_equiv': abs(d_obs) < MARGIN,
        })
        if decision == 'EQUIVALENT': n_eq += 1
        elif decision == 'NOT_EQUIVALENT': n_ne += 1
        else: n_in += 1
    print(f'     TOST: EQUIVALENT={n_eq}, INCONCLUSIVE={n_in}, NOT_EQUIV={n_ne}')

rq = pd.DataFrame(rows_pq)

# ---------------------------------------------------------------------
# Save and summary
# ---------------------------------------------------------------------
out = pd.concat([rg, rq], ignore_index=True)
out.to_parquet(f'{RESULTS_DIR}/tost_equivalence.parquet', index=False)
print(f'\n5. Saved: {RESULTS_DIR}/tost_equivalence.parquet ({len(out)} comparisons)')

# Comparison threshold vs TOST
print('\n6. CONCORDANCE between threshold-equiv and TOST:')
both = out[(out.threshold_equiv) & (out.tost_decision=='EQUIVALENT')]
thr_only = out[(out.threshold_equiv) & (out.tost_decision!='EQUIVALENT')]
tost_only = out[(~out.threshold_equiv) & (out.tost_decision=='EQUIVALENT')]
print(f'   Both threshold AND TOST agree EQUIV: {len(both)}')
print(f'   Threshold ONLY (TOST inconclusive/not-equiv): {len(thr_only)}')
print(f'   TOST ONLY (threshold says non-equiv):  {len(tost_only)}')

print('\nDONE')
