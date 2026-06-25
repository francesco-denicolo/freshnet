"""Newsvendor cost evaluation. For each cell's exported daily order q(s,d), and the
reference completed daily demand y*(s,d) (newsvendor_yref.parquet), compute the
asymmetric order cost at several underage/overage ratios r = c_u/c_o, summarised by
the cross-series median of the per-series total cost. Tabulates cells and the spread
across imputers within each lag-based forecaster, and compares the cost ranking to WAPE.

Usage: nv_cost.py            -> evaluate every newsvendor_q_*.parquet found
       nv_cost.py <glob>     -> restrict to matching cells
"""
import os, sys, glob, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
RES = os.path.join(os.path.dirname(__file__), 'results')
RATIOS = [1.0, 2.0, 5.0]

def cost(q, y, cu, co=1.0):
    return co*np.clip(q-y, 0, None) + cu*np.clip(y-q, 0, None)

yref = pd.read_parquet(f'{RES}/newsvendor_yref.parquet')   # store_id, product_id, day_idx, y_star, y_obs
pat = sys.argv[1] if len(sys.argv) > 1 else 'newsvendor_q_*.parquet'
files = sorted(glob.glob(os.path.join(RES, pat)))
print(f'{len(files)} cell(s)\n')

rows = []
for f in files:
    cell = os.path.basename(f)[len('newsvendor_q_'):-len('.parquet')]
    q = pd.read_parquet(f)
    m = q.merge(yref, on=['store_id', 'product_id', 'day_idx'], how='inner')
    if len(m) == 0:
        print(f'  SKIP {cell}: no overlap with yref'); continue
    rec = {'cell': cell, 'n_series': m[['store_id', 'product_id']].drop_duplicates().shape[0]}
    for r in RATIOS:
        c = cost(m['q'].values, m['y_star'].values, cu=r)
        per_series = pd.Series(c).groupby([m['store_id'].values, m['product_id'].values]).sum()
        rec[f'cost_r{r:g}'] = float(per_series.median())
    # secondary: cost vs observed-only demand (in-stock), to show the weighting effect
    c1 = cost(m['q'].values, m['y_obs'].values, cu=1.0)
    rec['cost_r1_obs'] = float(pd.Series(c1).groupby([m['store_id'].values, m['product_id'].values]).sum().median())
    rows.append(rec)

df = pd.DataFrame(rows).sort_values('cost_r2').reset_index(drop=True)
pd.set_option('display.width', 160)
print(df.to_string(index=False))

# parse imputer/forecaster and report spread across imputers within each forecaster
def split(cell):
    imp, fc = cell.split('__', 1)
    return imp, fc.replace('_hpo', '')
df['imputer'] = df['cell'].map(lambda c: split(c)[0])
df['forecaster'] = df['cell'].map(lambda c: split(c)[1])
print('\n=== between-imputer spread of newsvendor cost (median per-series, r=2) ===')
for fc, g in df.groupby('forecaster'):
    if len(g) < 2: continue
    v = g['cost_r2'].values
    print(f'  {fc}: n={len(g)} | spread(max-min)={v.max()-v.min():.4f} '
          f'(min {v.min():.4f} [{g.loc[g.cost_r2.idxmin(),"imputer"]}], '
          f'max {v.max():.4f} [{g.loc[g.cost_r2.idxmax(),"imputer"]}])')

out = f'{RES}/newsvendor_cost_summary.parquet'
df.to_parquet(out, index=False)
print(f'\nsaved {out}')
