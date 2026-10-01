"""
referee2_nv_tau_grid.py — Second referee report, Major #3: order-level (newsvendor) cost of the
imputer x tau grid. The daily order of each cell is the sum of its 17 hourly test forecasts
(as in the EXPORT_Q orders of censored_aware_mlp.py), scored against the paper's reference demand
(newsvendor_yref.parquet, Appendix C) at r = c_u/c_o in {1, 2, 5}, same cost as nv_cost.py.

Output: results/referee2_nv_tau_grid.parquet (one row per cell) and a per-block summary
(no-imputation cost, best imputer, cross-imputer spread).
"""
import os, glob, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

RES = os.path.join(os.path.dirname(__file__), 'results')
RATIOS = [1.0, 2.0, 5.0]
yref = pd.read_parquet(f'{RES}/newsvendor_yref.parquet')


def cost(q, y, cu, co=1.0):
    return co * np.clip(q - y, 0, None) + cu * np.clip(y - q, 0, None)


rows = []
for f in sorted(glob.glob(f'{RES}/preds/censored_*_hpo_imp-*.npz')):
    name = os.path.basename(f)[:-4]
    fam = 'LGB-M5' if '_lgb_' in name else 'MLP-M5'
    tau = float(name.split('_q')[1].split('_')[0])
    imp = name.split('_imp-')[1]
    z = np.load(f)
    q = pd.DataFrame({'store_id': z['store_id'], 'product_id': z['product_id'],
                      'day_idx': np.tile(np.arange(7), len(z['store_id']) // 7),
                      'q': z['preds'].sum(1).astype(np.float64)})
    m = q.merge(yref, on=['store_id', 'product_id', 'day_idx'], how='inner')
    rec = dict(family=fam, tau=tau, imputer=imp, n_series=m[['store_id', 'product_id']].drop_duplicates().shape[0])
    for r in RATIOS:
        c = cost(m['q'].values, m['y_star'].values, cu=r)
        rec[f'cost_r{r:g}'] = float(pd.Series(c).groupby([m['store_id'].values, m['product_id'].values]).sum().median())
    rows.append(rec)

df = pd.DataFrame(rows)
df.to_parquet(f'{RES}/referee2_nv_tau_grid.parquet', index=False)
print(f'{len(df)} cells scored\n')
for (fam, tau), g in df.groupby(['family', 'tau']):
    g = g.set_index('imputer')
    line = [f'{fam} tau={tau}:']
    for r in RATIOS:
        c = g[f'cost_r{r:g}']
        line.append(f'r={r:g}: no_imp {c["no_imp"]:.2f}, best {c.min():.2f} ({c.idxmin()}), spread {c.max() - c.min():.2f}')
    print('  ' + ' | '.join(line))
