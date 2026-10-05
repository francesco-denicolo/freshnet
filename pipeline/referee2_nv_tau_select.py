"""
referee2_nv_tau_select.py — Second referee report, round 2.

(A) Select the pinball level tau on VALIDATION order cost, separately for each underage/overage
    ratio r, and report the TEST order cost at the selected tau. Direct forecaster with no imputer,
    tau in {0.5, ..., 0.9}, for LGB-M5 and MLP-M5 (run_overnight_nv_tau.sh, RUN_TAG=_nvsel).
(C) Seed band of the order cost at tau = 0.7: the base run (seed 42) plus SEED=1..5, compared with
    the cross-imputer cost spread at tau = 0.7 (referee2_nv_tau_grid.parquet).

Daily order = sum of the 17 hourly forecasts of the day; cost against the paper's reference demand
(newsvendor_yref.parquet on test days 91-97, newsvendor_yref_val.parquet on validation days 84-90,
built from days 1-83 only); summary = cross-series median of the per-series total cost (nv_cost.py).
Works on partial runs. Outputs: results/referee2_nv_tau_select.parquet, referee2_nv_seedband.parquet
"""
import os, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

RES = os.path.join(os.path.dirname(__file__), 'results')
PRED = os.path.join(RES, 'preds')
RATIOS = [1.0, 2.0, 5.0]
TAUS = ['0.5', '0.6', '0.7', '0.8', '0.9']
TAG = os.getenv('RUN_TAG', '_nvsel')
SMOKE = '_SMOKE' if os.getenv('SMOKE') == '1' else ''
REF = {'test': pd.read_parquet(f'{RES}/newsvendor_yref.parquet'),
       'val': pd.read_parquet(f'{RES}/newsvendor_yref_val.parquet')}


def cell(fam, tau, seed=''):
    return f'censored_{fam}_m5_q{tau}_hpo_imp-no_imp{TAG}{seed}{SMOKE}'


def order_cost(npz, split):
    if not os.path.exists(npz):
        return None
    z = np.load(npz)
    n = len(z['store_id'])
    q = pd.DataFrame({'store_id': z['store_id'], 'product_id': z['product_id'],
                      'day_idx': np.tile(np.arange(7), n // 7), 'q': z['preds'].sum(1).astype(np.float64)})
    m = q.merge(REF[split], on=['store_id', 'product_id', 'day_idx'], how='inner')
    out = {}
    for r in RATIOS:
        c = np.clip(m['q'] - m['y_star'], 0, None) + r * np.clip(m['y_star'] - m['q'], 0, None)
        out[r] = float(c.groupby([m['store_id'], m['product_id']]).sum().median())
    return out


# ---------------- (A) tau selected on validation cost ----------------
rows = []
for fam in ['lgb', 'mlp']:
    for tau in TAUS:
        c = cell(fam, tau)
        v = order_cost(f'{PRED}/{c}_VAL.npz', 'val')
        t = order_cost(f'{PRED}/{c}.npz', 'test')
        if v is None or t is None:
            continue
        for r in RATIOS:
            rows.append(dict(family=f'{fam.upper()}-M5', tau=float(tau), r=r, val_cost=v[r], test_cost=t[r]))
A = pd.DataFrame(rows)
if len(A):
    A.to_parquet(f'{RES}/referee2_nv_tau_select.parquet', index=False)
    print('(A) order cost by tau (validation | test), per-series median total cost')
    for (fam, r), g in A.groupby(['family', 'r']):
        g = g.sort_values('tau')
        best = g.loc[g.val_cost.idxmin()]
        grid = '  '.join(f"tau={t:.1f}: {v:.2f}|{x:.2f}" for t, v, x in zip(g.tau, g.val_cost, g.test_cost))
        print(f'  {fam} r={r:g}: {grid}')
        print(f'      -> tau* (validation) = {best.tau:.1f}, test cost {best.test_cost:.2f}'
              f'  [critical fractile r/(1+r) = {r / (1 + r):.2f}; {len(g)} tau levels available]')
else:
    print('(A) no complete cells yet')

# ---------------- (C) seed band of the order cost at tau = 0.7 ----------------
rows = []
grid = pd.read_parquet(f'{RES}/referee2_nv_tau_grid.parquet') if os.path.exists(f'{RES}/referee2_nv_tau_grid.parquet') else None
print('\n(C) seed band of the test order cost at tau = 0.7 (no imputer)')
for fam in ['lgb', 'mlp']:
    costs = {}
    for seed in [''] + [f'_seed{n}' for n in range(1, 6)]:
        t = order_cost(f'{PRED}/{cell(fam, "0.7", seed)}.npz', 'test')
        if t is not None:
            costs[seed or '_seed42'] = t
    if len(costs) < 2:
        print(f'  {fam.upper()}-M5: {len(costs)} seed(s) available, band not computed'); continue
    for r in RATIOS:
        vals = np.array([c[r] for c in costs.values()])
        imp_spread = np.nan
        if grid is not None:
            g = grid[(grid.family == f'{fam.upper()}-M5') & (grid.tau == 0.7)][f'cost_r{r:g}']
            imp_spread = float(g.max() - g.min()) if len(g) else np.nan
        rows.append(dict(family=f'{fam.upper()}-M5', r=r, n_seeds=len(vals), seed_min=vals.min(), seed_max=vals.max(),
                         seed_range=vals.max() - vals.min(), seed_sd=vals.std(ddof=1), imputer_spread_tau07=imp_spread))
        print(f'  {fam.upper()}-M5 r={r:g}: {len(vals)} seeds, cost {vals.min():.2f}-{vals.max():.2f} '
              f'(range {vals.max() - vals.min():.2f}, sd {vals.std(ddof=1):.2f}) | cross-imputer spread at tau=0.7: {imp_spread:.2f}')
if rows:
    pd.DataFrame(rows).to_parquet(f'{RES}/referee2_nv_seedband.parquet', index=False)
