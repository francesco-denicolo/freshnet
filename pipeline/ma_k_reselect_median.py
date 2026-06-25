"""
Re-select MA window K using the per-series MEDIAN val WAPE criterion
(consistent with the Optuna HPO of the ML/DL forecasters, min_hours=34),
and reproduce the pooled-WAPE selection as a sanity check.

Validation: days 84-90, anchor day 83, in-stock hours only, operational hours 6-22.
"""
import os, functools, numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
H_START, H_END = 6, 23
N_HOURS = H_END - H_START
MA_K_CANDIDATES = [3, 5, 7, 10, 14, 21, 28, 42, 56, 83]
MIN_HOURS_VAL = 34

print('Loading data...')
df = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'),
                     columns=['store_id', 'product_id', 'dt', 'hours_sale', 'hours_stock_status'])
df['dt_parsed'] = pd.to_datetime(df['dt'])
df = df.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)
dates = sorted(df['dt_parsed'].unique())
d2d = {d: i + 1 for i, d in enumerate(dates)}
df['day_num'] = df['dt_parsed'].map(d2d)

sales_all = np.array(df['hours_sale'].tolist(), dtype=np.float64)[:, H_START:H_END]
stock_all = np.array(df['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]

print('Building per-series caches...')
series = []
for (sid, pid), grp in df.groupby(['store_id', 'product_id'], sort=False):
    gs = grp.sort_values('day_num')
    idx = gs.index.values
    days = df.loc[idx, 'day_num'].values
    sa = sales_all[idx]; st = stock_all[idx]
    tr = days <= 83
    va = (days >= 84) & (days <= 90)
    if not tr.any() or not va.any():
        continue
    obs_v = sa[va]; ins_v = (st[va] == 0)
    series.append({
        'train_sales': sa[tr],          # (n_tr, 17) sorted by day
        'obs_v': obs_v[ins_v],          # in-stock val observations (flat)
        'n_in': int(ins_v.sum()),
        'ins_v': ins_v,                 # (n_val,17) bool
        'obs_v_full': obs_v,            # (n_val,17)
    })
print(f'  {len(series):,} series with both train and val data')

print('\nGrid search (pooled vs per-series median, min_hours=34):')
print(f'{"K":>4} | {"WAPE_pooled":>12} | {"WAPE_median":>12} | {"n_series_med":>12}')
res = {}
for K in MA_K_CANDIDATES:
    sae = 0.0; sao = 0.0
    per = []
    for s in series:
        ts = s['train_sales']
        prof = ts[-min(K, ts.shape[0]):].mean(axis=0)   # (17,)
        ins = s['ins_v']
        # broadcast profile across val days, take in-stock entries
        pred_in = np.broadcast_to(prof, ins.shape)[ins]
        obs_in = s['obs_v']
        num = np.abs(pred_in - obs_in).sum()
        den = np.abs(obs_in).sum()
        sae += num; sao += den
        if s['n_in'] >= MIN_HOURS_VAL and den > 0:
            per.append(num / den)
    wp = sae / sao if sao > 0 else np.nan
    wm = float(np.median(per)) if per else np.nan
    res[K] = (wp, wm)
    print(f'{K:>4} | {wp:>12.5f} | {wm:>12.5f} | {len(per):>12,}')

best_pool = min(res, key=lambda k: res[k][0])
best_med = min(res, key=lambda k: res[k][1])
print(f'\nBest K (pooled)  : {best_pool}  (WAPE_pooled={res[best_pool][0]:.5f})')
print(f'Best K (median)  : {best_med}  (WAPE_median={res[best_med][1]:.5f})')
