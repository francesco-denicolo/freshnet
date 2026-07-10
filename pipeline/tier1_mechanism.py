"""Tier-1 mechanism validation (refined). Do the rolling lag features absorb the
cross-imputer disagreement present in the stock-out fills?

The 13 completed series are identical on in-stock hours and differ only on
stock-out hours. For each series we form, under each imputer, a ladder of
increasingly time-aggregated daily-total features at the test anchor (day 90):
  lag_1d   = daily total of day 90              (no time averaging)
  rmean_7d = mean daily total over days 84-90
  rmean_14d= mean daily total over days 77-90   (most aggregated)
We measure the CROSS-IMPUTER dispersion of each, normalised by a common, stable,
imputer-independent per-series scale L_s (mean in-stock daily total on train),
and take the median over series. The mechanism predicts a monotone decrease
lag_1d > rmean_7d > rmean_14d: the more a feature aggregates, the more it absorbs
the imputer disagreement. Cross-referenced with LGB feature importance, the
high-importance features (the rolling means) are the low-dispersion ones."""
import os, functools, time
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
t0 = time.time()
PR = os.path.join(os.path.dirname(__file__), '..'); DATA = os.path.join(PR, 'data')
COMP = os.path.join(DATA, 'completed_sales_622'); RES = os.path.join(os.path.dirname(__file__), 'results')
H0, H1 = 6, 23; NH = H1 - H0
IMPS = ['dlinear','forward_fill','imputeformer','itransformer','lgb','linear_interp',
        'media_cond','media_glob','mediana_cond','mediana_glob','saits','seasonal_naive','timesnet']

def load(fn):
    d = pd.read_parquet(os.path.join(DATA, fn)); d['dt_parsed'] = pd.to_datetime(d['dt']); return d
tr = pd.concat([load('frn50k_train.parquet'), load('frn50k_eval.parquet')], ignore_index=True)
tr = tr.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
dates = sorted(tr['dt_parsed'].unique()); d2n = {d: i+1 for i, d in enumerate(dates)}
tr['day_num'] = tr['dt_parsed'].map(d2n)
S = tr[['store_id','product_id']].drop_duplicates().shape[0]; ND = len(tr)//S
sid = tr['store_id'].values.reshape(S, ND)[:,0]; pid = tr['product_id'].values.reshape(S, ND)[:,0]
days = tr['day_num'].values.reshape(S, ND)[0]
sales = np.array(tr['hours_sale'].tolist(), np.float64)[:, H0:H1].reshape(S, ND, NH)
stock = np.array(tr['hours_stock_status'].tolist(), np.int8)[:, H0:H1].reshape(S, ND, NH)
ins = stock == 0
# common per-series scale: mean in-stock daily total over training days 2-90
trm = (days >= 2) & (days <= 90)
daily_obs = np.where(ins, sales, 0.0).sum(2)               # (S,ND) in-stock daily total
L = daily_obs[:, trm].mean(1)                              # (S,)
L = np.where(L > 1e-9, L, np.nan)
print(f'{S} series, L_s median={np.nanmedian(L):.3f}, t={time.time()-t0:.0f}s')

def completed(imp):
    df = pd.read_parquet(os.path.join(COMP, f'{imp}.parquet'))
    cs = np.array(df['hours_sale'].tolist(), np.float64)
    km = {k: i for i, k in enumerate((df.store_id.astype(str)+'_'+df.product_id.astype(str)+'_'+df.dt).values)}
    out = sales.copy()
    fk = (np.repeat(sid, ND).astype(str)+'_'+np.repeat(pid, ND).astype(str)+'_'+tr['dt'].values)
    for j, k in enumerate(fk):
        if k in km: out[j//ND, j%ND] = cs[km[k]]
    return out

w7  = (days >= 84) & (days <= 90)
w14 = (days >= 77) & (days <= 90)
d90 = days == 90
feats = {'lag_1d': [], 'rmean_7d': [], 'rmean_14d': []}     # each -> (S, n_imp)
for imp in IMPS:
    c = completed(imp); D = c.sum(2)                        # (S,ND) completed daily total
    feats['lag_1d'].append(D[:, d90].mean(1))
    feats['rmean_7d'].append(D[:, w7].mean(1))
    feats['rmean_14d'].append(D[:, w14].mean(1))
    print(f'  {imp:15s} done ({time.time()-t0:.0f}s)')

def disp(stack):                                            # cross-imputer std / L, median over series
    M = np.column_stack(stack)                              # (S, n_imp)
    sd = np.nanstd(M, 1)
    rel = sd / L
    return np.nanmedian(rel)

print('\n' + '='*60)
print('TIER-1 (refined): cross-imputer dispersion (std/L_s, median over series)')
order = ['lag_1d', 'rmean_7d', 'rmean_14d']
vals = {f: disp(feats[f]) for f in order}
for f in order:
    print(f'  {f:11s}: {vals[f]*100:.1f}%  of series in-stock level')
print(f'  --- compression lag_1d -> rmean_14d: {vals["lag_1d"]/vals["rmean_14d"]:.1f}x ---')
# stratify by stock-out rate (where imputation matters most)
so_rate = (stock[:, trm, :] == 1).mean((1,2))
hi = so_rate >= np.nanmedian(so_rate)
def disp_sub(stack, mask):
    M = np.column_stack(stack); sd = np.nanstd(M, 1); rel = sd / L
    return np.nanmedian(rel[mask])
print('  high-stockout half of series:')
for f in order:
    print(f'    {f:11s}: {disp_sub(feats[f], hi)*100:.1f}%')
print(f'\ntime={time.time()-t0:.0f}s')
