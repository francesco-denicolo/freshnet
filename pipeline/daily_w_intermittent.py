"""Referee-review check #1: is the imputer immaterial at DAILY granularity for the
INTERMITTENT-demand forecasters too (Croston/SBA/TSB), or only for the naive
aggregates and lag-based models? tab:app_daily stores daily metrics for 6 of 11
forecasters; the intermittent methods are missing, yet §4.6 claims immateriality
"for every forecaster we test". Since an intermittent forecast IS a daily rate
built from the imputed daily totals, its daily imputer effect may NOT collapse.

We reconstruct the direct intermittent forecast per imputer, score daily in-stock
WAPE per series, and compute Kendall's W across the 14 imputers, for each of
Croston/SBA/TSB. Prints hourly-vs-daily W so we can compare to the naive collapse
(0.45 -> 0.05).
"""
import os, functools, numpy as np, pandas as pd
print = functools.partial(print, flush=True)
PR = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(PR, 'data'); COMP = os.path.join(DATA, 'completed_sales_622')
RES = os.path.join(os.path.dirname(__file__), 'results')
H0, H1 = 6, 23; NH = H1 - H0
ALPHA = 0.1
IMPUTERS = ['no_imp','media_glob','media_cond','mediana_glob','mediana_cond','forward_fill',
            'seasonal_naive','linear_interp','lgb','dlinear','saits','itransformer','timesnet','imputeformer']

def load(fn):
    d = pd.read_parquet(os.path.join(DATA, fn)); d['dt_parsed'] = pd.to_datetime(d['dt'])
    return d.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)

print('Loading base...')
tr = load('frn50k_train.parquet'); ev = load('frn50k_eval.parquet')
key = tr[['store_id','product_id']].drop_duplicates().reset_index(drop=True)
S = len(key); NDT = len(tr)//S; NDE = len(ev)//S
ev_sales = np.array(ev['hours_sale'].tolist(), np.float32)[:, H0:H1].reshape(S, NDE, NH)
ev_stock = np.array(ev['hours_stock_status'].tolist(), np.float32)[:, H0:H1].reshape(S, NDE, NH)
ev_instock = ev_stock == 0  # (S, NDE, NH)
print(f'  {S:,} series, test days={NDE}')

def completed_cube(imp):
    if imp == 'no_imp':
        c = tr
    else:
        c = pd.read_parquet(os.path.join(COMP, f'{imp}.parquet'))
        c['dt_parsed'] = pd.to_datetime(c['dt'])
        c = c.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
    arr = np.array(c['hours_sale'].tolist(), np.float32)
    if arr.shape[1] == 24: arr = arr[:, H0:H1]
    return arr.reshape(S, NDT, NH)

def croston_family(D, alpha, variant):
    """D: (S, NDT) daily totals -> per-series rate at anchor."""
    Sn, T = D.shape
    z = np.where(D[:,0] > 0, D[:,0], 0.0).astype(np.float64)
    p = np.ones(Sn); q = np.ones(Sn); seen = D[:,0] > 0
    if variant == 'tsb':
        prob = (D[:,0] > 0).astype(float)
        zt = np.where(D[:,0] > 0, D[:,0], 0.0).astype(np.float64)
        for t in range(1, T):
            d = D[:,t]; act = d > 0
            zt[act] = alpha*d[act] + (1-alpha)*zt[act]
            prob = alpha*(act.astype(float)) + (1-alpha)*prob
        return prob * zt
    for t in range(1, T):
        d = D[:,t]; upd = d > 0
        z[upd] = alpha*d[upd] + (1-alpha)*z[upd]
        p[upd] = alpha*q[upd] + (1-alpha)*p[upd]
        q += 1; q[upd] = 1; seen |= upd
    rate = np.where(seen, z/np.maximum(p, 1e-9), 0.0)
    if variant == 'sba': rate = (1 - alpha/2.0)*rate
    return rate

def daily_wape_for(imp, variant):
    cube = completed_cube(imp)                       # (S, NDT, NH)
    D = cube.sum(2)                                  # (S, NDT) daily totals
    rate = croston_family(D, ALPHA, variant)         # (S,)
    # intra-day profile: mean normalised hourly shape over training days with sales
    daily = cube.sum(2, keepdims=True)               # (S,NDT,1)
    prof = np.divide(cube, np.maximum(daily, 1e-9))  # per-day normalised
    profile = prof.mean(1)                            # (S, NH) mean profile
    profile = profile / np.maximum(profile.sum(1, keepdims=True), 1e-9)
    # hourly forecast for each test day = rate * profile (same all test days)
    fc_h = rate[:, None] * profile                    # (S, NH)
    # daily in-stock: per test day, sum over in-stock hours
    fc_daily = (fc_h[:, None, :] * ev_instock).sum(2)      # (S, NDE)
    act_daily = (ev_sales * ev_instock).sum(2)            # (S, NDE)
    num = np.abs(fc_daily - act_daily).sum(1)             # (S,)
    den = np.abs(act_daily).sum(1)
    w = np.where(den > 0, num/den, np.nan)
    return pd.Series(w, index=pd.MultiIndex.from_frame(key))

def kendall_w(M):
    ranks = M.rank(axis=1); n, k = M.shape
    Rsum = ranks.sum(0)
    return 12*((Rsum - n*(k+1)/2)**2).sum() / (n**2 * k * (k**2-1))

for variant in ['croston','sba','tsb']:
    cols = {}
    for imp in IMPUTERS:
        cols[imp] = daily_wape_for(imp, variant)
    M = pd.DataFrame(cols).dropna()
    W = kendall_w(M)
    # hourly W from stored per-series parquets for comparison
    hcols = {}
    for imp in IMPUTERS:
        f = f'{RES}/{imp}__{variant}_test_per_series.parquet'
        if os.path.exists(f):
            hcols[imp] = pd.read_parquet(f).set_index(['store_id','product_id'])['hourly_wape']
    Hm = pd.DataFrame(hcols).dropna(); Wh = kendall_w(Hm) if len(hcols) else float('nan')
    print(f'{variant.upper():8s}: W(hourly)={Wh:.3f}  W(daily)={W:.3f}   daily WAPE med(best)={M.median(0).min():.3f}')
