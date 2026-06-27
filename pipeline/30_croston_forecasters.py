"""Intermittent-demand baselines (referee R2.1): Croston, SBA, TSB.

Design (chosen): a daily demand rate is estimated per series with the Croston
family on the daily-total series (sum over the 17 operational hours of the
completed series), then distributed over the day by the series' mean normalised
intra-day profile, and held flat across the 7 test days (direct protocol, like
the naive aggregates). The smoothing parameter alpha is selected per cell on the
validation horizon (days 84-90) by the same criterion used elsewhere: minimum
per-series median in-stock WAPE.

Crossed with all 14 imputers (13 completed-sales files + no_imp = raw observed).
Outputs results/{imputer}__{croston,sba,tsb}_test_per_series.parquet.
Set SMOKE=1 to run on the first 2 imputers only.
"""
import os, sys, functools, time, glob
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
t0 = time.time()
PR = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(PR, 'data'); COMP = os.path.join(DATA, 'completed_sales_622')
RES = os.path.join(os.path.dirname(__file__), 'results')
H0, H1 = 6, 23; NH = H1 - H0
ALPHAS = [0.05, 0.1, 0.2, 0.3]
IMPUTERS = ['no_imp', 'dlinear', 'forward_fill', 'imputeformer', 'itransformer', 'lgb',
            'linear_interp', 'media_cond', 'media_glob', 'mediana_cond', 'mediana_glob',
            'saits', 'seasonal_naive', 'timesnet']
if os.getenv('SMOKE') == '1': IMPUTERS = IMPUTERS[:2]

print('Loading base data...')
def load(fn):
    d = pd.read_parquet(os.path.join(DATA, fn)); d['dt_parsed'] = pd.to_datetime(d['dt'])
    return d
tr = pd.concat([load('frn50k_train.parquet'), load('frn50k_eval.parquet')], ignore_index=True)
tr = tr.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)
dates = sorted(tr['dt_parsed'].unique()); d2n = {d: i + 1 for i, d in enumerate(dates)}
tr['day_num'] = tr['dt_parsed'].map(d2n)
S = tr[['store_id', 'product_id']].drop_duplicates().shape[0]
ND = len(tr) // S
sid = tr['store_id'].values.reshape(S, ND)[:, 0]
pid = tr['product_id'].values.reshape(S, ND)[:, 0]
days = tr['day_num'].values.reshape(S, ND)[0]                       # 1..97 shared
sales_raw = np.array(tr['hours_sale'].tolist(), np.float64)[:, H0:H1].reshape(S, ND, NH)
stock = np.array(tr['hours_stock_status'].tolist(), np.int8)[:, H0:H1].reshape(S, ND, NH)
dt_str = tr['dt'].values.reshape(S, ND)
print(f'  {S:,} series x {ND} days')

def load_completed(imp):
    if imp == 'no_imp':
        return sales_raw
    df = pd.read_parquet(os.path.join(COMP, f'{imp}.parquet'))
    cs = np.array(df['hours_sale'].tolist(), np.float64)            # 17-h vectors
    key_c = (df['store_id'].astype(str) + '_' + df['product_id'].astype(str) + '_' + df['dt']).values
    km = {k: i for i, k in enumerate(key_c)}
    out = sales_raw.copy()
    flat_keys = (np.repeat(sid, ND).astype(str) + '_' + np.repeat(pid, ND).astype(str) + '_' + dt_str.reshape(-1))
    for j, k in enumerate(flat_keys):
        if k in km: out[j // ND, j % ND] = cs[km[k]]
    return out

# ---- vectorised Croston-family rate on daily totals ----
def croston_rate(D, alpha, variant):
    Sn, T = D.shape
    z = np.zeros(Sn); p = np.ones(Sn); q = np.zeros(Sn); seen = np.zeros(Sn, bool)
    for t in range(T):
        d = D[:, t]; nz = d > 0; q += 1.0
        first = nz & ~seen
        z[first] = d[first]; p[first] = q[first]; seen[first] = True; q[first] = 0.0
        upd = nz & seen & ~first
        z[upd] = alpha * d[upd] + (1 - alpha) * z[upd]
        p[upd] = alpha * q[upd] + (1 - alpha) * p[upd]
        q[upd] = 0.0
    rate = np.where(seen, z / np.maximum(p, 1e-9), 0.0)
    if variant == 'sba': rate = (1 - alpha / 2.0) * rate
    return rate

def tsb_rate(D, alpha):
    Sn, T = D.shape
    z = np.zeros(Sn); prob = np.zeros(Sn); seen = np.zeros(Sn, bool)
    for t in range(T):
        d = D[:, t]; nz = d > 0
        first = nz & ~seen
        z[first] = d[first]; prob[first] = 1.0; seen[first] = True
        upd = nz & seen & ~first
        z[upd] = alpha * d[upd] + (1 - alpha) * z[upd]
        act = seen & ~first
        prob[act] = alpha * nz[act].astype(float) + (1 - alpha) * prob[act]
    return prob * z

def rate_of(D, alpha, variant):
    return tsb_rate(D, alpha) if variant == 'tsb' else croston_rate(D, alpha, variant)

def per_series_wape(pred, obs, stk):
    """pred,obs,stk: (S, nday, NH). Returns per-series hourly WAPE/WPE on in-stock."""
    ins = stk == 0
    ae = np.where(ins, np.abs(pred - obs), 0.0).sum((1, 2))
    ao = np.where(ins, np.abs(obs), 0.0).sum((1, 2))
    e = np.where(ins, pred - obs, 0.0).sum((1, 2))
    o = np.where(ins, obs, 0.0).sum((1, 2))
    wape = np.where(ao > 0, ae / ao, np.nan)
    wpe = np.where(o != 0, e / o, np.nan)
    return wape, wpe, ae, ao, e, o

VARIANTS = ['croston', 'sba', 'tsb']
val_m = (days >= 84) & (days <= 90); te_m = (days >= 91) & (days <= 97)
summary = []
for imp in IMPUTERS:
    comp = load_completed(imp)                                       # (S,ND,NH)
    daily = comp.sum(2)                                              # (S,ND)
    # normalised intra-day profile from days<=anchor
    def profile(anchor):
        m = days <= anchor
        mean_h = comp[:, m, :].mean(1)                               # (S,NH)
        ssum = mean_h.sum(1, keepdims=True)
        return np.where(ssum > 0, mean_h / ssum, 1.0 / NH)
    prof_val = profile(83); prof_te = profile(90)
    obs_val = sales_raw[:, val_m, :]; stk_val = stock[:, val_m, :]; nval = val_m.sum()
    obs_te = sales_raw[:, te_m, :]; stk_te = stock[:, te_m, :]; nte = te_m.sum()
    Dval = daily[:, days <= 83]; Dte = daily[:, days <= 90]
    for v in VARIANTS:
        ck = f'{imp}__{v}'
        out_path = os.path.join(RES, f'{ck}_test_per_series.parquet')
        if os.path.exists(out_path): print(f'  SKIP {ck}'); continue
        # alpha selection on validation by median per-series WAPE
        best_a, best_med = None, np.inf
        for a in ALPHAS:
            r = rate_of(Dval, a, v)
            pv = (r[:, None, None] * prof_val[:, None, :]).repeat(nval, 1)
            w, _, _, _, _, _ = per_series_wape(pv, obs_val, stk_val)
            med = np.nanmedian(w)
            if med < best_med: best_med, best_a = med, a
        # test prediction with selected alpha
        r = rate_of(Dte, best_a, v)
        pt = (r[:, None, None] * prof_te[:, None, :]).repeat(nte, 1)
        w, wp, ae, ao, e, o = per_series_wape(pt, obs_te, stk_te)
        ps = pd.DataFrame({'store_id': sid, 'product_id': pid, 'hourly_wape': w, 'hourly_wpe': wp})
        ps.to_parquet(out_path, index=False)
        pooled = ae.sum() / ao.sum(); pooled_wpe = e.sum() / o.sum()
        med_w = np.nanmedian(w); med_wp = np.nanmedian(wp)
        summary.append((ck, best_a, pooled, med_w, pooled_wpe, med_wp))
        print(f'  {ck:28s} a*={best_a}  WAPE pool={pooled:.4f} med={med_w:.4f}  WPE pool={pooled_wpe:+.4f} med={med_wp:+.4f}')

print('\n=== SUMMARY (sorted by median WAPE) ===')
for ck, a, pl, mw, plw, mww in sorted(summary, key=lambda x: x[3]):
    print(f'  {ck:28s} a*={a}  WAPE_med={mw:.4f}  WPE_med={mww:+.4f}')
print(f'\ntime={time.time()-t0:.0f}s')
