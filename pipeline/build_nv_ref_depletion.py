"""MAJOR 2(b): a STRUCTURALLY DIFFERENT newsvendor reference y*.
Instead of filling test stock-out hours with a smooth conditional-median imputer, we
fill them with a parametric estimate of the censored tail: the per-(series,hour)
in-stock baseline INFLATED by the hourly depletion/endogeneity factor R(h)---the
ratio, at each operational hour, of in-stock demand in hours preceding a same-day
stock-out to in-stock demand on days without one (the probe of Section 3.4). This
lets us check that the newsvendor conclusions are not an artefact of a mean/median
reference. Saved to newsvendor_yref_depletion.parquet."""
import os, functools, time
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
t0 = time.time()
PR = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(PR, 'data'); RES = os.path.join(os.path.dirname(__file__), 'results')
H0, H1 = 6, 23; NH = H1 - H0

def load(fn):
    d = pd.read_parquet(os.path.join(DATA, fn)); d['dt_parsed'] = pd.to_datetime(d['dt']); return d

print('Loading...')
tr = load('frn50k_train.parquet').sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
ev = load('frn50k_eval.parquet').sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
key_tr = tr[['store_id','product_id']].drop_duplicates().reset_index(drop=True)
S = len(key_tr); ND_TR = len(tr)//S; ND_EV = len(ev)//S
print(f'  {S:,} series, train days={ND_TR}, eval days={ND_EV}')

def cube(d, nd):
    sa = np.array(d['hours_sale'].tolist(), dtype=np.float32)[:, H0:H1]
    st = np.array(d['hours_stock_status'].tolist(), dtype=np.float32)[:, H0:H1]
    return sa.reshape(S, nd, NH), st.reshape(S, nd, NH)
tr_sa, tr_st = cube(tr, ND_TR); ev_sa, ev_st = cube(ev, ND_EV)

# ---- per-(series,hour) conditional-median in-stock baseline (same as the plain ref) ----
masked = np.where(tr_st == 0, tr_sa, np.nan)
with np.errstate(all='ignore'):
    ref = np.nanmedian(masked, axis=1)                        # (S,17)
glob_hour = np.nanmedian(masked.reshape(-1, NH), axis=0); glob_hour = np.where(np.isnan(glob_hour), 0.0, glob_hour)
nanmask = np.isnan(ref); ref[nanmask] = np.broadcast_to(glob_hour, ref.shape)[nanmask]

# ---- hourly depletion/endogeneity factor R(h) from the TRAIN data ----
# A(h): in-stock hour h on a (series,day) that runs out later that same day
# B(h): in-stock hour h on a (series,day) with no stock-out at any later hour
print('Computing hourly depletion factor R(h)...')
instock = tr_st == 0                                          # (S,90,17)
future_so = np.zeros_like(tr_st, bool)
acc = np.zeros((S, ND_TR), bool)
for h in range(NH-1, -1, -1):
    future_so[:, :, h] = acc
    acc = acc | (tr_st[:, :, h] == 1)
A = instock & future_so; B = instock & (~future_so)
Rh = np.ones(NH)
for h in range(NH):
    a = tr_sa[:, :, h][A[:, :, h]]; b = tr_sa[:, :, h][B[:, :, h]]
    if len(a) > 50 and len(b) > 50 and b.mean() > 0:
        Rh[h] = float(a.mean() / b.mean())
Rh = np.clip(Rh, 1.0, None)                                   # depletion can only inflate demand
print('  R(h) by operational hour:', ' '.join(f'{H0+h}:{Rh[h]:.2f}' for h in range(NH)))

# ---- y*_depletion: test stock-out hours filled with ref[s,h] * R(h) ----
fill = (ref * Rh[None, :])[:, None, :]                        # (S,1,17)
completed = np.where(ev_st == 0, ev_sa, fill)                 # in-stock observed, stock-out inflated
ystar = completed.sum(axis=2)
obs_daily = np.where(ev_st == 0, ev_sa, 0.0).sum(axis=2)

sid = key_tr['store_id'].values; pid = key_tr['product_id'].values
rows = pd.DataFrame({'store_id': np.repeat(sid, ND_EV),'product_id': np.repeat(pid, ND_EV),
                     'day_idx': np.tile(np.arange(ND_EV), S),
                     'y_star': ystar.reshape(-1).astype(np.float64),
                     'y_obs': obs_daily.reshape(-1).astype(np.float64)})
out = os.path.join(RES, 'newsvendor_yref_depletion.parquet')
rows.to_parquet(out, index=False)
print(f'\nsaved {out}  ({len(rows):,} rows)')
print(f'  y_star: mean={rows.y_star.mean():.3f} median={rows.y_star.median():.3f} '
      f'| uplift(y*/y_obs)={rows.y_star.sum()/max(rows.y_obs.sum(),1e-9):.3f}')
# compare to the plain conditional-median reference if present
p = os.path.join(RES, 'newsvendor_yref.parquet')
if os.path.exists(p):
    base = pd.read_parquet(p)
    print(f'  plain-ref uplift for comparison: {base.y_star.sum()/max(base.y_obs.sum(),1e-9):.3f}')
print(f'time={time.time()-t0:.0f}s')
