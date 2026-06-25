"""Build the reference completed daily demand y*(series, day) on the test horizon
(days 91-97) for the newsvendor evaluation. Reference imputer = conditional median:
for each (series, hour) the median of in-stock sales over the 90 train days. Test
stock-out hours are filled with this reference; in-stock test hours keep observed
sales. y*(s,d) = sum over the 17 operational hours. Saved to newsvendor_yref.parquet."""
import os, functools, time
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
t0 = time.time()
PR = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(PR, 'data'); RES = os.path.join(os.path.dirname(__file__), 'results')
H0, H1 = 6, 23; NH = H1 - H0

def load(fn):
    d = pd.read_parquet(os.path.join(DATA, fn))
    d['dt_parsed'] = pd.to_datetime(d['dt'])
    return d

print('Loading...')
tr = load('frn50k_train.parquet'); ev = load('frn50k_eval.parquet')
tr = tr.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)
ev = ev.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)

# series order must match between train and eval
key_tr = tr[['store_id', 'product_id']].drop_duplicates().reset_index(drop=True)
key_ev = ev[['store_id', 'product_id']].drop_duplicates().reset_index(drop=True)
assert key_tr.equals(key_ev), 'series mismatch train/eval'
S = len(key_tr); print(f'  {S:,} series')
ND_TR = len(tr) // S; ND_EV = len(ev) // S
print(f'  train days/series={ND_TR}, eval days/series={ND_EV}')
assert len(tr) == S * ND_TR and len(ev) == S * ND_EV

def cube(d, nd):
    sa = np.array(d['hours_sale'].tolist(), dtype=np.float32)[:, H0:H1]
    st = np.array(d['hours_stock_status'].tolist(), dtype=np.float32)[:, H0:H1]
    return sa.reshape(S, nd, NH), st.reshape(S, nd, NH)

print('Building cubes...')
tr_sa, tr_st = cube(tr, ND_TR)            # (S, 90, 17)
ev_sa, ev_st = cube(ev, ND_EV)            # (S, 7, 17)

# reference: per (series, hour) median of in-stock (stock==0) train sales
print('Computing conditional-median reference...')
masked = np.where(tr_st == 0, tr_sa, np.nan)          # (S,90,17) NaN where stock-out
with np.errstate(all='ignore'):
    ref = np.nanmedian(masked, axis=1)                # (S,17)
# fallback for (series,hour) with no in-stock train obs: global per-hour median
glob_hour = np.nanmedian(masked.reshape(-1, NH), axis=0)   # (17,)
glob_hour = np.where(np.isnan(glob_hour), 0.0, glob_hour)
nanmask = np.isnan(ref)
ref[nanmask] = np.broadcast_to(glob_hour, ref.shape)[nanmask]
print(f'  ref NaN cells filled by global hour median: {int(nanmask.sum()):,}/{ref.size:,}')

# y*(s,d) = sum_h ( observed if in-stock else ref[s,h] )
refb = ref[:, None, :]                                  # (S,1,17)
completed = np.where(ev_st == 0, ev_sa, refb)           # (S,7,17)
ystar = completed.sum(axis=2)                           # (S,7)

# also the "observed-only" daily demand (in-stock sales summed) for reference/diagnostics
obs_daily = np.where(ev_st == 0, ev_sa, 0.0).sum(axis=2)

sid = key_tr['store_id'].values; pid = key_tr['product_id'].values
rows = pd.DataFrame({
    'store_id': np.repeat(sid, ND_EV),
    'product_id': np.repeat(pid, ND_EV),
    'day_idx': np.tile(np.arange(ND_EV), S),            # 0..6 == test days 91..97
    'y_star': ystar.reshape(-1).astype(np.float64),
    'y_obs': obs_daily.reshape(-1).astype(np.float64),
})
out = os.path.join(RES, 'newsvendor_yref.parquet')
rows.to_parquet(out, index=False)
print(f'\nsaved {out}  ({len(rows):,} rows)')
print(f'  y_star: mean={rows.y_star.mean():.3f} median={rows.y_star.median():.3f} '
      f'| y_obs: mean={rows.y_obs.mean():.3f} | uplift(y*/y_obs)={rows.y_star.sum()/max(rows.y_obs.sum(),1e-9):.3f}')
print(f'time={time.time()-t0:.0f}s')
