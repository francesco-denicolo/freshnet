"""Build the reference completed daily demand y*(series, val-day) on the VALIDATION
horizon (days 84-90) for the critical-fractile calibration of the newsvendor analysis
(referee point #4). Leakage-free: the per-(series,hour) conditional-median reference is
computed from train days 1-83 only (the val days are never used to build their own
reference). Mirrors build_nv_ref.py but for the validation split.
Output: newsvendor_yref_val.parquet with day_idx 0..6 == val days 84..90."""
import os, functools, time
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
t0 = time.time()
PR = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(PR, 'data'); RES = os.path.join(os.path.dirname(__file__), 'results')
H0, H1 = 6, 23; NH = H1 - H0
REF_DAYS = 83          # reference median from days 1..83 (0-indexed 0..82)
VAL_LO, VAL_HI = 84, 90  # validation days (1-indexed)

print('Loading train...')
tr = pd.read_parquet(os.path.join(DATA, 'frn50k_train.parquet'))
tr['dt_parsed'] = pd.to_datetime(tr['dt'])
tr = tr.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)
key = tr[['store_id', 'product_id']].drop_duplicates().reset_index(drop=True)
S = len(key); ND = len(tr) // S
print(f'  {S:,} series, {ND} train days/series')
assert len(tr) == S * ND and ND == 90, f'unexpected shape ND={ND}'

sa = np.array(tr['hours_sale'].tolist(), dtype=np.float32)[:, H0:H1].reshape(S, ND, NH)
st = np.array(tr['hours_stock_status'].tolist(), dtype=np.float32)[:, H0:H1].reshape(S, ND, NH)

# reference: per (series,hour) median of in-stock sales over days 1..83 ONLY (leakage-free for val)
print(f'Computing conditional-median reference from days 1..{REF_DAYS}...')
masked = np.where(st[:, :REF_DAYS, :] == 0, sa[:, :REF_DAYS, :], np.nan)   # (S,83,17)
with np.errstate(all='ignore'):
    ref = np.nanmedian(masked, axis=1)                                     # (S,17)
glob_hour = np.nanmedian(masked.reshape(-1, NH), axis=0)
glob_hour = np.where(np.isnan(glob_hour), 0.0, glob_hour)
nanmask = np.isnan(ref); ref[nanmask] = np.broadcast_to(glob_hour, ref.shape)[nanmask]
print(f'  ref NaN cells filled by global hour median: {int(nanmask.sum()):,}/{ref.size:,}')

# validation completed demand on days 84..90 (0-indexed 83..89)
vs = sa[:, VAL_LO - 1:VAL_HI, :]      # (S,7,17)
vt = st[:, VAL_LO - 1:VAL_HI, :]
NV = vs.shape[1]
completed = np.where(vt == 0, vs, ref[:, None, :])
ystar = completed.sum(axis=2)                    # (S,7)
obs_daily = np.where(vt == 0, vs, 0.0).sum(axis=2)

sid = key['store_id'].values; pid = key['product_id'].values
rows = pd.DataFrame({
    'store_id': np.repeat(sid, NV), 'product_id': np.repeat(pid, NV),
    'day_idx': np.tile(np.arange(NV), S),        # 0..6 == val days 84..90
    'y_star': ystar.reshape(-1).astype(np.float64),
    'y_obs': obs_daily.reshape(-1).astype(np.float64)})
out = os.path.join(RES, 'newsvendor_yref_val.parquet')
rows.to_parquet(out, index=False)
print(f'\nsaved {out}  ({len(rows):,} rows)')
print(f'  y_star: mean={rows.y_star.mean():.3f} median={rows.y_star.median():.3f} | time={time.time()-t0:.0f}s')
