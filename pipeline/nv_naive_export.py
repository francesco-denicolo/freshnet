"""Export daily order q(series, test-day) for the low-bias forecaster cells the
referee asks to add to the newsvendor evaluation (R2.3): DoW-Mean on mediana_glob
and SBA on linear_interp. Direct forecast (anchored at the training history), so
q is a per-series profile/rate applied to the 7 test days. Writes
newsvendor_q_{cell}.parquet, consumed by nv_cost.py.
"""
import os, functools, numpy as np, pandas as pd
print = functools.partial(print, flush=True)
PR = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(PR, 'data'); COMP = os.path.join(DATA, 'completed_sales_622')
RES = os.path.join(os.path.dirname(__file__), 'results')
H0, H1 = 6, 23; NH = H1 - H0

def load(fn):
    d = pd.read_parquet(os.path.join(DATA, fn)); d['dt_parsed'] = pd.to_datetime(d['dt'])
    return d.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)

print('Loading base...')
tr = load('frn50k_train.parquet'); ev = load('frn50k_eval.parquet')
key = tr[['store_id', 'product_id']].drop_duplicates().reset_index(drop=True)
S = len(key); NDT = len(tr) // S; NDE = len(ev) // S
print(f'  {S:,} series, train days={NDT}, test days={NDE}')
tr_dow = tr['dt_parsed'].dt.dayofweek.values.reshape(S, NDT)[0]   # same calendar for all series
ev_dow = ev['dt_parsed'].dt.dayofweek.values.reshape(S, NDE)[0]

def completed_cube(imp):
    """(S, NDT, NH) completed training sales for imputer `imp`, aligned to key order."""
    c = pd.read_parquet(os.path.join(COMP, f'{imp}.parquet'))
    c['dt_parsed'] = pd.to_datetime(c['dt'])
    c = c.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)
    arr = np.array(c['hours_sale'].tolist(), np.float32)
    if arr.shape[1] == 24: arr = arr[:, H0:H1]      # raw 24h -> operational
    assert arr.shape[1] == NH, f'{imp}: got {arr.shape[1]} hours'
    return arr.reshape(S, NDT, NH)

def save_q(cell, q_sd):
    """q_sd: (S, NDE) daily order forecast. Write long format."""
    sid = np.repeat(key['store_id'].values, NDE)
    pid = np.repeat(key['product_id'].values, NDE)
    di = np.tile(np.arange(NDE), S)
    out = pd.DataFrame({'store_id': sid, 'product_id': pid, 'day_idx': di, 'q': q_sd.ravel().astype(np.float64)})
    p = os.path.join(RES, f'newsvendor_q_{cell}.parquet')
    out.to_parquet(p, index=False)
    print(f'  saved {p}  mean q={out.q.mean():.3f}')

# ---- DoW Mean on mediana_glob ----
print('DoW-Mean / mediana_glob...')
cube = completed_cube('mediana_glob')                    # (S, NDT, NH)
prof = np.zeros((S, 7, NH), np.float32); cnt = np.zeros(7, np.int32)
for w in range(7):
    m = tr_dow == w
    if m.any(): prof[:, w] = cube[:, m].mean(1); cnt[w] = m.sum()
q_dow = np.stack([prof[:, ev_dow[d]].sum(1) for d in range(NDE)], axis=1)   # (S, NDE)
save_q('mediana_glob__dow_mean', q_dow)

# ---- SBA on linear_interp ----
print('SBA / linear_interp...')
def croston_rate(D, alpha, variant='sba'):
    """D: (S, NDT) daily totals. Returns per-series smoothed rate at the anchor."""
    S_, T = D.shape
    z = D[:, 0].astype(np.float64).copy(); p = np.ones(S_)
    q = np.ones(S_); seen = D[:, 0] > 0
    z = np.where(seen, D[:, 0], 0.0).astype(np.float64)
    for t in range(1, T):
        d = D[:, t]; upd = d > 0
        z[upd] = alpha * d[upd] + (1 - alpha) * z[upd]
        p[upd] = alpha * q[upd] + (1 - alpha) * p[upd]
        q += 1; q[upd] = 1; seen |= upd
    rate = np.where(seen, z / np.maximum(p, 1e-9), 0.0)
    if variant == 'sba': rate = (1 - alpha / 2.0) * rate
    return rate

cube_li = completed_cube('linear_interp')
D = cube_li.sum(2)                                        # (S, NDT) daily totals
# alpha selected per paper grid; use 0.1 (mid of {0.05,0.1,0.2,0.3}); direct rate applied to all test days
rate = croston_rate(D, alpha=0.1, variant='sba')          # (S,)
q_sba = np.repeat(rate[:, None], NDE, axis=1)              # (S, NDE)
save_q('linear_interp__sba', q_sba)
print('DONE')
