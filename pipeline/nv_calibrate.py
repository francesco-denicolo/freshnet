"""Referee point #4 --- critical-fractile calibration of the newsvendor analysis.

The base newsvendor scoring compares the raw point-forecast daily order q against the
reference demand under asymmetric cost. A rational newsvendor instead orders the
critical fractile F^{-1}(c_u/(c_u+c_o)); comparing uncalibrated point forecasts under
asymmetric cost therefore measures how accidentally well-calibrated each cell is, not
how much information it carries. Here we fit, per cell and per cost ratio r, a SINGLE
multiplicative calibration factor k on the VALIDATION split (leakage-free) that
minimises validation newsvendor cost, apply it to the TEST orders, and re-score. If
the large uncalibrated gap (e.g. lag cells at r=5) collapses after calibration, the
lever is calibration, not the imputer or even the forecaster family.

Requires (from the NV_OVERNIGHT=1 re-run): newsvendor_qval_{cell}.parquet per lag cell,
plus newsvendor_yref_val.parquet (build_nv_ref_val.py) and newsvendor_yref.parquet.
Run: nv_calibrate.py
"""
import os, glob, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
RES = os.path.join(os.path.dirname(__file__), 'results')
RATIOS = [1.0, 2.0, 5.0]
KGRID = np.exp(np.linspace(np.log(0.3), np.log(8.0), 400))   # multiplicative factors to search
KEY = ['store_id', 'product_id', 'day_idx']


def cost(q, y, cu, co=1.0):
    return co * np.clip(q - y, 0, None) + cu * np.clip(y - q, 0, None)


def per_series_median_cost(q, y, sid, pid, cu):
    c = cost(q, y, cu)
    return float(pd.Series(c).groupby([sid, pid]).sum().median())


def fit_k(qv, yv, cu):
    """One scalar k minimising pooled validation cost at ratio r=cu (co=1)."""
    best_k, best_c = 1.0, np.inf
    for k in KGRID:
        c = cost(k * qv, yv, cu).sum()
        if c < best_c:
            best_c, best_k = c, k
    return best_k


yref_val = pd.read_parquet(f'{RES}/newsvendor_yref_val.parquet')
yref_test = pd.read_parquet(f'{RES}/newsvendor_yref.parquet')

# lag cells that have validation orders (produced by the NV_OVERNIGHT re-run)
val_files = sorted(glob.glob(f'{RES}/newsvendor_qval_*.parquet'))
print(f'{len(val_files)} calibratable lag cells\n')

rows = []
for vf in val_files:
    cell = os.path.basename(vf)[len('newsvendor_qval_'):-len('.parquet')]
    tf = f'{RES}/newsvendor_q_{cell}.parquet'
    if not os.path.exists(tf):
        print(f'  SKIP {cell}: no test orders'); continue
    qv = pd.read_parquet(vf).merge(yref_val, on=KEY, how='inner')
    qt = pd.read_parquet(tf).merge(yref_test, on=KEY, how='inner')
    imp, fc = cell.replace('_hpo', '').split('__', 1)
    for r in RATIOS:
        k = fit_k(qv['q'].values, qv['y_star'].values, cu=r)
        c_un = per_series_median_cost(qt['q'].values, qt['y_star'].values, qt['store_id'].values, qt['product_id'].values, r)
        c_ca = per_series_median_cost(k * qt['q'].values, qt['y_star'].values, qt['store_id'].values, qt['product_id'].values, r)
        rows.append({'cell': cell, 'imputer': imp, 'forecaster': fc, 'r': r,
                     'k_star': k, 'cost_uncal': c_un, 'cost_cal': c_ca})
    print(f'  {cell}: k*(r=1,2,5)=' + ', '.join(f'{d["k_star"]:.2f}' for d in rows[-3:]))

df = pd.DataFrame(rows)
out = f'{RES}/newsvendor_calibrated.parquet'
df.to_parquet(out, index=False)

# reference naive/intermittent cells (uncalibrated), for the gap comparison
print('\n=== naive/intermittent reference cells (uncalibrated) ===')
naive = {}
for cell in ['mediana_glob__dow_mean', 'linear_interp__sba']:
    p = f'{RES}/newsvendor_q_{cell}.parquet'
    if not os.path.exists(p):
        continue
    m = pd.read_parquet(p).merge(yref_test, on=KEY, how='inner')
    naive[cell] = {r: per_series_median_cost(m['q'].values, m['y_star'].values, m['store_id'].values, m['product_id'].values, r) for r in RATIOS}
    print(f'  {cell}: ' + ', '.join(f'r={int(r)} {naive[cell][r]:.2f}' for r in RATIOS))

print('\n=== lag-family median newsvendor cost: uncalibrated -> calibrated ===')
for fc, g in df.groupby('forecaster'):
    print(f'  {fc}:')
    for r in RATIOS:
        gr = g[g.r == r]
        print(f'    r={int(r)}: median cost {gr.cost_uncal.median():.2f} -> {gr.cost_cal.median():.2f}  '
              f'(within-imputer spread {gr.cost_uncal.max()-gr.cost_uncal.min():.2f} -> {gr.cost_cal.max()-gr.cost_cal.min():.2f}; '
              f'median k* {gr.k_star.median():.2f})')

print(f'\nsaved {out}')
print('Interpretation: if calibrated lag cost approaches the naive reference, the gap the '
      'referee flagged (e.g. 20.8 vs 4.3 at r=5) is a calibration artefact, not an imputer/forecaster one.')
