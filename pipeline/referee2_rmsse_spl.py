"""
referee2_rmsse_spl.py — Second referee report, Major #2: scale-free metrics for intermittent demand.

WAPE/MAE reward forecasts shrunk towards zero on zero-inflated series (Kolassa 2016); the M5
accuracy track used RMSSE and the uncertainty track the scaled pinball loss (SPL). This script
re-scores the imputer x tau grid (hourly test predictions saved by run_overnight_tau_grid.sh in
results/preds/*.npz) on:
  - RMSSE hourly : sqrt( mean_{in-stock test hours} e^2 / mean((Δy)^2) ), scale on the hourly
                   observed series of days 1-90 (hours 6-22), from the first non-zero sale (M5)
  - RMSSE daily  : same on daily totals (forecast/obs summed over the in-stock hours of each test
                   day, as the paper's daily WAPE); scale on the daily totals of days 1-90
  - SPL hourly   : mean pinball_tau(e) over in-stock test hours / mean|Δy| (hourly scale), with
                   tau = the training level of the block (0.6 or 0.7) — comparable WITHIN a block
Then, for each (family, tau) block of 14 cells (no_imp + 13 imputers), Friedman / Kendall W and
the paired Cliff's delta of every imputer against no-imputation, exactly as for WAPE.

Output: results/referee2_rmsse_spl_per_series.parquet (long), results/referee2_rmsse_spl_W.parquet
"""
import os, glob, functools
import numpy as np, pandas as pd
from scipy import stats
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(os.path.dirname(__file__), 'results')
PRED = os.path.join(RES, 'preds')
H_START, H_END = 6, 23
IMPS = ['no_imp', 'itransformer', 'mediana_glob', 'media_glob', 'saits', 'dlinear', 'forward_fill',
        'timesnet', 'lgb', 'imputeformer', 'seasonal_naive', 'linear_interp', 'media_cond', 'mediana_cond']

# ---------------------------------------------------------------- scales from days 1-90
print('1. Scales from the observed series (days 1-90, hours 6-22)...')
df = pd.read_parquet(os.path.join(PROJECT_ROOT, 'data', 'frn50k_train.parquet'),
                     columns=['store_id', 'product_id', 'dt', 'hours_sale'])
df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
Y = np.array(df['hours_sale'].tolist(), dtype=np.float64)[:, H_START:H_END]      # (rows, 17)
keys = df[['store_id', 'product_id']].values
starts = np.r_[0, np.flatnonzero((keys[1:] != keys[:-1]).any(axis=1)) + 1]
ends = np.r_[starts[1:], len(df)]
scale = {}
for s, e in zip(starts, ends):
    h = Y[s:e].ravel(); d = Y[s:e].sum(1)
    nzh = np.flatnonzero(h > 0); nzd = np.flatnonzero(d > 0)
    if len(nzh) < 2 or len(nzd) < 2:
        continue
    hh, dd = h[nzh[0]:], d[nzd[0]:]
    scale[(keys[s, 0], keys[s, 1])] = (np.mean(np.diff(hh) ** 2), np.mean(np.abs(np.diff(hh))), np.mean(np.diff(dd) ** 2))
sc = pd.DataFrame.from_dict(scale, orient='index', columns=['s2_h', 's1_h', 's2_d'])
sc.index = pd.MultiIndex.from_tuples(sc.index, names=['store_id', 'product_id'])
sc = sc[(sc > 0).all(axis=1)]
print(f'   {len(sc):,} series with a valid (non-zero) scale')
del df, Y


def score(path, tau):
    z = np.load(path)
    P, O, S = z['preds'].astype(np.float64), z['obs'].astype(np.float64), z['stock']
    sid, pid = z['store_id'][::7], z['product_id'][::7]
    n = len(sid)
    P, O, M = P.reshape(n, 7, 17), O.reshape(n, 7, 17), (S.reshape(n, 7, 17) == 0)
    E = (P - O) * M
    nin = M.sum((1, 2))
    mse_h = (E ** 2).sum((1, 2)) / np.maximum(nin, 1)
    pin = np.where(E < 0, tau * (-E), (1 - tau) * E) * M          # pinball on e = pred - obs
    spl_raw = pin.sum((1, 2)) / np.maximum(nin, 1)
    dP, dO, dn = (P * M).sum(2), (O * M).sum(2), M.any(2)
    mse_d = (((dP - dO) ** 2) * dn).sum(1) / np.maximum(dn.sum(1), 1)
    out = pd.DataFrame({'store_id': sid, 'product_id': pid, 'mse_h': mse_h, 'spl_raw': spl_raw,
                        'mse_d': mse_d, 'n_in': nin}).set_index(['store_id', 'product_id'])
    out = out[out.n_in > 0].join(sc, how='inner')
    return pd.DataFrame({'rmsse_h': np.sqrt(out.mse_h / out.s2_h),
                         'rmsse_d': np.sqrt(out.mse_d / out.s2_d),
                         'spl_h': out.spl_raw / out.s1_h}, index=out.index)


print('\n2. Scoring the 56 cells...')
long, wrows = [], []
for fam in ['lgb', 'mlp']:
    for tau in ['0.6', '0.7']:
        cells = {}
        for imp in IMPS:
            f = f'{PRED}/censored_{fam}_m5_q{tau}_hpo_imp-{imp}.npz'
            if os.path.exists(f):
                cells[imp] = score(f, float(tau))
        for imp, d in cells.items():
            long.append(d.assign(family=fam, tau=float(tau), imputer=imp).reset_index())
        for met in ['rmsse_h', 'rmsse_d', 'spl_h']:
            M = pd.DataFrame({i: d[met] for i, d in cells.items()}).replace([np.inf, -np.inf], np.nan).dropna()
            k, N = M.shape[1], len(M)
            chi2, _ = stats.friedmanchisquare(*[M[c].values for c in M.columns])
            W = chi2 / (N * (k - 1))
            med = M.median()
            dl = {i: float(((M['no_imp'] - M[i]) > 0).mean() - ((M['no_imp'] - M[i]) < 0).mean())
                  for i in M.columns if i != 'no_imp'}
            mx = max(dl, key=lambda x: abs(dl[x]))
            wrows.append(dict(family=f'{fam.upper()}-M5', tau=float(tau), metric=met, k=k, N=N, kendall_W=W,
                              med_no_imp=med['no_imp'], med_min=med.min(), med_max=med.max(),
                              best_imputer=med.idxmin(), max_abs_delta_vs_noimp=dl[mx], max_delta_imputer=mx))
            r = wrows[-1]
            print(f"   {r['family']} tau={tau} {met:8s} k={k} N={N:,} W={W:.3f} | median no_imp {med['no_imp']:.3f}, "
                  f"imputers {med.drop('no_imp').min():.3f}-{med.drop('no_imp').max():.3f} (best {med.idxmin()}) | "
                  f"max δ vs no_imp {dl[mx]:+.2f} ({mx})")

pd.concat(long, ignore_index=True).to_parquet(f'{RES}/referee2_rmsse_spl_per_series.parquet', index=False)
pd.DataFrame(wrows).to_parquet(f'{RES}/referee2_rmsse_spl_W.parquet', index=False)
print('\nsaved referee2_rmsse_spl_per_series.parquet, referee2_rmsse_spl_W.parquet')
