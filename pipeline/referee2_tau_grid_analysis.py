"""
referee2_tau_grid_analysis.py — Second referee report, Major #1 (the decisive test).

Is the imputer still immaterial for the lag-based forecasters once they are NON-degenerate,
i.e. trained with a pinball loss at tau = 0.7 (WPE ~ -0.1/-0.2) instead of MAE (WPE ~ -0.9)?

For each lag-based family (LGB-M5, MLP-M5) and each loss (MAE = the paper's matrix cells;
pinball tau = 0.6 / 0.7 = the censored_aware_* runs with IMPUTER set), restricts to the
14 cells {no_imp + 13 imputers} and reports, exactly as RQ1 (script 48):
  Friedman chi2, Kendall's W = chi2 / (N (k-1)), Nemenyi CD, and whether no_imp lies in the
  equivalence set of the best imputer; plus the cross-imputer spread of the median WAPE/WPE,
  and W on the DAILY WAPE (referee #2: daily grain as primary).
Works on partial grids (reports k), so it can be run while the overnight job is in progress.

Output: results/referee2_tau_grid_W.parquet
"""
import os, glob, functools
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import studentized_range
print = functools.partial(print, flush=True)

RES = os.path.join(os.path.dirname(__file__), 'results')
KEY = ['store_id', 'product_id']
IMPS = ['itransformer', 'mediana_glob', 'media_glob', 'saits', 'dlinear', 'forward_fill', 'timesnet',
        'lgb', 'imputeformer', 'seasonal_naive', 'linear_interp', 'media_cond', 'mediana_cond']


def cells_for(fam, loss):
    out = {}
    for imp in ['no_imp'] + IMPS:
        if loss == 'mae':
            f = f'{RES}/{fam}_m5lags_hpo_test_per_series.parquet' if imp == 'no_imp' \
                else f'{RES}/{imp}__{fam}_m5lags_hpo_test_per_series.parquet'
        else:
            f = f'{RES}/censored_{fam}_m5_q{loss}_hpo_imp-{imp}_test_per_series.parquet'
        if os.path.exists(f):
            out[imp] = pd.read_parquet(f).set_index(KEY)
    return out


def friedman_block(cells, col):
    M = pd.DataFrame({k: v[col] for k, v in cells.items()}).dropna()
    k, N = M.shape[1], len(M)
    chi2, _ = stats.friedmanchisquare(*[M[c].values for c in M.columns])
    W = chi2 / (N * (k - 1))
    R = M.rank(axis=1).mean().sort_values()
    CD = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
    return W, CD, R, N


rows = []
for fam in ['lgb', 'mlp']:
    for loss in ['mae', '0.6', '0.7']:
        cells = cells_for(fam, loss)
        if len(cells) < 3:
            print(f'{fam.upper()}-M5 | {loss:>4}: only {len(cells)} cells available, skipped'); continue
        W, CD, R, N = friedman_block(cells, 'hourly_wape')
        Wd, _, _, Nd = friedman_block(cells, 'daily_wape')
        med = pd.Series({k: v['hourly_wape'].median() for k, v in cells.items()})
        wpe = pd.Series({k: v['hourly_wpe'].median() for k, v in cells.items()})
        dmed = pd.Series({k: v['daily_wape'].median() for k, v in cells.items()})
        best = R.index[0]
        noimp_in = ('no_imp' in R.index) and (R['no_imp'] - R.iloc[0] <= CD)
        rows.append(dict(family=f'{fam.upper()}-M5', loss='MAE' if loss == 'mae' else f'pinball tau={loss}',
                         k=len(cells), N=N, kendall_W=W, kendall_W_daily=Wd, CD=CD, best_imputer=best,
                         no_imp_in_equiv_set=noimp_in,
                         wape_med_min=med.min(), wape_med_max=med.max(), wape_spread=med.max() - med.min(),
                         wpe_med_mean=wpe.mean(), daily_wape_med_min=dmed.min(),
                         daily_wape_spread=dmed.max() - dmed.min()))
        r = rows[-1]
        print(f"{r['family']} | {r['loss']:<16} k={r['k']:2d} N={N:,} | W_hourly={W:.3f} W_daily={Wd:.3f} | "
              f"WAPE_med {med.min():.3f}-{med.max():.3f} (spread {r['wape_spread']:.3f}) | "
              f"WPE_med~{r['wpe_med_mean']:+.2f} | daily WAPE {dmed.min():.3f} (spread {r['daily_wape_spread']:.3f}) | "
              f"best={best}, no_imp in CD set: {noimp_in}")

out = pd.DataFrame(rows)
out.to_parquet(f'{RES}/referee2_tau_grid_W.parquet', index=False)
print('\nReading guide: the paper reports W ~ 0 (negligible, < 0.1) for both lag-based families under MAE.'
      '\nIf W stays < 0.1 at tau = 0.7 the imputer is immaterial also for the NON-degenerate forecasters.')
