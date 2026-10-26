"""
referee3_target_completed.py — internal review (Oct 2026), point (b): does the imputer matter for the
lag-based forecasters when it also enters their TRAINING TARGET?

In the benchmark, LGB-M5 and MLP-M5 are trained on observed sales (stock-out zeros included) and receive
the imputer only through the lag features. Here the same production cells are re-trained with the
completed series as target (TARGET_COMPLETED=1, CELL_SUFFIX=_tgtcomp; run_overnight_target.sh).
For each family, over {no imputation + 13 imputers} (no imputation is identical under both targets):
  Kendall's W (hourly and daily WAPE), spread of median WAPE / WPE, and the paired Cliff's delta of each
  imputer against no imputation, read against the 20-seed band (hourly |delta| up to 0.11 LGB / 0.13 MLP).
Compared with the input-only channel of the paper (W = 0.009 / 0.029). Works on partial runs.
Output: results/referee3_target_completed.parquet
"""
import os, functools
import numpy as np, pandas as pd
from scipy import stats
print = functools.partial(print, flush=True)
RES = os.path.join(os.path.dirname(__file__), 'results')
K = ['store_id', 'product_id']
IMPS = ['itransformer', 'mediana_glob', 'media_glob', 'saits', 'dlinear', 'forward_fill', 'timesnet',
        'lgb', 'imputeformer', 'seasonal_naive', 'linear_interp', 'media_cond', 'mediana_cond']
SEED = {'lgb': 0.11, 'mlp': 0.13}
SFX = '_tgtcomp' + ('_SMOKE' if os.getenv('SMOKE') == '1' else '')
rows = []
for fam in ['lgb', 'mlp']:
    for target, sfx in [('observed (paper)', ''), ('completed', SFX)]:
        cells = {'no_imp': pd.read_parquet(f'{RES}/{fam}_m5lags_hpo_test_per_series.parquet').set_index(K)}
        for imp in IMPS:
            f = f'{RES}/{imp}__{fam}_m5lags_hpo{sfx}_test_per_series.parquet'
            if os.path.exists(f):
                cells[imp] = pd.read_parquet(f).set_index(K)
        if len(cells) < 3:
            print(f'{fam.upper()}-M5 target={target}: {len(cells) - 1} imputer cells, skipped'); continue
        out = dict(family=f'{fam.upper()}-M5', target=target, k=len(cells))
        for col in ['hourly_wape', 'daily_wape']:
            M = pd.DataFrame({k: v[col] for k, v in cells.items()}).dropna()
            chi2, _ = stats.friedmanchisquare(*[M[c].values for c in M.columns])
            out[f'W_{col}'] = chi2 / (len(M) * (M.shape[1] - 1))
        H = pd.DataFrame({k: v['hourly_wape'] for k, v in cells.items()}).dropna()
        med = H.median(); wpe = pd.Series({k: v['hourly_wpe'].median() for k, v in cells.items()})
        dl = {i: float(((H['no_imp'] - H[i]) > 0).mean() - ((H['no_imp'] - H[i]) < 0).mean()) for i in H if i != 'no_imp'}
        beyond = {i: round(d, 2) for i, d in dl.items() if abs(d) > SEED[fam]}
        out.update(wape_spread=med.max() - med.min(), wape_min=med.min(), wape_max=med.max(),
                   wpe_mean_imputers=wpe.drop('no_imp').mean(), wpe_noimp=wpe['no_imp'],
                   n_beyond_seed=len(beyond), beyond_seed=str(beyond))
        rows.append(out)
        print(f"{out['family']} | target={target:<16} k={out['k']:2d} | W hourly={out['W_hourly_wape']:.3f} "
              f"daily={out['W_daily_wape']:.3f} | WAPE_med {med.min():.3f}-{med.max():.3f} (spread {out['wape_spread']:.3f}) | "
              f"WPE no-imp {wpe['no_imp']:+.2f}, imputers {out['wpe_mean_imputers']:+.2f} | beyond seed band: {beyond}")
if rows:
    pd.DataFrame(rows).to_parquet(f'{RES}/referee3_target_completed{"_SMOKE" if os.getenv("SMOKE") == "1" else ""}.parquet', index=False)
