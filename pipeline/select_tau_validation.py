"""Referee point #3: select the pinball level tau on the VALIDATION criterion of
Section 3.7 (per-series median in-stock WAPE, series with >=34 in-stock val hours),
then report that tau's TEST performance and its paired Cliff's delta against the
best two-stage cell of the same family. This removes the asymmetry the referee
flagged (tau read off the test sweep vs two-stage configs frozen on validation).

Run AFTER censored_aware_mlp.py / censored_aware_lgb.py have produced the
*_VAL_per_series.parquet files.
"""
import os, glob, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
RES = os.path.join(os.path.dirname(__file__), 'results')
TAUS = ['0.5', '0.6', '0.7', '0.8', '0.9']

# best two-stage cell of each family (paper tab:censored caption)
TWO_STAGE = {'mlp': 'timesnet__mlp_m5lags_hpo', 'lgb': 'mediana_cond__lgb_m5lags_hpo'}
KEY = ['store_id', 'product_id']


def cliff_delta(a, b):
    """Paired Cliff's delta of 'direct better than two-stage' on hourly_wape:
    fraction of series where direct < two-stage minus fraction where direct > two-stage."""
    d = a - b  # direct - two_stage ; negative => direct more accurate
    n = len(d)
    return (np.sum(d < 0) - np.sum(d > 0)) / n


def band(delta):
    a = abs(delta)
    return ('negligible' if a < 0.147 else 'small' if a < 0.33 else 'medium' if a < 0.474 else 'large')


for fam in ('mlp', 'lgb'):
    print(f'\n================  {fam.upper()}-M5  ================')
    # 1) validation criterion per tau
    val_med = {}
    for t in TAUS:
        vp = os.path.join(RES, f'censored_{fam}_m5_q{t}_hpo_VAL_per_series.parquet')
        if not os.path.exists(vp):
            print(f'  tau={t}: VAL file missing ({os.path.basename(vp)}) -- run the sweep first')
            continue
        v = pd.read_parquet(vp)
        m = v[v['n_instock'] >= 34]['val_hourly_wape'].dropna().median()
        val_med[t] = m
    if not val_med:
        continue
    for t in TAUS:
        if t in val_med:
            print(f'  tau={t}:  val median WAPE = {val_med[t]:.4f}')
    tau_star = min(val_med, key=val_med.get)
    print(f'  --> validation-selected tau* = {tau_star}  (val WAPE {val_med[tau_star]:.4f})')

    # 2) TEST performance of tau*
    tp = os.path.join(RES, f'censored_{fam}_m5_q{tau_star}_hpo_test_per_series.parquet')
    direct = pd.read_parquet(tp)
    tw = direct['hourly_wape'].dropna().median()
    tpe = direct['hourly_wpe'].dropna().median()
    print(f'      TEST at tau*:  median WAPE = {tw:.4f}   median WPE = {tpe:.4f}')

    # 3) paired Cliff's delta vs best two-stage cell of same family
    ts = pd.read_parquet(os.path.join(RES, f'{TWO_STAGE[fam]}_test_per_series.parquet'))
    mg = direct.merge(ts, on=KEY, suffixes=('_dir', '_ts')).dropna(subset=['hourly_wape_dir', 'hourly_wape_ts'])
    dlt = cliff_delta(mg['hourly_wape_dir'].values, mg['hourly_wape_ts'].values)
    ts_wape = ts['hourly_wape'].dropna().median()
    frac_better = float(np.mean(mg['hourly_wape_dir'].values < mg['hourly_wape_ts'].values))
    print(f'      two-stage ref ({TWO_STAGE[fam]}): median WAPE {ts_wape:.4f}')
    print(f'      paired Cliff delta (direct vs two-stage) = {dlt:+.3f} ({band(dlt)}); '
          f'direct more accurate on {100*frac_better:.0f}% of series')
    verdict = 'MATCHES two-stage (negligible/small)' if abs(dlt) < 0.33 else 'differs (>= medium)'
    print(f'      VERDICT at validation-selected tau*: {verdict}')
