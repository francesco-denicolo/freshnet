"""
rq2_chronos_q4_mechanism.py — test the proposed mechanism for Chronos-bolt's Q4 inversion
=========================================================================================
RQ2 reports that Chronos-bolt's recovery->forecasting concordance turns INVERSE in the
highest-volume quartile (P=0.33). The proposed mechanism: on dense series the model's
predictive median saturates, so imputers whose reconstructions inject MORE variability
around the hourly profile displace that median and help, whereas smooth high-fidelity
fills reinforce it.

Direct test (referee-requested): rank the 13 recovery imputers by a reconstruction-dynamism
index (DYN = std of stock-out fills / std of observed in-stock sales, per series, median
across series; from 42d_tft_dynamics_analysis.py) and correlate it with Chronos-bolt's
per-imputer forecasting WAPE within each volume quartile.

Prediction: DYN vs Q4 WAPE is NEGATIVE (more dynamic -> lower WAPE -> better), and the
association is absent at low volume (Q1).
"""
import os, glob, functools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
print = functools.partial(print, flush=True)

RES = os.path.join(os.path.dirname(__file__), 'results')

dyn = pd.read_parquet(f'{RES}/rq2_imputer_dynamicity.parquet')[['imputer', 'DYN_median']]
strat = pd.read_parquet(f'{RES}/stratification.parquet')

rows = []
for f in sorted(glob.glob(f'{RES}/*__chronos_bolt_test_per_series.parquet')):
    imp = os.path.basename(f).replace('__chronos_bolt_test_per_series.parquet', '')
    d = pd.read_parquet(f)
    rec = {'imputer': imp}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        keys = strat[strat.vol_bin == q][['store_id', 'product_id']]
        rec[f'chronos_{q.lower()}_wape'] = d.merge(keys, on=['store_id', 'product_id'])['hourly_wape'].dropna().median()
    rows.append(rec)
ch = pd.DataFrame(rows)

M = dyn.merge(ch, on='imputer').dropna(subset=['DYN_median']).sort_values('DYN_median')
print(M.to_string(index=False))

print('\nSpearman(DYN_median, Chronos WAPE) per quartile:')
out = []
for q in ['q1', 'q2', 'q3', 'q4']:
    col = f'chronos_{q}_wape'
    sub = M.dropna(subset=[col])
    rho, p = spearmanr(sub['DYN_median'], sub[col])
    out.append({'quartile': q.upper(), 'spearman_rho': rho, 'p_value': p, 'n': len(sub)})
    print(f'  {q.upper()}: rho={rho:+.3f}  p={p:.3f}  n={len(sub)}')
res = pd.DataFrame(out)
res.to_parquet(f'{RES}/rq2_chronos_q4_mechanism.parquet', index=False)
M.to_parquet(f'{RES}/rq2_chronos_q4_dynamism_wape.parquet', index=False)
print('\nsaved rq2_chronos_q4_mechanism.parquet + rq2_chronos_q4_dynamism_wape.parquet')
print('\nReading: negative rho in Q4 supports the saturated-median mechanism; '
      'near-zero/positive rho in Q1 confirms it is specific to the high-volume regime.')
