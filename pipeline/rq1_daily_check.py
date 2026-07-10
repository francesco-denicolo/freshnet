"""
rq1_daily_check.py — MAJOR 3 (referee): does the imputer's irrelevance survive at
DAILY granularity, where forecasting is genuinely informative (WAPE well below 1)?
Recomputes the per-forecaster Friedman + Kendall's W on daily_wape (already stored in
every cell) and compares with the hourly_wape result. No retraining.
"""
import os, glob, functools
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import studentized_range
print = functools.partial(print, flush=True)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
NON_HPO_FC = {'chronos_bolt','timesfm','global_mean','dow_mean','ma_k56','croston','sba','tsb'}

def parse_name(n): return n.split('__', 1) if '__' in n else ('no_imp', n)
def kcat(W):
    return 'negligible' if W<0.1 else 'small' if W<0.3 else 'moderate' if W<0.5 else 'large'

# ---- load cells (exclude capacity-sweep hicap artefacts) ----
per_series, seen = {}, set()
for f in sorted(glob.glob(f'{RESULTS_DIR}/*_hpo_test_per_series.parquet')):
    name = os.path.basename(f).replace('_hpo_test_per_series.parquet','')
    if 'hicap' in name: continue
    per_series[name] = pd.read_parquet(f); seen.add(name)
for f in sorted(glob.glob(f'{RESULTS_DIR}/*_test_per_series.parquet')):
    fn = os.path.basename(f)
    if '_hpo_test_per_series' in fn or 'hicap' in fn: continue
    name = fn.replace('_test_per_series.parquet','')
    if name in seen: continue
    if name.startswith('naive_'):
        base = name.replace('naive_','',1)
        if base in NON_HPO_FC:
            nn=f'no_imp__{base}'
            if nn not in seen: per_series[nn]=pd.read_parquet(f); seen.add(nn)
        continue
    imp, fc = parse_name(name)
    if fc not in NON_HPO_FC: continue
    per_series[name] = pd.read_parquet(f); seen.add(name)

cells_by_fc = {}
for name in per_series:
    imp, fc = parse_name(name); cells_by_fc.setdefault(fc, []).append((imp, name))

def friedman(W_sub):
    cols=list(W_sub.columns); k=len(cols); N=len(W_sub)
    if k<3 or N<50: return None
    chi2,_=stats.friedmanchisquare(*[W_sub[c].values for c in cols])
    Wk=chi2/(N*(k-1))
    CD=studentized_range.ppf(0.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*N))
    R=W_sub.rank(axis=1,method='average').mean(axis=0).sort_values()
    best=R.index[0]; br=R.iloc[0]
    equiv=[c for c in R.index if (R.loc[c]-br)<=CD]
    noimp=[c for c in cols if parse_name(c)[0]=='no_imp']
    noimp_in=any(c in equiv for c in noimp)
    return {'k':k,'N':N,'W':Wk,'CD':CD,'best':best,'n_equiv':len(equiv),
            'noimp_in_equiv':noimp_in, 'best_daily_med':None}

def run(metric):
    res={}
    for fc,cells in cells_by_fc.items():
        cols=[c for _,c in cells if metric in per_series[c].columns]
        if len(cols)<3: continue
        common=None
        for c in cols:
            idx=per_series[c].set_index(['store_id','product_id']).index
            common=idx if common is None else common.intersection(idx)
        W=pd.DataFrame(index=common,columns=cols,dtype=float)
        for c in cols:
            W[c]=per_series[c].set_index(['store_id','product_id'])[metric].loc[common].values
        W=W.dropna()
        r=friedman(W)
        if r is None: continue
        # median metric of the best cell (to show task is informative)
        r['best_med']=float(per_series[r['best']][metric].dropna().median())
        res[fc]=r
    return res

print('Computing per-forecaster Kendall W on HOURLY and DAILY wape...\n')
H=run('hourly_wape'); D=run('daily_wape')

order=['mlp_m5lags','lgb_m5lags','tft','chronos_bolt','timesfm',
       'global_mean','dow_mean','ma_k56','croston','sba','tsb']
print(f'{"forecaster":14s} | {"W_hour":>7s} {"cat":>10s} | {"W_day":>7s} {"cat":>10s} | {"day_med":>7s} | {"noimp∈eq(day)":>13s}')
print('-'*82)
for fc in order:
    if fc not in H or fc not in D: continue
    h,d=H[fc],D[fc]
    print(f'{fc:14s} | {h["W"]:7.3f} {kcat(h["W"]):>10s} | {d["W"]:7.3f} {kcat(d["W"]):>10s} | '
          f'{d["best_med"]:7.3f} | {str(d["noimp_in_equiv"]):>13s}')

# save
rows=[]
for fc in order:
    if fc in H and fc in D:
        rows.append({'forecaster':fc,'W_hourly':H[fc]['W'],'W_daily':D[fc]['W'],
                     'daily_best_med':D[fc]['best_med'],'daily_n_equiv':D[fc]['n_equiv'],
                     'daily_noimp_in_equiv':D[fc]['noimp_in_equiv'],
                     'daily_best':D[fc]['best'],'hourly_best':H[fc]['best']})
pd.DataFrame(rows).to_parquet(f'{RESULTS_DIR}/rq1_daily_vs_hourly.parquet',index=False)
print('\nSaved: rq1_daily_vs_hourly.parquet')
