"""Aggregate the overnight robustness results: seed range (Maj-5) and recursive
vs frozen within-forecaster imputer spread (Maj-4). Writes robustness_summary.txt."""
import os, functools, numpy as np, pandas as pd
print = functools.partial(print, flush=True)
RES = os.path.join(os.path.dirname(__file__), 'results')
IMPS = ['dlinear','forward_fill','imputeformer','itransformer','lgb','linear_interp',
        'media_cond','media_glob','mediana_cond','mediana_glob','saits','seasonal_naive','timesnet']
out_lines = []
def emit(s): print(s); out_lines.append(s)

def med(path):
    try: return float(pd.read_parquet(path)['hourly_wape'].dropna().median())
    except Exception: return None

emit('='*60); emit('SEED ROBUSTNESS (Maj-5) — itransformer cell, WAPE_h median')
for fc in ['lgb','mlp']:
    vals=[]
    f42=f'{RES}/itransformer__{fc}_m5lags_hpo_test_per_series.parquet'
    m=med(f42)
    if m is not None: vals.append(('seed42',m))
    for n in range(1,6):
        m=med(f'{RES}/seedrun_itransformer__{fc}_m5lags_hpo_seed{n}_test_per_series.parquet')
        if m is not None: vals.append((f'seed{n}',m))
    if vals:
        v=np.array([x[1] for x in vals])
        emit(f'  {fc.upper()}-M5: n={len(v)} seeds | values={[round(x,4) for x in v]}')
        emit(f'           range(max-min)={v.max()-v.min():.4f}  (min {v.min():.4f}, max {v.max():.4f})')
    else:
        emit(f'  {fc.upper()}-M5: no seed files found')

emit('='*60); emit('RECURSIVE vs FROZEN imputer spread (Maj-4) — across 13 imputers')
for fc in ['lgb','mlp']:
    froz={}; rec={}
    for imp in IMPS:
        mf=med(f'{RES}/{imp}__{fc}_m5lags_hpo_test_per_series.parquet')
        mr=med(f'{RES}/recursive_{imp}__{fc}_m5lags_hpo_test_per_series.parquet')
        if mf is not None: froz[imp]=mf
        if mr is not None: rec[imp]=mr
    common=sorted(set(froz)&set(rec))
    emit(f'  {fc.upper()}-M5: frozen cells={len(froz)}, recursive cells={len(rec)}, common={len(common)}')
    if froz:
        fv=np.array([froz[i] for i in froz]); emit(f'    FROZEN  spread(max-min)={fv.max()-fv.min():.4f}  (min {fv.min():.4f}, max {fv.max():.4f})')
    if rec:
        rv=np.array([rec[i] for i in rec]); emit(f'    RECURS. spread(max-min)={rv.max()-rv.min():.4f}  (min {rv.min():.4f}, max {rv.max():.4f})')
    if common:
        emit(f'    per-imputer (frozen -> recursive):')
        for i in common: emit(f'      {i:16s} {froz[i]:.4f} -> {rec[i]:.4f}')

with open(os.path.join(RES,'robustness_summary.txt'),'w') as f:
    f.write('\n'.join(out_lines)+'\n')
emit('='*60); emit(f'saved {RES}/robustness_summary.txt')
