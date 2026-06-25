"""
Maj-8 robustness: does the headline (best cell, winning family) survive the
non-independence of series within cities (one city = 51.6% of series)?
(a) Per-series Friedman within City 0 alone vs the other 17 cities.
(b) City-blocked Friedman: aggregate to per-city mean WAPE per cell, rank the
    113 cells within each of the 18 cities, Friedman across the 18 city blocks
    (effective N = 18 -> much wider Nemenyi CD).
"""
import os, glob, functools, numpy as np, pandas as pd
from scipy import stats
print = functools.partial(print, flush=True)
RES = os.path.join(os.path.dirname(__file__), 'results')
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
NON_HPO_FC = {'chronos_bolt','timesfm','global_mean','dow_mean','ma_k21','ma_k56'}

def parse(n):
    return n.split('__',1) if '__' in n else (None,n)

# --- replicate script 45 cell loading ---
per={}; seen=set()
for f in sorted(glob.glob(f'{RES}/*_hpo_test_per_series.parquet')):
    n=os.path.basename(f).replace('_hpo_test_per_series.parquet',''); per[n]=pd.read_parquet(f); seen.add(n)
for f in sorted(glob.glob(f'{RES}/*_test_per_series.parquet')):
    fn=os.path.basename(f)
    if '_hpo_test_per_series' in fn: continue
    n=fn.replace('_test_per_series.parquet','')
    if n in seen: continue
    if n.startswith('naive_'):
        base=n.replace('naive_','',1)
        if base in NON_HPO_FC:
            nn=f'no_imp__{base}'
            if nn not in seen: per[nn]=pd.read_parquet(f); seen.add(nn)
        continue
    imp,fc=parse(n)
    if fc not in NON_HPO_FC: continue
    per[n]=pd.read_parquet(f); seen.add(n)
cells=sorted(per.keys()); print('cells:',len(cells))

# --- wide matrix on common series ---
idx=None
for n in cells:
    i=per[n].set_index(['store_id','product_id']).index
    idx=i if idx is None else idx.intersection(i)
print('common series:',len(idx))
W=pd.DataFrame(index=idx,columns=cells,dtype=float)
for n in cells:
    W[n]=per[n].set_index(['store_id','product_id'])['hourly_wape'].reindex(idx)
W=W.dropna()
print('series after dropna:',len(W))

# --- store->city map ---
tr=pd.read_parquet(os.path.join(DATA,'frn50k_train.parquet'),columns=['store_id','city_id']).drop_duplicates('store_id')
s2c=dict(zip(tr.store_id,tr.city_id))
city=np.array([s2c[s] for s,_ in W.index])
print('cities:',len(set(city)),'| City0 share: %.3f'%(np.mean(city==0)))

def friedman_best(Wsub,label):
    R=Wsub.rank(axis=1,method='average'); mr=R.mean(0).sort_values()
    k=Wsub.shape[1]; N=len(Wsub)
    chi2,_=stats.friedmanchisquare(*[Wsub[c].values for c in Wsub.columns])
    Wk=chi2/(N*(k-1))
    q=3.354  # q_alpha(k=113,inf)/... approx studentized range /sqrt2 ~ use Demsar table; reuse script-45 CD if available
    cd=q*np.sqrt(k*(k+1)/(6*N))
    best=mr.index[0]; nequiv=int((mr-mr.iloc[0]<=cd).sum())
    print(f'  [{label}] N={N:6d}  best={best:26s}  W={Wk:.3f}  CD={cd:.3f}  #equiv={nequiv}  top3={list(mr.index[:3])}')
    return mr

# (a) City 0 vs rest
print('\n(a) Per-series Friedman, City 0 vs rest:')
friedman_best(W[city==0],'City 0')
friedman_best(W[city!=0],'other 17 cities')

# (b) City-blocked: per-city mean WAPE per cell -> 18 x 113, Friedman across cities
print('\n(b) City-blocked Friedman (N=18 cities):')
Wc=W.copy(); Wc['city']=city
city_mean=Wc.groupby('city')[cells].mean()   # 18 x 113
R=city_mean.rank(axis=1,method='average'); mr=R.mean(0).sort_values()
k=len(cells); N=city_mean.shape[0]
chi2,p=stats.friedmanchisquare(*[city_mean[c].values for c in cells])
Wk=chi2/(N*(k-1))
q=3.354
cd=q*np.sqrt(k*(k+1)/(6*N))
print(f'  N(cities)={N}  best={mr.index[0]}  W={Wk:.3f}  CD={cd:.2f}  #within-CD={int((mr-mr.iloc[0]<=cd).sum())}')
print('  top-8 by city-block mean rank:')
for c in mr.index[:8]: print(f'    {c:28s} {mr[c]:.2f}')
print('  forecaster of top-12:', [mr.index[i].split('__')[1] for i in range(12)])
