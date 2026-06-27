"""R2.2: MASE for every cell, at the in-stock hourly granularity (the same errors
that WAPE aggregates), so the head ranking can be re-checked under a scale-free
metric robust to small per-series denominators. Uses the identity
  MASE_s = WAPE_s * mean_instock_test_obs_s / scale_s
where scale_s is the in-sample seasonal-naive (one-day-back, same-hour) MAE.
The per-series factor is cell-independent, so MASE for all 155 cells is derived
from the stored per-series hourly WAPE -- no retraining."""
import os, glob, functools
import numpy as np, pandas as pd
from scipy.stats import spearmanr
print = functools.partial(print, flush=True)
PR=os.path.join(os.path.dirname(__file__),'..'); DATA=os.path.join(PR,'data')
RES=os.path.join(os.path.dirname(__file__),'results'); H0,H1=6,23; NH=H1-H0

def load(fn):
    d=pd.read_parquet(os.path.join(DATA,fn)); d['dt_parsed']=pd.to_datetime(d['dt'])
    return d
tr=pd.concat([load('frn50k_train.parquet'),load('frn50k_eval.parquet')],ignore_index=True)
tr=tr.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
dates=sorted(tr['dt_parsed'].unique()); d2n={d:i+1 for i,d in enumerate(dates)}
tr['day_num']=tr['dt_parsed'].map(d2n)
S=tr[['store_id','product_id']].drop_duplicates().shape[0]; ND=len(tr)//S
sid=tr['store_id'].values.reshape(S,ND)[:,0]; pid=tr['product_id'].values.reshape(S,ND)[:,0]
days=tr['day_num'].values.reshape(S,ND)[0]
sales=np.array(tr['hours_sale'].tolist(),np.float64)[:,H0:H1].reshape(S,ND,NH)
stock=np.array(tr['hours_stock_status'].tolist(),np.int8)[:,H0:H1].reshape(S,ND,NH)
ins=stock==0
trm=(days>=2)&(days<=90); tem=(days>=91)&(days<=97)
# in-sample seasonal (1-day) naive MAE per series, over operational hours
S1=sales[:,1:,:]-sales[:,:-1,:]                            # day-over-day same-hour diff
sd=days[1:]; tmask=(sd<=90)&(sd>=2)
scale=np.abs(S1[:,tmask,:]).reshape(S,-1).mean(1)          # (S,)
# mean in-stock test obs per series
te_obs=np.where(ins[:,tem,:],sales[:,tem,:],np.nan)
mean_obs=np.nanmean(te_obs.reshape(S,-1),axis=1)
factor=np.where(scale>0, mean_obs/scale, np.nan)
fac=pd.DataFrame({'store_id':sid,'product_id':pid,'factor':factor})
print(f'factor: median={np.nanmedian(factor):.3f}  (scale med={np.nanmedian(scale):.4f}, mean_obs med={np.nanmedian(mean_obs):.4f})')

mat=pd.read_parquet(f'{RES}/hpo_matrix_pareto.parquet')
rows=[]
for _,r in mat.iterrows():
    cell=r['cell']; suf='_hpo_test_per_series.parquet' if r['hpo'] else '_test_per_series.parquet'
    f=f'{RES}/{cell}{suf}'
    if not os.path.exists(f):
        f2=f'{RES}/{cell}_test_per_series.parquet'
        if os.path.exists(f2): f=f2
        else: continue
    ps=pd.read_parquet(f)[['store_id','product_id','hourly_wape']].merge(fac,on=['store_id','product_id'])
    ps['mase']=ps.hourly_wape*ps.factor
    rows.append({'cell':cell,'imputer':r['imputer'],'forecaster':r['forecaster'],
                 'wape_med':r['wape_h_med'],'mase_med':float(ps['mase'].median())})
d=pd.DataFrame(rows)
d.to_parquet(f'{RES}/mase_full_matrix.parquet',index=False)
print(f'\n{len(d)} cells. Spearman(MASE_med, WAPE_med) over cells = {spearmanr(d.mase_med,d.wape_med).correlation:.3f}')
print('\n=== Top 10 by MASE_med ===')
print(d.sort_values('mase_med').head(10)[['cell','mase_med','wape_med']].to_string(index=False))
# family-level best
fam={'naive':['global_mean','dow_mean','ma_k56'],'intermittent':['croston','sba','tsb'],
     'lag-ML':['lgb_m5lags','mlp_m5lags'],'TFT':['tft'],'foundation':['chronos_bolt','timesfm']}
print('\n=== best MASE_med per family ===')
for fm,fcs in fam.items():
    g=d[d.forecaster.isin(fcs)]
    if len(g): b=g.loc[g.mase_med.idxmin()]; print(f'  {fm:13s} {b.cell:30s} MASE={b.mase_med:.3f}  WAPE={b.wape_med:.3f}')
