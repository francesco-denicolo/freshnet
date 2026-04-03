"""
08d_lgb_single.py — Train LGB M5-lags per un singolo imputer
==============================================================
Usage: freshnet/bin/python notebooks_final/08d_lgb_single.py <imputer_key>
"""

import sys, os, gc, time, functools
import numpy as np, pandas as pd

print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)
import lightgbm as lgb

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

SEED = 42; np.random.seed(SEED)
CONT_FEATURES = ['discount', 'avg_temperature', 'avg_humidity',
                  'precpt', 'avg_wind_level', 'holiday_flag', 'activity_flag']
CAT_FEATURES = ['store_id', 'product_id', 'city_id', 'dow', 'hour']
LAG_NAMES = ['lag_1d','lag_7d','lag_14d','rmean_7d','rmean_14d','rstd_7d',
             'lag_dow','rmean_dow','daily_total_lag1','daily_total_rmean7','momentum_1d_7d']
LGB_PARAMS = {'objective':'regression','metric':'mae','num_leaves':31,'learning_rate':0.1,
              'feature_fraction':0.8,'bagging_fraction':0.3,'bagging_freq':1,
              'min_child_samples':500,'max_bin':127,'verbose':-1,'num_threads':-1,'seed':SEED}

IMP_KEY = sys.argv[1]
IMP_LABELS = {'media_cond':'Media condizionata','media_glob':'Media globale',
              'mediana_cond':'Mediana condizionata','lgb':'LGB imputer'}

cell_key = f'{IMP_KEY}__lgb_m5lags'
out_path = os.path.join(RESULTS_DIR, f'{cell_key}_test_per_series.parquet')
if os.path.exists(out_path):
    print(f'SKIP: {out_path} exists'); sys.exit(0)

print(f'=== LGB (M5 lags) × {IMP_LABELS[IMP_KEY]} ({IMP_KEY}) ===')

# Load data
print('\n1. Loading data...')
df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_train_hf['dt_parsed'] = pd.to_datetime(df_train_hf['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])
df_full = pd.concat([df_train_hf, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i+1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek
sales_orig = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)
stock_orig = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)
del df_train_hf, df_eval

# Align completed_sales
df_cs = pd.read_parquet(os.path.join(COMPLETED_DIR, f'{IMP_KEY}.parquet'))
cs_sales = np.array(df_cs['hours_sale'].tolist(), dtype=np.float32)
completed_full = sales_orig.copy()
cs_keys = (df_cs['store_id'].astype(str)+'_'+df_cs['product_id'].astype(str)+'_'+df_cs['dt']).values
full_keys = (df_full['store_id'].astype(str)+'_'+df_full['product_id'].astype(str)+'_'+df_full['dt']).values
km = dict(zip(cs_keys, range(len(df_cs))))
for i in range(len(df_full)):
    k = full_keys[i]
    if k in km: completed_full[i] = cs_sales[km[k]]
del df_cs, cs_sales, cs_keys, full_keys, km; gc.collect()

# Series cache
print('  Building series cache...')
series_cache = {}
for (sid,pid), grp in df_full.groupby(['store_id','product_id'], sort=False):
    gs = grp.sort_values('day_num')
    idx = gs.index.values
    series_cache[(sid,pid)] = {
        'days': gs['day_num'].values, 'dows': gs['dow'].values,
        'sales': completed_full[idx],
    }
print(f'  {len(series_cache):,} series')

# Lag computation
def compute_lags(avail_sales, avail_dows, dow, K):
    z = np.float32
    L = {n: np.full(24,np.nan,dtype=z) for n in LAG_NAMES}
    if K==0: return L
    L['lag_1d']=avail_sales[-1]
    if K>=7: L['lag_7d']=avail_sales[-7]
    if K>=14: L['lag_14d']=avail_sales[-14]
    if K>=7: L['rmean_7d']=avail_sales[-7:].mean(0)
    if K>=14: L['rmean_14d']=avail_sales[-14:].mean(0)
    if K>=2: L['rstd_7d']=avail_sales[-min(7,K):].std(0)
    sd=avail_dows==dow
    if sd.any():
        ds=avail_sales[sd]; L['lag_dow']=ds[-1]; L['rmean_dow']=ds.mean(0)
    dt=avail_sales.sum(1)
    L['daily_total_lag1']=np.full(24,dt[-1],dtype=z)
    if K>=7: L['daily_total_rmean7']=np.full(24,dt[-7:].mean(),dtype=z)
    r,l=L['rmean_7d'],L['lag_1d']
    if not np.isnan(r).all():
        v=(~np.isnan(l))&(~np.isnan(r))&(r>0)
        if v.any():
            m=np.full(24,np.nan,dtype=z); m[v]=l[v]/r[v]; L['momentum_1d_7d']=m
    return L

def build_lgb_ds(split):
    if split=='train': d_min,d_max=2,83
    elif split=='val': d_min,d_max=84,90
    else: d_min,d_max=91,97
    mask=(df_full['day_num']>=d_min)&(df_full['day_num']<=d_max)
    ds=df_full[mask]; idx=np.where(mask.values)[0]; nd=len(ds)
    sids=ds['store_id'].values; pids=ds['product_id'].values
    cids=ds['city_id'].values; dows=ds['dow'].values
    conts=ds[CONT_FEATURES].values.astype(np.float32)
    dnums=ds['day_num'].values
    sd=sales_orig[idx]; sk=stock_orig[idx]
    nh=nd*24
    hrs=np.tile(np.arange(24,dtype=np.int32),nd)
    sh=np.repeat(sids,24); ph=np.repeat(pids,24)
    ch=np.repeat(cids,24); dh=np.repeat(dows,24)
    coh=np.repeat(conts,24,axis=0)
    y=sd.ravel().astype(np.float32); sf=sk.ravel().astype(np.float32)
    fd={'store_id':sh,'product_id':ph,'city_id':ch,'dow':dh,'hour':hrs}
    for j,c in enumerate(CONT_FEATURES): fd[c]=coh[:,j]
    la={n:np.full(nh,np.nan,dtype=np.float32) for n in LAG_NAMES}
    for ri in range(nd):
        if (ri+1)%500000==0: print(f'      ... {ri+1:,}/{nd:,}')
        sid,pid,d,dv=sids[ri],pids[ri],dnums[ri],dows[ri]
        sc=series_cache[(sid,pid)]
        ad=d-1 if split=='train' else (83 if split=='val' else 90)
        am=sc['days']<=ad; K=int(am.sum())
        hs=ri*24
        if K>0:
            lg=compute_lags(sc['sales'][am],sc['dows'][am],dv,K)
            for n in LAG_NAMES: la[n][hs:hs+24]=lg[n]
    for n in LAG_NAMES: fd[n]=la[n]
    del la
    X=pd.DataFrame(fd); del fd; gc.collect()
    for c in CAT_FEATURES: X[c]=X[c].astype('category')
    return X,y,sf,sh,ph

# Build and train
print('\n2. Building train...')
t0=time.time()
Xtr,ytr,_,_,_=build_lgb_ds('train')
print(f'  Train: {len(Xtr):,} rows')
print('  Building val...')
Xva,yva,sva,siva,piva=build_lgb_ds('val')
print(f'  Val: {len(Xva):,} rows')

print('  Training...')
ltr=lgb.Dataset(Xtr,ytr,free_raw_data=True)
lva=lgb.Dataset(Xva,yva,reference=ltr,free_raw_data=True)
model=lgb.train(LGB_PARAMS,ltr,num_boost_round=500,valid_sets=[lva],valid_names=['val'],
                callbacks=[lgb.early_stopping(30),lgb.log_evaluation(100)])
print(f'  Best iter: {model.best_iteration}')
del Xtr,ytr,ltr,lva,Xva; gc.collect()

print('  Building test...')
Xte,yte,ste,site,pite=build_lgb_ds('test')
preds=np.clip(model.predict(Xte),0,None)

# Eval
ins=ste==0
ph,oh=preds[ins],yte[ins]
sae_h,sao_h=np.abs(ph-oh).sum(),np.abs(oh).sum()
se_h,so_h=(ph-oh).sum(),oh.sum()
ndt=len(preds)//24
pr,obr,skr=preds.reshape(ndt,24),yte.reshape(ndt,24),ste.reshape(ndt,24)
sidr,pidr=site.reshape(ndt,24)[:,0],pite.reshape(ndt,24)[:,0]
sad,sod,sed,ssd=0.,0.,0.,0.
for d in range(ndt):
    m=skr[d]==0
    if m.any():
        pv,ov=pr[d,m].sum(),obr[d,m].sum()
        sad+=abs(pv-ov);sod+=abs(ov);sed+=pv-ov;ssd+=ov
pooled={'hourly_wape':sae_h/sao_h,'hourly_wpe':se_h/so_h,
        'daily_wape':sad/sod,'daily_wpe':sed/ssd}

# Per-series
dft=pd.DataFrame({'sid':site,'pid':pite,'day_idx':np.repeat(np.arange(ndt),24),
                   'pred':preds.astype(np.float64),'obs':yte.astype(np.float64),'stock':ste})
recs=[]
for (sid,pid),grp in dft.groupby(['sid','pid'],sort=False):
    ins_g=grp['stock'].values==0
    sao_s=np.abs(grp['obs'].values[ins_g]).sum()
    sae_s=np.abs(grp['pred'].values[ins_g]-grp['obs'].values[ins_g]).sum()
    se_s=(grp['pred'].values[ins_g]-grp['obs'].values[ins_g]).sum()
    so_s=grp['obs'].values[ins_g].sum()
    hw=sae_s/sao_s if sao_s>0 else np.nan
    hwp=se_s/so_s if so_s!=0 else np.nan
    sd2,ao2,se2,so2,nv=0.,0.,0.,0.,0
    for di,dg in grp.groupby('day_idx',sort=False):
        dm=dg['stock'].values==0
        if dm.any():
            pv,ov=dg['pred'].values[dm].sum(),dg['obs'].values[dm].sum()
            sd2+=abs(pv-ov);ao2+=abs(ov);se2+=pv-ov;so2+=ov;nv+=1
    recs.append({'store_id':sid,'product_id':pid,'hourly_wape':hw,'hourly_wpe':hwp,
                 'daily_wape':sd2/ao2 if ao2>0 else np.nan,'daily_wpe':se2/so2 if so2!=0 else np.nan,
                 'n_hours_instock':int(ins_g.sum()),'n_days_valid':nv})

ps=pd.DataFrame(recs)
ps.to_parquet(out_path,index=False)
med={c:ps[c].dropna().median() for c in ['hourly_wape','hourly_wpe']}

print(f'\n  WAPE_h pool={pooled["hourly_wape"]:.4f}, med={med["hourly_wape"]:.4f}')
print(f'  WPE_h pool={pooled["hourly_wpe"]:.4f}, time={time.time()-t0:.0f}s')
print(f'  Salvato: {out_path}')
print('DONE')
