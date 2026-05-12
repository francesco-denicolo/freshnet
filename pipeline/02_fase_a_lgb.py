"""
04_baseline_lgb.py — Fase A2: LightGBM Baseline (ore 6-22)
============================================================
Serie ristrette a ore 6-22 (17 ore/giorno).
2 varianti: no lags, M5 lags.

Eseguire con: freshnet/bin/python notebooks_622/04_baseline_lgb.py
"""
import sys, os, gc, time, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)
import lightgbm as lgb

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42; np.random.seed(SEED)
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
HOURS_RANGE = np.arange(H_START, H_END, dtype=np.int32)  # [6,7,...,22]

CONT_FEATURES = ['discount','avg_temperature','avg_humidity',
                  'precpt','avg_wind_level','holiday_flag','activity_flag']
CAT_FEATURES = ['store_id','product_id','city_id','dow','hour']
LAG_NAMES = ['lag_1d','lag_7d','lag_14d','rmean_7d','rmean_14d','rstd_7d',
             'lag_dow','rmean_dow','daily_total_lag1','daily_total_rmean7','momentum_1d_7d']
LGB_PARAMS = {'objective':'regression_l1','metric':'mae','num_leaves':31,'learning_rate':0.1,
              'feature_fraction':0.8,'bagging_fraction':0.3,'bagging_freq':1,
              'min_child_samples':500,'max_bin':127,'verbose':-1,'num_threads':-1,'seed':SEED}

print('=' * 72)
print('  FASE A2 — LIGHTGBM (ore 6-22)')
print('=' * 72)

# Load data
print('\n1. Caricamento dati...')
df_train = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_train['dt_parsed'] = pd.to_datetime(df_train['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])
df_full = pd.concat([df_train, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i+1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek
del df_train, df_eval

# SLICE TO 6-22
sales_all = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
stock_all = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)[:, H_START:H_END]
print(f'  Shape: {sales_all.shape} ({N_HOURS}h/day)')

# Series cache
print('  Building series cache...')
series_cache = {}
for (sid,pid), grp in df_full.groupby(['store_id','product_id'], sort=False):
    gs = grp.sort_values('day_num')
    idx = gs.index.values
    series_cache[(sid,pid)] = {'days':gs['day_num'].values,'dows':gs['dow'].values,
                                'sales':sales_all[idx]}
print(f'  {len(series_cache):,} serie')

# Lag computation
def compute_lags(avail_sales, avail_dows, dow, K):
    z = np.float32; NH = N_HOURS
    L = {n: np.full(NH, np.nan, dtype=z) for n in LAG_NAMES}
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
    L['daily_total_lag1']=np.full(NH,dt[-1],dtype=z)
    if K>=7: L['daily_total_rmean7']=np.full(NH,dt[-7:].mean(),dtype=z)
    r,l=L['rmean_7d'],L['lag_1d']
    if not np.isnan(r).all():
        v=(~np.isnan(l))&(~np.isnan(r))&(r>0)
        if v.any():
            m=np.full(NH,np.nan,dtype=z); m[v]=l[v]/r[v]; L['momentum_1d_7d']=m
    return L

def build_lgb_dataset(split, use_lags):
    if split=='train': d_min,d_max=2,83
    elif split=='val': d_min,d_max=84,90
    else: d_min,d_max=91,97
    mask=(df_full['day_num']>=d_min)&(df_full['day_num']<=d_max)
    ds=df_full[mask]; idx_s=np.where(mask.values)[0]; nd=len(ds)

    sids=ds['store_id'].values; pids=ds['product_id'].values
    cids=ds['city_id'].values; dows=ds['dow'].values
    conts=ds[CONT_FEATURES].values.astype(np.float32); dnums=ds['day_num'].values

    sd=sales_all[idx_s]; sk=stock_all[idx_s]
    nh=nd*N_HOURS
    hrs=np.tile(HOURS_RANGE, nd)  # [6,7,...,22, 6,7,...,22, ...]
    sh=np.repeat(sids,N_HOURS); ph=np.repeat(pids,N_HOURS)
    ch=np.repeat(cids,N_HOURS); dh=np.repeat(dows,N_HOURS)
    coh=np.repeat(conts,N_HOURS,axis=0)
    y=sd.ravel().astype(np.float32); sf=sk.ravel().astype(np.float32)
    fd={'store_id':sh,'product_id':ph,'city_id':ch,'dow':dh,'hour':hrs}
    for j,c in enumerate(CONT_FEATURES): fd[c]=coh[:,j]

    if use_lags:
        la={n:np.full(nh,np.nan,dtype=np.float32) for n in LAG_NAMES}
        print(f'      Computing lags for {nd:,} days...')
        for ri in range(nd):
            if (ri+1)%500000==0: print(f'        ... {ri+1:,}/{nd:,}')
            sid,pid,d,dv=sids[ri],pids[ri],dnums[ri],dows[ri]
            sc=series_cache[(sid,pid)]
            ad=d-1 if split=='train' else (83 if split=='val' else 90)
            am=sc['days']<=ad; K=int(am.sum()); hs=ri*N_HOURS
            if K>0:
                lg=compute_lags(sc['sales'][am],sc['dows'][am],dv,K)
                for n in LAG_NAMES: la[n][hs:hs+N_HOURS]=lg[n]
        for n in LAG_NAMES: fd[n]=la[n]
        del la

    X=pd.DataFrame(fd); del fd; gc.collect()
    for c in CAT_FEATURES: X[c]=X[c].astype('category')
    return X,y,sf,sh,ph

# Eval helper
def eval_lgb(preds, y, stk, sids, pids, label):
    instock=stk==0
    ph,oh=preds[instock],y[instock]
    sae_h,sao_h=np.abs(ph-oh).sum(),np.abs(oh).sum()
    se_h,so_h=(ph-oh).sum(),oh.sum()
    nd=len(preds)//N_HOURS
    pr=preds.reshape(nd,N_HOURS); ob=y.reshape(nd,N_HOURS); sk=stk.reshape(nd,N_HOURS)
    sid_d=sids.reshape(nd,N_HOURS)[:,0]; pid_d=pids.reshape(nd,N_HOURS)[:,0]
    sae_d,sao_d,se_d,so_d=0.,0.,0.,0.
    for d in range(nd):
        m=sk[d]==0
        if m.any():
            pv,ov=pr[d,m].sum(),ob[d,m].sum()
            sae_d+=abs(pv-ov);sao_d+=abs(ov);se_d+=pv-ov;so_d+=ov
    pooled={'hourly_wape':sae_h/sao_h,'hourly_wpe':se_h/so_h,
            'daily_wape':sae_d/sao_d if sao_d>0 else np.nan,'daily_wpe':se_d/so_d if so_d!=0 else np.nan}
    # Per-series
    dft=pd.DataFrame({'sid':sids,'pid':pids,'day_idx':np.repeat(np.arange(nd),N_HOURS),
                       'pred':preds.astype(np.float64),'obs':y.astype(np.float64),'stock':stk})
    recs=[]
    for (sid,pid),grp in dft.groupby(['sid','pid'],sort=False):
        ins=grp['stock'].values==0
        sao_s=np.abs(grp['obs'].values[ins]).sum()
        sae_s=np.abs(grp['pred'].values[ins]-grp['obs'].values[ins]).sum()
        se_s=(grp['pred'].values[ins]-grp['obs'].values[ins]).sum()
        so_s=grp['obs'].values[ins].sum()
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
                     'n_hours_instock':int(ins.sum()),'n_days_valid':nv})
    ps=pd.DataFrame(recs)
    ps.to_parquet(os.path.join(RESULTS_DIR,f'{label}_test_per_series.parquet'),index=False)
    med={c:ps[c].dropna().median() for c in ['hourly_wape','hourly_wpe']}
    print(f'    {label}: WAPE_h pool={pooled["hourly_wape"]:.4f}, med={med["hourly_wape"]:.4f}, '
          f'WPE_h={pooled["hourly_wpe"]:.4f}')
    return pooled, med, ps

# ===========================================================================
# Train and evaluate both variants
# ===========================================================================
all_results = {}

for use_lags, label in [(False, 'lgb_nolags'), (True, 'lgb_m5lags')]:
    vl = 'M5 lags' if use_lags else 'no lags'
    print(f'\n  === LGB ({vl}) ===')
    t0 = time.time()

    print('    Building train...')
    Xtr, ytr, _, _, _ = build_lgb_dataset('train', use_lags)
    print(f'    Train: {len(Xtr):,} rows, {Xtr.shape[1]} feat')

    print('    Building val...')
    Xva, yva, sva, _, _ = build_lgb_dataset('val', use_lags)
    print(f'    Val: {len(Xva):,} rows')

    print('    Training...')
    ltr = lgb.Dataset(Xtr, ytr, free_raw_data=True)
    lva = lgb.Dataset(Xva, yva, reference=ltr, free_raw_data=True)
    model = lgb.train(LGB_PARAMS, ltr, num_boost_round=500,
                      valid_sets=[lva], valid_names=['val'],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)])
    print(f'    Best iter: {model.best_iteration}')
    model.save_model(os.path.join(RESULTS_DIR, f'{label}.txt'))

    del Xtr, ytr, ltr, lva, Xva; gc.collect()

    print('    Building test...')
    Xte, yte, ste, site, pite = build_lgb_dataset('test', use_lags)
    preds = np.clip(model.predict(Xte), 0, None)
    pooled, med, ps = eval_lgb(preds, yte, ste, site, pite, label)
    all_results[label] = {'pooled': pooled, 'median': med, 'elapsed': time.time()-t0}
    print(f'    Time: {time.time()-t0:.0f}s')

    del Xte, preds, model; gc.collect()

print('\n' + '=' * 72)
print('  DONE — 04_baseline_lgb.py (ore 6-22)')
print('=' * 72)
