"""
censored_aware_lgb.py — MAJOR 1 (generalised): a censored-aware DIRECT forecaster.
LGB-M5 trained on the raw CENSORED series (no imputer) with a quantile (pinball) loss,
swept over tau, to trace the accuracy-bias trade-off without any imputation stage.
Usage:  [HPO_VARIANT=1] TAUS="0.5,0.7,0.8,0.9" python censored_aware_lgb.py
Outputs censored_lgb_m5_q<tau>_test_per_series.parquet per tau.
"""
import os, gc, time, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
import lightgbm as lgb
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

TAUS = [float(x) for x in os.getenv('TAUS', '0.5,0.7,0.8,0.9').split(',')]
N_ROUNDS = 10 if os.getenv('SMOKE') == '1' else 500
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
HOURS_RANGE = np.arange(H_START, H_END, dtype=np.int32)
CONT_FEATURES = ['discount','avg_temperature','avg_humidity','precpt','avg_wind_level','holiday_flag','activity_flag']
CAT_FEATURES = ['store_id','product_id','city_id','dow','hour']
LAG_NAMES = ['lag_1d','lag_7d','lag_14d','rmean_7d','rmean_14d','rstd_7d',
             'lag_dow','rmean_dow','daily_total_lag1','daily_total_rmean7','momentum_1d_7d']
# base HP: mirror the M5 LGB cell, but the objective is quantile (set per tau below)
LGB_BASE = {'metric':'quantile','num_leaves':31,'learning_rate':0.1,'feature_fraction':0.8,
            'bagging_fraction':0.3,'bagging_freq':1,'min_child_samples':500,'max_bin':127,
            'verbose':-1,'num_threads':-1,'seed':42}
if os.getenv('HPO_VARIANT') == '1':
    import json
    with open(os.path.join(RESULTS_DIR, 'hpo_lgb_best.json')) as f:
        hpo = json.load(f)['best_params']
    for k in ['num_leaves','learning_rate','min_child_samples','bagging_fraction','feature_fraction']:
        LGB_BASE[k] = hpo[k]
SUFFIX = '_hpo' if os.getenv('HPO_VARIANT') == '1' else ''
print(f'=== Censored-aware LGB-M5 (no imputer, quantile loss) | taus={TAUS} | rounds={N_ROUNDS} ===')

# ---- load raw CENSORED series (no imputation: series = S_obs) ----
print('\n1. Loading (raw censored series)...')
df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_train_hf['dt_parsed'] = pd.to_datetime(df_train_hf['dt']); df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])
df_full = pd.concat([df_train_hf, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
all_dates = sorted(df_full['dt_parsed'].unique()); date_to_day = {d:i+1 for i,d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day); df_full['dow'] = df_full['dt_parsed'].dt.dayofweek
sales_orig = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]     # S_obs (censored)
stock_orig = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)[:, H_START:H_END]
del df_train_hf, df_eval

print('  Building series cache (lags from CENSORED series)...')
series_cache = {}
for (sid,pid), grp in df_full.groupby(['store_id','product_id'], sort=False):
    gs = grp.sort_values('day_num'); idx = gs.index.values
    series_cache[(sid,pid)] = {'days':gs['day_num'].values,'dows':gs['dow'].values,'sales':sales_orig[idx]}
print(f'  {len(series_cache):,} series')

def clags(as_, ad, dw, K):
    z=np.float32; NH=N_HOURS
    L={n:np.full(NH,np.nan,dtype=z) for n in LAG_NAMES}
    if K==0: return L
    L['lag_1d']=as_[-1]
    if K>=7: L['lag_7d']=as_[-7]
    if K>=14: L['lag_14d']=as_[-14]
    if K>=7: L['rmean_7d']=as_[-7:].mean(0)
    if K>=14: L['rmean_14d']=as_[-14:].mean(0)
    if K>=2: L['rstd_7d']=as_[-min(7,K):].std(0)
    sd=ad==dw
    if sd.any(): ds=as_[sd]; L['lag_dow']=ds[-1]; L['rmean_dow']=ds.mean(0)
    dt=as_.sum(1); L['daily_total_lag1']=np.full(NH,dt[-1],dtype=z)
    if K>=7: L['daily_total_rmean7']=np.full(NH,dt[-7:].mean(),dtype=z)
    r,l=L['rmean_7d'],L['lag_1d']
    if not np.isnan(r).all():
        v=(~np.isnan(l))&(~np.isnan(r))&(r>0)
        if v.any(): m=np.full(NH,np.nan,dtype=z); m[v]=l[v]/r[v]; L['momentum_1d_7d']=m
    return L

def build_ds(split):
    if split=='train': d0,d1=2,83
    elif split=='val': d0,d1=84,90
    else: d0,d1=91,97
    mask=(df_full['day_num']>=d0)&(df_full['day_num']<=d1)
    ds=df_full[mask]; idx_s=np.where(mask.values)[0]; nd=len(ds)
    sids=ds['store_id'].values; pids=ds['product_id'].values
    cids=ds['city_id'].values; dows=ds['dow'].values
    conts=ds[CONT_FEATURES].values.astype(np.float32); dnums=ds['day_num'].values
    sd=sales_orig[idx_s]; sk=stock_orig[idx_s]
    nh=nd*N_HOURS; hrs=np.tile(HOURS_RANGE,nd)
    sh=np.repeat(sids,N_HOURS); ph=np.repeat(pids,N_HOURS)
    ch=np.repeat(cids,N_HOURS); dh=np.repeat(dows,N_HOURS); coh=np.repeat(conts,N_HOURS,axis=0)
    y=sd.ravel().astype(np.float32); sf=sk.ravel().astype(np.float32)
    fd={'store_id':sh,'product_id':ph,'city_id':ch,'dow':dh,'hour':hrs}
    for j,c in enumerate(CONT_FEATURES): fd[c]=coh[:,j]
    la={n:np.full(nh,np.nan,dtype=np.float32) for n in LAG_NAMES}
    print(f'    Computing lags for {nd:,} days...')
    for ri in range(nd):
        if (ri+1)%1000000==0: print(f'      ... {ri+1:,}/{nd:,}')
        sid,pid,d,dv=sids[ri],pids[ri],dnums[ri],dows[ri]
        scc=series_cache[(sid,pid)]
        ad=d-1 if split=='train' else (83 if split=='val' else 90)
        am=scc['days']<=ad; K=int(am.sum()); hs=ri*N_HOURS
        if K>0:
            lg=clags(scc['sales'][am],scc['dows'][am],dv,K)
            for n in LAG_NAMES: la[n][hs:hs+N_HOURS]=lg[n]
    for n in LAG_NAMES: fd[n]=la[n]
    del la
    X=pd.DataFrame(fd); del fd; gc.collect()
    for c in CAT_FEATURES: X[c]=X[c].astype('category')
    return X,y,sf,sh,ph

print('\n2. Building datasets (once)...')
t0=time.time()
Xtr,ytr,_,_,_=build_ds('train'); print(f'  Train: {len(Xtr):,}')
ltr=lgb.Dataset(Xtr,ytr,free_raw_data=True); del Xtr,ytr; gc.collect()
Xva,yva,sva,siva,piva=build_ds('val'); print(f'  Val: {len(Xva):,}')
lva=lgb.Dataset(Xva,yva,reference=ltr,free_raw_data=False); gc.collect()
Xte,yte,ste,site,pite=build_ds('test'); print(f'  Test: {len(Xte):,}')
del df_full,sales_orig,stock_orig,series_cache; gc.collect()
print(f'  Built in {time.time()-t0:.0f}s')

def eval_and_save(preds, out_path):
    nd=len(preds)//N_HOURS
    dft=pd.DataFrame({'sid':site,'pid':pite,'day_idx':np.repeat(np.arange(nd),N_HOURS),
                      'pred':preds.astype(np.float64),'obs':yte.astype(np.float64),'stock':ste})
    recs=[]
    for (sid,pid),grp in dft.groupby(['sid','pid'],sort=False):
        ig=grp['stock'].values==0
        sao=np.abs(grp['obs'].values[ig]).sum(); sae=np.abs(grp['pred'].values[ig]-grp['obs'].values[ig]).sum()
        se=(grp['pred'].values[ig]-grp['obs'].values[ig]).sum(); so=grp['obs'].values[ig].sum()
        hw=sae/sao if sao>0 else np.nan; hwp=se/so if so!=0 else np.nan
        sd2,ao2,se2,so2=0.,0.,0.,0.
        for di,dg in grp.groupby('day_idx',sort=False):
            dm=dg['stock'].values==0
            if dm.any():
                pv,ov=dg['pred'].values[dm].sum(),dg['obs'].values[dm].sum(); sd2+=abs(pv-ov);ao2+=abs(ov);se2+=pv-ov;so2+=ov
        recs.append({'store_id':sid,'product_id':pid,'hourly_wape':hw,'hourly_wpe':hwp,
                     'daily_wape':sd2/ao2 if ao2>0 else np.nan,'daily_wpe':se2/so2 if so2!=0 else np.nan})
    ps=pd.DataFrame(recs); ps.to_parquet(out_path,index=False)
    return ps['hourly_wape'].dropna().median(), ps['hourly_wpe'].dropna().median()

def eval_val_save(preds_val, tag):
    """Per-series in-stock VALIDATION hourly WAPE, for selecting tau on the Sec 3.7 criterion."""
    nd=len(preds_val)//N_HOURS
    dft=pd.DataFrame({'sid':siva,'pid':piva,'pred':preds_val.astype(np.float64),
                      'obs':yva.astype(np.float64),'stock':sva})
    recs=[]
    for (sid,pid),grp in dft.groupby(['sid','pid'],sort=False):
        ig=grp['stock'].values==0
        sao=np.abs(grp['obs'].values[ig]).sum(); sae=np.abs(grp['pred'].values[ig]-grp['obs'].values[ig]).sum()
        recs.append({'store_id':sid,'product_id':pid,'val_hourly_wape':sae/sao if sao>0 else np.nan,'n_instock':int(ig.sum())})
    vp=pd.DataFrame(recs); vp.to_parquet(os.path.join(RESULTS_DIR,f'censored_lgb_m5_q{tag}{SUFFIX}_VAL_per_series.parquet'),index=False)
    return vp[vp['n_instock']>=34]['val_hourly_wape'].dropna().median()

print('\n3. Quantile sweep (censored-aware, direct)...')
for tau in TAUS:
    ts=f'{tau:.2f}'.rstrip('0').rstrip('.')
    out_path=os.path.join(RESULTS_DIR, f'censored_lgb_m5_q{ts}{SUFFIX}_test_per_series.parquet')
    val_path=os.path.join(RESULTS_DIR, f'censored_lgb_m5_q{ts}{SUFFIX}_VAL_per_series.parquet')
    if os.path.exists(out_path) and os.path.exists(val_path): print(f'  tau={tau}: SKIP (exists)'); continue
    t1=time.time()
    params=dict(LGB_BASE); params['objective']='quantile'; params['alpha']=tau
    model=lgb.train(params,ltr,num_boost_round=N_ROUNDS,valid_sets=[lva],valid_names=['val'],
                    callbacks=[lgb.early_stopping(30),lgb.log_evaluation(0)])
    preds=np.clip(model.predict(Xte),0,None)
    wm,pm=eval_and_save(preds,out_path)
    vm=eval_val_save(np.clip(model.predict(Xva),0,None),ts)
    print(f'  tau={tau}: best_iter={model.best_iteration}  VAL_WAPE_med={vm:.4f}  TEST_WAPE_med={wm:.4f}  WPE_med={pm:.4f}  ({time.time()-t1:.0f}s)')
    del model,preds; gc.collect()
print('\nDONE censored-aware LGB sweep')
