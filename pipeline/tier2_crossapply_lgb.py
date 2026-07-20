"""Tier-2 mechanism control (cross-application), LGB-M5 version — symmetric to
tier2_crossapply.py (MLP). Train ONE LGB-M5 on a single imputer (itransformer)
and, without retraining, run it on the test-time lag features built from EACH of
the 13 imputers' completed series. Per-series in-stock WAPE (median) tells us
whether the imputer-irrelevance is intrinsic model insensitivity (model_A applied
to features_B matches the native cell B for all B) rather than per-cell
re-calibration. Referee: replicate the MLP cross-application for LGB.

Output: pipeline/results/tier2_crossapply_lgb.parquet  and a printed summary.
"""
import os, sys, gc, time, json, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
sys.path.insert(0, os.path.dirname(__file__))
import lightgbm as lgb

t0 = time.time()
PR = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(PR, 'data'); COMP = os.path.join(DATA, 'completed_sales_622')
RES = os.path.join(os.path.dirname(__file__), 'results')
SEED = 42; np.random.seed(SEED)
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
HOURS_RANGE = np.arange(H_START, H_END, dtype=np.int32)
CONT_FEATURES = ['discount','avg_temperature','avg_humidity','precpt','avg_wind_level','holiday_flag','activity_flag']
CAT_FEATURES = ['store_id','product_id','city_id','dow','hour']
LAG_NAMES = ['lag_1d','lag_7d','lag_14d','rmean_7d','rmean_14d','rstd_7d',
             'lag_dow','rmean_dow','daily_total_lag1','daily_total_rmean7','momentum_1d_7d']
LGB_PARAMS = {'objective':'regression_l1','metric':'mae','num_leaves':31,'learning_rate':0.1,
              'feature_fraction':0.8,'bagging_fraction':0.3,'bagging_freq':1,
              'min_child_samples':500,'max_bin':127,'verbose':-1,'num_threads':-1,'seed':SEED}
with open(f'{RES}/hpo_lgb_best.json') as f:
    _hpo = json.load(f)['best_params']
for k in ['num_leaves','learning_rate','min_child_samples','bagging_fraction','feature_fraction']:
    LGB_PARAMS[k] = _hpo[k]
print(f'[HPO] LGB_PARAMS: {_hpo}')

TRAIN_IMP = 'itransformer'
TEST_IMPS = ['itransformer','mediana_glob','media_glob','saits','dlinear','forward_fill','timesnet',
             'lgb','imputeformer','seasonal_naive','linear_interp','media_cond','mediana_cond']

print('1. Loading base data...')
dt = pd.read_parquet(f'{DATA}/frn50k_train.parquet'); de = pd.read_parquet(f'{DATA}/frn50k_eval.parquet')
for d in (dt, de): d['dt_parsed'] = pd.to_datetime(d['dt'])
df = pd.concat([dt, de], ignore_index=True).sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
del dt, de
if os.getenv('SMOKE') == '1':
    sp = df[['store_id','product_id']].drop_duplicates().head(400)
    df = df.merge(sp, on=['store_id','product_id']).reset_index(drop=True)
    print(f'  SMOKE: {len(sp)} series')
alldates = sorted(df['dt_parsed'].unique())
df['day_num'] = df['dt_parsed'].map({d:i+1 for i,d in enumerate(alldates)})
df['dow'] = df['dt_parsed'].dt.dayofweek
sales_orig = np.array(df['hours_sale'].tolist(), np.float32)[:, H_START:H_END]
stock_orig = np.array(df['hours_stock_status'].tolist(), np.float32)[:, H_START:H_END]
full_keys = (df['store_id'].astype(str)+'_'+df['product_id'].astype(str)+'_'+df['dt']).values
print(f'  {df[["store_id","product_id"]].drop_duplicates().shape[0]:,} series, t={time.time()-t0:.0f}s')

def series_cache_for(imp):
    """Completed-series lag cache for one imputer."""
    cs = pd.read_parquet(f'{COMP}/{imp}.parquet')
    arr = np.array(cs['hours_sale'].tolist(), np.float32)
    km = dict(zip((cs['store_id'].astype(str)+'_'+cs['product_id'].astype(str)+'_'+cs['dt']).values, range(len(cs))))
    completed = sales_orig.copy()
    for i in range(len(df)):
        k = full_keys[i]
        if k in km: completed[i] = arr[km[k]]
    sc = {}
    for (sid,pid), grp in df.groupby(['store_id','product_id'], sort=False):
        gs = grp.sort_values('day_num'); idx = gs.index.values
        sc[(sid,pid)] = {'days':gs['day_num'].values,'dows':gs['dow'].values,'sales':completed[idx]}
    del completed; gc.collect()
    return sc

def clags(as_, ad, dw, K):
    z = np.float32; NH = N_HOURS
    L = {n:np.full(NH, np.nan, dtype=z) for n in LAG_NAMES}
    if K == 0: return L
    L['lag_1d'] = as_[-1]
    if K >= 7: L['lag_7d'] = as_[-7]
    if K >= 14: L['lag_14d'] = as_[-14]
    if K >= 7: L['rmean_7d'] = as_[-7:].mean(0)
    if K >= 14: L['rmean_14d'] = as_[-14:].mean(0)
    if K >= 2: L['rstd_7d'] = as_[-min(7,K):].std(0)
    sd = ad == dw
    if sd.any(): ds = as_[sd]; L['lag_dow'] = ds[-1]; L['rmean_dow'] = ds.mean(0)
    dtl = as_.sum(1); L['daily_total_lag1'] = np.full(NH, dtl[-1], dtype=z)
    if K >= 7: L['daily_total_rmean7'] = np.full(NH, dtl[-7:].mean(), dtype=z)
    r, l = L['rmean_7d'], L['lag_1d']
    if not np.isnan(r).all():
        v = (~np.isnan(l)) & (~np.isnan(r)) & (r > 0)
        if v.any(): m = np.full(NH, np.nan, dtype=z); m[v] = l[v]/r[v]; L['momentum_1d_7d'] = m
    return L

def build_ds(split, sc):
    if split == 'train': d0, d1 = 2, 83
    elif split == 'val': d0, d1 = 84, 90
    else: d0, d1 = 91, 97
    mask = (df['day_num'] >= d0) & (df['day_num'] <= d1)
    ds = df[mask]; idx_s = np.where(mask.values)[0]; nd = len(ds)
    sids = ds['store_id'].values; pids = ds['product_id'].values
    cids = ds['city_id'].values; dows = ds['dow'].values
    conts = ds[CONT_FEATURES].values.astype(np.float32); dnums = ds['day_num'].values
    sd = sales_orig[idx_s]; sk = stock_orig[idx_s]
    nh = nd*N_HOURS; hrs = np.tile(HOURS_RANGE, nd)
    fd = {'store_id':np.repeat(sids,N_HOURS),'product_id':np.repeat(pids,N_HOURS),
          'city_id':np.repeat(cids,N_HOURS),'dow':np.repeat(dows,N_HOURS),'hour':hrs}
    coh = np.repeat(conts, N_HOURS, axis=0)
    for j,c in enumerate(CONT_FEATURES): fd[c] = coh[:,j]
    y = sd.ravel().astype(np.float32); sf = sk.ravel().astype(np.float32)
    la = {n:np.full(nh, np.nan, dtype=np.float32) for n in LAG_NAMES}
    for ri in range(nd):
        sid,pid,d,dv = sids[ri],pids[ri],dnums[ri],dows[ri]
        s = sc[(sid,pid)]
        ad = d-1 if split=='train' else (83 if split=='val' else 90)
        am = s['days'] <= ad; K = int(am.sum()); hs = ri*N_HOURS
        if K > 0:
            lg = clags(s['sales'][am], s['dows'][am], dv, K)
            for n in LAG_NAMES: la[n][hs:hs+N_HOURS] = lg[n]
    for n in LAG_NAMES: fd[n] = la[n]
    X = pd.DataFrame(fd)
    for c in CAT_FEATURES: X[c] = X[c].astype('category')
    return X, y, sf, np.repeat(sids,N_HOURS), np.repeat(pids,N_HOURS)

def perseries_wape(pred, y, stock, sid, pid):
    ins = stock == 0
    d = pd.DataFrame({'sid':sid[ins],'pid':pid[ins],'ae':np.abs(pred[ins]-y[ins]),'ao':np.abs(y[ins])})
    g = d.groupby(['sid','pid']).sum()
    w = (g['ae']/g['ao']).replace([np.inf,-np.inf], np.nan).dropna()
    return w  # per-series in-stock WAPE

# Train on itransformer
print(f'\n2. Train LGB-M5 on {TRAIN_IMP}...')
sc_tr = series_cache_for(TRAIN_IMP)
Xtr,ytr,_,_,_ = build_ds('train', sc_tr); print(f'  train {len(Xtr):,}  t={time.time()-t0:.0f}s')
Xva,yva,_,_,_ = build_ds('val', sc_tr)
ltr = lgb.Dataset(Xtr, ytr, free_raw_data=True); lva = lgb.Dataset(Xva, yva, reference=ltr, free_raw_data=True)
model = lgb.train(LGB_PARAMS, ltr, num_boost_round=500, valid_sets=[lva], valid_names=['val'],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
print(f'  best_iter={model.best_iteration}  t={time.time()-t0:.0f}s')
del Xtr,ytr,Xva,yva,ltr,lva; gc.collect()

# Apply to each imputer's test features
print('\n3. Cross-apply to each imputer test features...')
rows = []
for B in TEST_IMPS:
    scB = series_cache_for(B) if B != TRAIN_IMP else sc_tr
    Xte,yte,ste,sid,pid = build_ds('test', scB)
    pred = np.clip(model.predict(Xte), 0, None).astype(np.float32)
    w = perseries_wape(pred, yte, ste, sid, pid)
    cross_med = float(w.median())
    # native cell median (from existing per-series parquet)
    nf = f'{RES}/{B}__lgb_m5lags_hpo_test_per_series.parquet'
    native_med = float(pd.read_parquet(nf)['hourly_wape'].median()) if os.path.exists(nf) else np.nan
    rows.append({'test_imputer':B,'crossapply_wape_med':cross_med,'native_wape_med':native_med,
                 'abs_diff':abs(cross_med-native_med) if not np.isnan(native_med) else np.nan})
    print(f'  {B:16s} cross={cross_med:.4f}  native={native_med:.4f}  |Δ|={rows[-1]["abs_diff"]:.4f}')
    del scB, Xte, yte; gc.collect()

out = pd.DataFrame(rows)
out.to_parquet(f'{RES}/tier2_crossapply_lgb.parquet', index=False)
md = out['abs_diff'].mean()
print(f'\n=== SUMMARY (LGB cross-application, train={TRAIN_IMP}) ===')
print(out.to_string(index=False))
print(f'\nMean |cross - native| median WAPE = {md:.4f}  (MLP reference: 0.009)')
print(f'DONE  t={time.time()-t0:.0f}s')
