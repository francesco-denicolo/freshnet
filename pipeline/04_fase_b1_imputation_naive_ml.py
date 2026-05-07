"""
05_imputation.py — Fase B1: Imputation (ore 6-22)
===================================================
Serie ristrette a ore 6-22 (17 ore/giorno).

4 imputer:
  1. Media condizionata: media per (store, product, dow, hour) su ore in-stock
  2. Media globale: media per (store, product, hour) su ore in-stock
  3. Mediana condizionata: mediana per (store, product, dow, hour) su ore in-stock
  4. LGB imputer: LightGBM trainato su ore in-stock

Procedura per ciascuno:
  1. Train su gg 1-83 (ore in-stock, ore 6-22)
  2. Valutazione su maschere MNAR (gg 84-90, filtrate ore 6-22)
  3. Retrain su gg 1-90
  4. Completed_sales per gg 1-90 → data/completed_sales_622/<nome>.parquet

Eseguire con: freshnet/bin/python notebooks_622/05_imputation.py
"""
import sys, os, gc, time, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)
import lightgbm as lgb

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(COMPLETED_DIR, exist_ok=True)

SEED = 42; np.random.seed(SEED)
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
HOURS_RANGE = np.arange(H_START, H_END, dtype=np.int32)

CONT_FEATURES = ['discount','avg_temperature','avg_humidity',
                  'precpt','avg_wind_level','holiday_flag','activity_flag']
CAT_FEATURES_IMP = ['store_id','product_id','city_id','dow','hour']

LGB_PARAMS = {'objective':'regression','metric':'mae','num_leaves':31,'learning_rate':0.1,
              'feature_fraction':0.8,'bagging_fraction':0.3,'bagging_freq':1,
              'min_child_samples':500,'max_bin':127,'verbose':-1,'num_threads':-1,'seed':SEED}

print('=' * 72)
print('  FASE B1 — IMPUTATION (ore 6-22)')
print('=' * 72)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print('\n1. Caricamento dati...')
df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_train_hf['dt_parsed'] = pd.to_datetime(df_train_hf['dt'])
all_dates = sorted(df_train_hf['dt_parsed'].unique())
date_to_day = {d: i+1 for i, d in enumerate(all_dates)}
df_train_hf['day_num'] = df_train_hf['dt_parsed'].map(date_to_day)
df_train_hf['dow'] = df_train_hf['dt_parsed'].dt.dayofweek

# SLICE TO 6-22
sales_all = np.array(df_train_hf['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
stock_all = np.array(df_train_hf['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]
print(f'  Train HF: {len(df_train_hf):,} righe, {len(all_dates)} giorni')
print(f'  Shape: sales={sales_all.shape}, stock={stock_all.shape}')

# Expand to hourly flat (only hours 6-22)
n_rows = len(df_train_hf)
n_hourly = n_rows * N_HOURS

store_flat = np.repeat(df_train_hf['store_id'].values, N_HOURS)
product_flat = np.repeat(df_train_hf['product_id'].values, N_HOURS)
day_flat = np.repeat(df_train_hf['day_num'].values, N_HOURS)
dow_flat = np.repeat(df_train_hf['dow'].values, N_HOURS)
city_flat = np.repeat(df_train_hf['city_id'].values, N_HOURS)
dt_flat = np.repeat(df_train_hf['dt'].values, N_HOURS)
conts_flat = np.repeat(df_train_hf[CONT_FEATURES].values.astype(np.float32), N_HOURS, axis=0)
hours_flat = np.tile(HOURS_RANGE, n_rows)  # [6,7,...,22, 6,7,...,22, ...]

sale_flat = sales_all.ravel().astype(np.float32)
stock_flat = stock_all.ravel().astype(np.int8)

print(f'  Hourly flat: {n_hourly:,} ore')
print(f'  In-stock: {(stock_flat==0).sum():,} ({100*(stock_flat==0).mean():.1f}%)')
print(f'  Stockout: {(stock_flat==1).sum():,} ({100*(stock_flat==1).mean():.1f}%)')

# Load MNAR masks — filter to hours 6-22
print('  Loading MNAR masks...')
masks_val = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_val.parquet'))
masks_val = masks_val[(masks_val['hour'] >= H_START) & (masks_val['hour'] < H_END)].reset_index(drop=True)
masks_val_dow = pd.to_datetime(masks_val['dt']).dt.dayofweek.values
print(f'  MNAR masks (6-22): {len(masks_val):,} ore mascherate')

# Pre-compute stockout indices
stockout_idx = np.where(stock_flat == 1)[0]
print(f'  Stockout indices: {len(stockout_idx):,}')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def eval_mnar_batch(preds, label):
    gt = masks_val['ground_truth'].values.astype(np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    sao = np.abs(gt).sum(); so = gt.sum()
    sae = np.abs(preds-gt).sum(); se = (preds-gt).sum()
    wape = sae/sao if sao>0 else np.nan
    wpe = se/so if so!=0 else np.nan
    print(f'  {label}: WAPE_recovery={wape:.4f}, WPE_recovery={wpe:.4f}')
    return {'wape_recovery': wape, 'wpe_recovery': wpe}

def save_completed(imputed_flat, label):
    completed = sale_flat.copy()
    so_mask = stock_flat == 1
    completed[so_mask] = np.clip(imputed_flat[so_mask], 0, None)
    completed_17 = completed.reshape(n_rows, N_HOURS)

    df_out = df_train_hf[['store_id','product_id','dt','day_num','dow']].copy()
    df_out['hours_sale'] = list(completed_17)
    df_out['hours_stock_status'] = list(stock_all)

    out_path = os.path.join(COMPLETED_DIR, f'{label}.parquet')
    df_out.to_parquet(out_path, index=False)

    imp_vals = completed[so_mask]
    print(f'  Salvato: {out_path}')
    print(f'  Media imputata (stockout): {imp_vals.mean():.4f}, Media S_obs (in-stock): {sale_flat[~so_mask].mean():.4f}')

# Training masks
train_mask_83 = (day_flat <= 83) & (stock_flat == 0)
train_mask_90 = (day_flat <= 90) & (stock_flat == 0)

# ===========================================================================
# IMPUTER 1: Media condizionata
# ===========================================================================
print('\n' + '=' * 72)
print('  IMPUTER 1 — MEDIA CONDIZIONATA')
print('=' * 72)

t0 = time.time()
df_cond = pd.DataFrame({'store_id':store_flat[train_mask_83],'product_id':product_flat[train_mask_83],
                         'dow':dow_flat[train_mask_83],'hour':hours_flat[train_mask_83],
                         'sale':sale_flat[train_mask_83]})
cond_mean_83 = df_cond.groupby(['store_id','product_id','dow','hour'])['sale'].mean()
glob_fb_83 = df_cond.groupby(['store_id','product_id','hour'])['sale'].mean()
del df_cond
print(f'  Train 83: {len(cond_mean_83):,} chiavi, {time.time()-t0:.1f}s')

# MNAR eval
preds_m = np.zeros(len(masks_val), dtype=np.float32)
for i in range(len(masks_val)):
    r = masks_val.iloc[i]
    k = (r['store_id'], r['product_id'], masks_val_dow[i], r['hour'])
    if k in cond_mean_83: preds_m[i] = cond_mean_83[k]
    elif (r['store_id'], r['product_id'], r['hour']) in glob_fb_83:
        preds_m[i] = glob_fb_83[(r['store_id'], r['product_id'], r['hour'])]
ta_1 = eval_mnar_batch(preds_m, 'Media cond (83)')

# Retrain 90 + completed_sales
df_c90 = pd.DataFrame({'store_id':store_flat[train_mask_90],'product_id':product_flat[train_mask_90],
                        'dow':dow_flat[train_mask_90],'hour':hours_flat[train_mask_90],
                        'sale':sale_flat[train_mask_90]})
cond_mean_90 = df_c90.groupby(['store_id','product_id','dow','hour'])['sale'].mean()
glob_fb_90 = df_c90.groupby(['store_id','product_id','hour'])['sale'].mean()
del df_c90

imp = np.zeros(n_hourly, dtype=np.float32)
for idx in stockout_idx:
    k = (store_flat[idx], product_flat[idx], dow_flat[idx], hours_flat[idx])
    if k in cond_mean_90: imp[idx] = cond_mean_90[k]
    elif (store_flat[idx], product_flat[idx], hours_flat[idx]) in glob_fb_90:
        imp[idx] = glob_fb_90[(store_flat[idx], product_flat[idx], hours_flat[idx])]
save_completed(imp, 'media_cond')

# ===========================================================================
# IMPUTER 2: Media globale
# ===========================================================================
print('\n' + '=' * 72)
print('  IMPUTER 2 — MEDIA GLOBALE')
print('=' * 72)

preds_m2 = np.zeros(len(masks_val), dtype=np.float32)
for i in range(len(masks_val)):
    r = masks_val.iloc[i]
    k = (r['store_id'], r['product_id'], r['hour'])
    if k in glob_fb_83: preds_m2[i] = glob_fb_83[k]
ta_2 = eval_mnar_batch(preds_m2, 'Media glob (83)')

imp2 = np.zeros(n_hourly, dtype=np.float32)
for idx in stockout_idx:
    k = (store_flat[idx], product_flat[idx], hours_flat[idx])
    if k in glob_fb_90: imp2[idx] = glob_fb_90[k]
save_completed(imp2, 'media_glob')

# ===========================================================================
# IMPUTER 3: Mediana condizionata
# ===========================================================================
print('\n' + '=' * 72)
print('  IMPUTER 3 — MEDIANA CONDIZIONATA')
print('=' * 72)

t0 = time.time()
df_med = pd.DataFrame({'store_id':store_flat[train_mask_83],'product_id':product_flat[train_mask_83],
                        'dow':dow_flat[train_mask_83],'hour':hours_flat[train_mask_83],
                        'sale':sale_flat[train_mask_83]})
cond_med_83 = df_med.groupby(['store_id','product_id','dow','hour'])['sale'].median()
glob_med_fb_83 = df_med.groupby(['store_id','product_id','hour'])['sale'].median()
del df_med
print(f'  Train 83: {len(cond_med_83):,} chiavi, {time.time()-t0:.1f}s')

preds_m3 = np.zeros(len(masks_val), dtype=np.float32)
for i in range(len(masks_val)):
    r = masks_val.iloc[i]
    k = (r['store_id'], r['product_id'], masks_val_dow[i], r['hour'])
    if k in cond_med_83: preds_m3[i] = cond_med_83[k]
    elif (r['store_id'], r['product_id'], r['hour']) in glob_med_fb_83:
        preds_m3[i] = glob_med_fb_83[(r['store_id'], r['product_id'], r['hour'])]
ta_3 = eval_mnar_batch(preds_m3, 'Mediana cond (83)')

df_m90 = pd.DataFrame({'store_id':store_flat[train_mask_90],'product_id':product_flat[train_mask_90],
                        'dow':dow_flat[train_mask_90],'hour':hours_flat[train_mask_90],
                        'sale':sale_flat[train_mask_90]})
cond_med_90 = df_m90.groupby(['store_id','product_id','dow','hour'])['sale'].median()
glob_med_fb_90 = df_m90.groupby(['store_id','product_id','hour'])['sale'].median()
del df_m90

imp3 = np.zeros(n_hourly, dtype=np.float32)
for idx in stockout_idx:
    k = (store_flat[idx], product_flat[idx], dow_flat[idx], hours_flat[idx])
    if k in cond_med_90: imp3[idx] = cond_med_90[k]
    elif (store_flat[idx], product_flat[idx], hours_flat[idx]) in glob_med_fb_90:
        imp3[idx] = glob_med_fb_90[(store_flat[idx], product_flat[idx], hours_flat[idx])]
save_completed(imp3, 'mediana_cond')

# ===========================================================================
# IMPUTER 4: LGB
# ===========================================================================
print('\n' + '=' * 72)
print('  IMPUTER 4 — LGB IMPUTER')
print('=' * 72)

def build_lgb_imp_data(day_min, day_max, instock_only=True):
    day_mask = (day_flat >= day_min) & (day_flat <= day_max)
    mask = day_mask & (stock_flat == 0) if instock_only else day_mask
    fd = {'store_id':store_flat[mask],'product_id':product_flat[mask],
          'city_id':city_flat[mask],'dow':dow_flat[mask],'hour':hours_flat[mask]}
    for j, c in enumerate(CONT_FEATURES): fd[c] = conts_flat[mask, j]
    X = pd.DataFrame(fd)
    for c in CAT_FEATURES_IMP: X[c] = X[c].astype('category')
    return X, sale_flat[mask].astype(np.float32)

print('  Training LGB su gg 1-83...')
t0 = time.time()
Xtr, ytr = build_lgb_imp_data(1, 83)
Xva, yva = build_lgb_imp_data(84, 90)
print(f'  Train: {len(Xtr):,}, Val: {len(Xva):,}')

ltr = lgb.Dataset(Xtr, ytr, free_raw_data=True)
lva = lgb.Dataset(Xva, yva, reference=ltr, free_raw_data=True)
model83 = lgb.train(LGB_PARAMS, ltr, num_boost_round=500,
                    valid_sets=[lva], valid_names=['val'],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)])
print(f'  Best iter: {model83.best_iteration}, MAE: {model83.best_score["val"]["l1"]:.6f}, '
      f'time: {time.time()-t0:.0f}s')
del Xtr, ytr, Xva, yva, ltr, lva; gc.collect()

# MNAR eval
print('  Valutazione MNAR...')
city_map = df_train_hf.groupby(['store_id','product_id'])['city_id'].first()
cont_map = df_train_hf.set_index(['store_id','product_id','dt'])[CONT_FEATURES]

mm = masks_val[['store_id','product_id','dt','hour']].copy()
mm['dow'] = masks_val_dow
mm['city_id'] = mm.set_index(['store_id','product_id']).index.map(lambda x: city_map.get(x, 0))
mi = mm.set_index(['store_id','product_id','dt'])
cv = mi.join(cont_map, how='left')[CONT_FEATURES].values.astype(np.float32)
mm[CONT_FEATURES] = cv

X_mnar = mm[CAT_FEATURES_IMP + CONT_FEATURES].copy()
for c in CAT_FEATURES_IMP: X_mnar[c] = X_mnar[c].astype('category')
preds_lgb = np.clip(model83.predict(X_mnar), 0, None)
ta_4 = eval_mnar_batch(preds_lgb, 'LGB (83)')
del X_mnar, mm, mi; gc.collect()

# Retrain 90
print('  Retrain su gg 1-90...')
t0 = time.time()
Xtr90, ytr90 = build_lgb_imp_data(1, 90)
print(f'  Train: {len(Xtr90):,}')
ltr90 = lgb.Dataset(Xtr90, ytr90, free_raw_data=True)
model90 = lgb.train(LGB_PARAMS, ltr90, num_boost_round=model83.best_iteration)
print(f'  Time: {time.time()-t0:.0f}s')
del Xtr90, ytr90, ltr90, model83; gc.collect()

# Imputazione
print('  Imputazione stockout...')
so_mask = stock_flat == 1
fd_so = {'store_id':store_flat[so_mask],'product_id':product_flat[so_mask],
          'city_id':city_flat[so_mask],'dow':dow_flat[so_mask],'hour':hours_flat[so_mask]}
for j, c in enumerate(CONT_FEATURES): fd_so[c] = conts_flat[so_mask, j]
X_so = pd.DataFrame(fd_so)
for c in CAT_FEATURES_IMP: X_so[c] = X_so[c].astype('category')
preds_so = np.clip(model90.predict(X_so), 0, None).astype(np.float32)

imp4 = np.zeros(n_hourly, dtype=np.float32)
imp4[so_mask] = preds_so
save_completed(imp4, 'lgb')
del X_so, preds_so, model90; gc.collect()

# ===========================================================================
# Summary: Traccia A
# ===========================================================================
print('\n' + '=' * 72)
print('  TRACCIA A — Ranking imputer (ore 6-22)')
print('=' * 72)

traccia = {'Media condizionata': ta_1, 'Media globale': ta_2,
           'Mediana condizionata': ta_3, 'LGB imputer': ta_4}

print(f'\n  {"Imputer":<24} {"WAPE_recovery":>14} {"WPE_recovery":>13}')
print('  ' + '-' * 54)
for l, r in traccia.items():
    print(f'  {l:<24} {r["wape_recovery"]:>14.4f} {r["wpe_recovery"]:>13.4f}')

ta_df = pd.DataFrame([{'imputer':k, **v} for k, v in traccia.items()])
ta_df.to_parquet(os.path.join(RESULTS_DIR, 'traccia_a.parquet'), index=False)

print('\n  Completed_sales salvati in:', COMPLETED_DIR)
for f in sorted(os.listdir(COMPLETED_DIR)):
    print(f'    {f}: {os.path.getsize(os.path.join(COMPLETED_DIR, f))/1e6:.1f} MB')

print('\n' + '=' * 72)
print('  DONE — 05_imputation.py (ore 6-22)')
print('=' * 72)
