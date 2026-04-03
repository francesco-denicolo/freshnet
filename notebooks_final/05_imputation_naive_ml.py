"""
05_imputation_naive_ml.py — Fase B1: Imputation (naive + ML)
=============================================================
Impatto della Qualità dell'Imputation sul Demand Forecasting
di Prodotti Deperibili — CLAUDE_FINAL.md

4 imputer:
  1. Media condizionata: media per (store, product, dow, hour) su ore in-stock
  2. Media globale: media per (store, product, hour) su ore in-stock
  3. Mediana condizionata: mediana per (store, product, dow, hour) su ore in-stock
  4. LGB imputer: LightGBM trainato su ore in-stock

Per ciascuno:
  Passo 1. Train su gg 1-83 (ore in-stock)
  Passo 2. Valutazione su maschere MNAR (gg 84-90, seed=42) → Traccia A
  Passo 3. Retrain su gg 1-90
  Passo 4. Applicazione a TUTTE le ore stockout dei gg 1-90
  Passo 5. Salvataggio completed_sales in data/completed_sales/<nome>.parquet

Eseguire con: freshnet/bin/python notebooks_final/05_imputation_naive_ml.py
"""

import sys
import os
import gc
import time
import functools
import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import lightgbm as lgb

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(COMPLETED_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

CONT_FEATURES = ['discount', 'avg_temperature', 'avg_humidity',
                  'precpt', 'avg_wind_level', 'holiday_flag', 'activity_flag']

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.3,
    'bagging_freq': 1,
    'min_child_samples': 500,
    'max_bin': 127,
    'verbose': -1,
    'num_threads': -1,
    'seed': SEED,
}
LGB_MAX_ROUNDS = 500
LGB_EARLY_STOP = 30

# ===========================================================================
print('=' * 72)
print('  FASE B1 — IMPUTATION (naive + ML)')
print('=' * 72)

# ---------------------------------------------------------------------------
# 1. Caricamento dati
# ---------------------------------------------------------------------------
print('\n1. Caricamento dati...')
df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_train_hf['dt_parsed'] = pd.to_datetime(df_train_hf['dt'])

all_dates = sorted(df_train_hf['dt_parsed'].unique())
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}
df_train_hf['day_num'] = df_train_hf['dt_parsed'].map(date_to_day)
df_train_hf['dow'] = df_train_hf['dt_parsed'].dt.dayofweek

# Parse hourly arrays
print('  Parsing hourly arrays...')
sales_all = np.array(df_train_hf['hours_sale'].tolist(), dtype=np.float32)
stock_all = np.array(df_train_hf['hours_stock_status'].tolist(), dtype=np.int8)

n_series = df_train_hf.groupby(['store_id', 'product_id']).ngroups
print(f'  Train HF: {len(df_train_hf):,} righe, {len(all_dates)} giorni, {n_series:,} serie')

# Load MNAR masks
print('  Caricamento maschere MNAR...')
masks_val = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_val.parquet'))
print(f'  Maschere MNAR val: {len(masks_val):,} ore mascherate')


# ---------------------------------------------------------------------------
# 2. Expand to hourly flat format for stockout identification
# ---------------------------------------------------------------------------
print('\n2. Costruzione formato orario...')

# We need: for each (store, product, day, hour) → sale, stock_status, day_num, dow
# Build this efficiently
store_ids = df_train_hf['store_id'].values
product_ids = df_train_hf['product_id'].values
day_nums = df_train_hf['day_num'].values
dows = df_train_hf['dow'].values
city_ids = df_train_hf['city_id'].values
dts = df_train_hf['dt'].values
conts = df_train_hf[CONT_FEATURES].values.astype(np.float32)

n_rows = len(df_train_hf)
n_hourly = n_rows * 24

# Expand to hourly
hours_flat = np.tile(np.arange(24, dtype=np.int32), n_rows)
store_flat = np.repeat(store_ids, 24)
product_flat = np.repeat(product_ids, 24)
day_flat = np.repeat(day_nums, 24)
dow_flat = np.repeat(dows, 24)
city_flat = np.repeat(city_ids, 24)
dt_flat = np.repeat(dts, 24)
conts_flat = np.repeat(conts, 24, axis=0)

sale_flat = sales_all.ravel().astype(np.float32)
stock_flat = stock_all.ravel().astype(np.int8)

print(f'  Formato orario: {n_hourly:,} ore totali')
print(f'  In-stock: {(stock_flat == 0).sum():,} ({100*(stock_flat == 0).mean():.1f}%)')
print(f'  Stockout: {(stock_flat == 1).sum():,} ({100*(stock_flat == 1).mean():.1f}%)')


# ---------------------------------------------------------------------------
# 3. Helper: evaluate imputer on MNAR masks
# ---------------------------------------------------------------------------
def evaluate_on_mnar(imputer_fn, masks_df, label):
    """Evaluate an imputer on MNAR masked hours.

    Args:
        imputer_fn: function(store_id, product_id, dt, hour, dow) → imputed value
            Can also be a vectorized version taking arrays.
        masks_df: DataFrame with store_id, product_id, dt, hour, ground_truth
        label: name for printing

    Returns:
        dict with wape_recovery, wpe_recovery
    """
    gt = masks_df['ground_truth'].values.astype(np.float64)
    preds = np.zeros(len(masks_df), dtype=np.float64)

    for idx, row in masks_df.iterrows():
        preds[idx - masks_df.index[0]] = imputer_fn(
            row['store_id'], row['product_id'], row['dt'], row['hour'])

    sao = np.abs(gt).sum()
    so = gt.sum()
    sae = np.abs(preds - gt).sum()
    se = (preds - gt).sum()

    wape = sae / sao if sao > 0 else np.nan
    wpe = se / so if so != 0 else np.nan

    return {'wape_recovery': wape, 'wpe_recovery': wpe, 'n_masked': len(masks_df)}


def evaluate_on_mnar_batch(preds, masks_df, label):
    """Evaluate imputer with pre-computed predictions."""
    gt = masks_df['ground_truth'].values.astype(np.float64)
    preds = np.asarray(preds, dtype=np.float64)

    sao = np.abs(gt).sum()
    so = gt.sum()
    sae = np.abs(preds - gt).sum()
    se = (preds - gt).sum()

    wape = sae / sao if sao > 0 else np.nan
    wpe = se / so if so != 0 else np.nan

    print(f'  {label}: WAPE_recovery={wape:.4f}, WPE_recovery={wpe:.4f}')
    return {'wape_recovery': wape, 'wpe_recovery': wpe, 'n_masked': len(masks_df)}


# ---------------------------------------------------------------------------
# 4. Helper: build completed_sales
# ---------------------------------------------------------------------------
def build_completed_sales(imputed_stockout_flat, label):
    """Build completed_sales parquet from imputed stockout values.

    Args:
        imputed_stockout_flat: array of imputed values for ALL hours (n_hourly,).
            Only stockout hours will be used; in-stock hours keep S_obs.

    Returns:
        DataFrame with same structure as df_train_hf but with completed hours_sale.
    """
    completed_flat = sale_flat.copy()
    stockout_mask = stock_flat == 1
    completed_flat[stockout_mask] = imputed_stockout_flat[stockout_mask]

    # Clip to non-negative
    completed_flat = np.clip(completed_flat, 0, None)

    # Reshape back to (n_rows, 24)
    completed_24 = completed_flat.reshape(n_rows, 24)

    # Build output DataFrame
    df_out = df_train_hf[['store_id', 'product_id', 'dt', 'day_num', 'dow']].copy()
    df_out['hours_sale'] = list(completed_24)
    df_out['hours_stock_status'] = list(stock_all)

    # Save
    out_path = os.path.join(COMPLETED_DIR, f'{label}.parquet')
    df_out.to_parquet(out_path, index=False)
    print(f'  Salvato: {out_path}')

    # Diagnostics
    n_so = stockout_mask.sum()
    imp_vals = completed_flat[stockout_mask]
    print(f'  Ore stockout imputate: {n_so:,}')
    print(f'  Media imputata (stockout): {imp_vals.mean():.4f}')
    print(f'  Media S_obs (in-stock):    {sale_flat[~stockout_mask].mean():.4f}')

    return df_out


# ===========================================================================
# IMPUTER 1: Media condizionata — mean per (store, product, dow, hour)
# ===========================================================================
print('\n' + '=' * 72)
print('  IMPUTER 1 — MEDIA CONDIZIONATA (store, product, dow, hour)')
print('=' * 72)

# --- Passo 1: Train su gg 1-83 ---
print('\n  Passo 1: Calcolo medie su gg 1-83 (ore in-stock)...')
t0 = time.time()

train_mask_83 = (day_flat <= 83) & (stock_flat == 0)
df_cond = pd.DataFrame({
    'store_id': store_flat[train_mask_83],
    'product_id': product_flat[train_mask_83],
    'dow': dow_flat[train_mask_83],
    'hour': hours_flat[train_mask_83],
    'sale': sale_flat[train_mask_83],
})
cond_mean_83 = df_cond.groupby(['store_id', 'product_id', 'dow', 'hour'])['sale'].mean()
# Fallback: global mean per (store, product, hour)
global_fb_83 = df_cond.groupby(['store_id', 'product_id', 'hour'])['sale'].mean()
del df_cond
print(f'  {len(cond_mean_83):,} chiavi (store, product, dow, hour), {time.time()-t0:.1f}s')

# --- Passo 2: Valutazione su maschere MNAR ---
print('\n  Passo 2: Valutazione su maschere MNAR (gg 84-90)...')
masks_val_dt = pd.to_datetime(masks_val['dt'])
masks_val_dow = masks_val_dt.dt.dayofweek.values

preds_mnar = np.zeros(len(masks_val), dtype=np.float32)
for i in range(len(masks_val)):
    sid = masks_val.iloc[i]['store_id']
    pid = masks_val.iloc[i]['product_id']
    dow = masks_val_dow[i]
    hour = masks_val.iloc[i]['hour']
    key = (sid, pid, dow, hour)
    if key in cond_mean_83:
        preds_mnar[i] = cond_mean_83[key]
    elif (sid, pid, hour) in global_fb_83:
        preds_mnar[i] = global_fb_83[(sid, pid, hour)]
    else:
        preds_mnar[i] = 0.0

traccia_a_1 = evaluate_on_mnar_batch(preds_mnar, masks_val, 'Media condizionata (train 83)')

# --- Passo 3: Retrain su gg 1-90 ---
print('\n  Passo 3: Ricalcolo medie su gg 1-90...')
train_mask_90 = (day_flat <= 90) & (stock_flat == 0)
df_cond90 = pd.DataFrame({
    'store_id': store_flat[train_mask_90],
    'product_id': product_flat[train_mask_90],
    'dow': dow_flat[train_mask_90],
    'hour': hours_flat[train_mask_90],
    'sale': sale_flat[train_mask_90],
})
cond_mean_90 = df_cond90.groupby(['store_id', 'product_id', 'dow', 'hour'])['sale'].mean()
global_fb_90 = df_cond90.groupby(['store_id', 'product_id', 'hour'])['sale'].mean()
del df_cond90

# --- Passo 4: Imputazione tutte le ore stockout gg 1-90 ---
print('\n  Passo 4: Imputazione ore stockout (gg 1-90)...')
imputed_flat = np.zeros(n_hourly, dtype=np.float32)
stockout_idx = np.where(stock_flat == 1)[0]

for idx in stockout_idx:
    sid = store_flat[idx]
    pid = product_flat[idx]
    dow = dow_flat[idx]
    hour = hours_flat[idx]
    key = (sid, pid, dow, hour)
    if key in cond_mean_90:
        imputed_flat[idx] = cond_mean_90[key]
    elif (sid, pid, hour) in global_fb_90:
        imputed_flat[idx] = global_fb_90[(sid, pid, hour)]

build_completed_sales(imputed_flat, 'media_cond')


# ===========================================================================
# IMPUTER 2: Media globale — mean per (store, product, hour)
# ===========================================================================
print('\n' + '=' * 72)
print('  IMPUTER 2 — MEDIA GLOBALE (store, product, hour)')
print('=' * 72)

# --- Passo 1: Train su gg 1-83 ---
print('\n  Passo 1: Calcolo medie su gg 1-83...')
# global_fb_83 already computed above

# --- Passo 2: Valutazione MNAR ---
print('\n  Passo 2: Valutazione su maschere MNAR...')
preds_mnar2 = np.zeros(len(masks_val), dtype=np.float32)
for i in range(len(masks_val)):
    sid = masks_val.iloc[i]['store_id']
    pid = masks_val.iloc[i]['product_id']
    hour = masks_val.iloc[i]['hour']
    key = (sid, pid, hour)
    if key in global_fb_83:
        preds_mnar2[i] = global_fb_83[key]

traccia_a_2 = evaluate_on_mnar_batch(preds_mnar2, masks_val, 'Media globale (train 83)')

# --- Passo 3+4: Retrain su gg 1-90 + imputazione ---
print('\n  Passo 3+4: Imputazione con medie da gg 1-90...')
# global_fb_90 already computed above
imputed_flat2 = np.zeros(n_hourly, dtype=np.float32)
for idx in stockout_idx:
    sid = store_flat[idx]
    pid = product_flat[idx]
    hour = hours_flat[idx]
    key = (sid, pid, hour)
    if key in global_fb_90:
        imputed_flat2[idx] = global_fb_90[key]

build_completed_sales(imputed_flat2, 'media_glob')


# ===========================================================================
# IMPUTER 3: Mediana condizionata — median per (store, product, dow, hour)
# ===========================================================================
print('\n' + '=' * 72)
print('  IMPUTER 3 — MEDIANA CONDIZIONATA (store, product, dow, hour)')
print('=' * 72)

# --- Passo 1: Train su gg 1-83 ---
print('\n  Passo 1: Calcolo mediane su gg 1-83...')
t0 = time.time()
df_med = pd.DataFrame({
    'store_id': store_flat[train_mask_83],
    'product_id': product_flat[train_mask_83],
    'dow': dow_flat[train_mask_83],
    'hour': hours_flat[train_mask_83],
    'sale': sale_flat[train_mask_83],
})
cond_median_83 = df_med.groupby(['store_id', 'product_id', 'dow', 'hour'])['sale'].median()
global_med_fb_83 = df_med.groupby(['store_id', 'product_id', 'hour'])['sale'].median()
del df_med
print(f'  {len(cond_median_83):,} chiavi, {time.time()-t0:.1f}s')

# --- Passo 2: Valutazione MNAR ---
print('\n  Passo 2: Valutazione su maschere MNAR...')
preds_mnar3 = np.zeros(len(masks_val), dtype=np.float32)
for i in range(len(masks_val)):
    sid = masks_val.iloc[i]['store_id']
    pid = masks_val.iloc[i]['product_id']
    dow = masks_val_dow[i]
    hour = masks_val.iloc[i]['hour']
    key = (sid, pid, dow, hour)
    if key in cond_median_83:
        preds_mnar3[i] = cond_median_83[key]
    elif (sid, pid, hour) in global_med_fb_83:
        preds_mnar3[i] = global_med_fb_83[(sid, pid, hour)]

traccia_a_3 = evaluate_on_mnar_batch(preds_mnar3, masks_val, 'Mediana condizionata (train 83)')

# --- Passo 3+4: Retrain su gg 1-90 + imputazione ---
print('\n  Passo 3+4: Ricalcolo e imputazione da gg 1-90...')
df_med90 = pd.DataFrame({
    'store_id': store_flat[train_mask_90],
    'product_id': product_flat[train_mask_90],
    'dow': dow_flat[train_mask_90],
    'hour': hours_flat[train_mask_90],
    'sale': sale_flat[train_mask_90],
})
cond_median_90 = df_med90.groupby(['store_id', 'product_id', 'dow', 'hour'])['sale'].median()
global_med_fb_90 = df_med90.groupby(['store_id', 'product_id', 'hour'])['sale'].median()
del df_med90

imputed_flat3 = np.zeros(n_hourly, dtype=np.float32)
for idx in stockout_idx:
    sid = store_flat[idx]
    pid = product_flat[idx]
    dow = dow_flat[idx]
    hour = hours_flat[idx]
    key = (sid, pid, dow, hour)
    if key in cond_median_90:
        imputed_flat3[idx] = cond_median_90[key]
    elif (sid, pid, hour) in global_med_fb_90:
        imputed_flat3[idx] = global_med_fb_90[(sid, pid, hour)]

build_completed_sales(imputed_flat3, 'mediana_cond')


# ===========================================================================
# IMPUTER 4: LGB imputer — LightGBM trained on in-stock hours
# ===========================================================================
print('\n' + '=' * 72)
print('  IMPUTER 4 — LGB IMPUTER')
print('=' * 72)

CAT_FEATURES_IMP = ['store_id', 'product_id', 'city_id', 'dow', 'hour']

def build_lgb_imputer_data(day_min, day_max, instock_only=True):
    """Build flat dataset for LGB imputer."""
    day_mask = (day_flat >= day_min) & (day_flat <= day_max)
    if instock_only:
        mask = day_mask & (stock_flat == 0)
    else:
        mask = day_mask

    feat_dict = {
        'store_id': store_flat[mask],
        'product_id': product_flat[mask],
        'city_id': city_flat[mask],
        'dow': dow_flat[mask],
        'hour': hours_flat[mask],
    }
    for j, c in enumerate(CONT_FEATURES):
        feat_dict[c] = conts_flat[mask, j]

    X = pd.DataFrame(feat_dict)
    for c in CAT_FEATURES_IMP:
        X[c] = X[c].astype('category')

    y = sale_flat[mask].astype(np.float32)
    return X, y

# --- Passo 1: Train su gg 1-83 (in-stock only) ---
print('\n  Passo 1: Training LGB su gg 1-83 (in-stock)...')
t0 = time.time()

X_train_imp, y_train_imp = build_lgb_imputer_data(1, 83, instock_only=True)
print(f'  Train: {len(X_train_imp):,} ore in-stock')

# Use days 84-90 in-stock as validation for early stopping
X_val_imp, y_val_imp = build_lgb_imputer_data(84, 90, instock_only=True)
print(f'  Val (in-stock): {len(X_val_imp):,} ore')

lgb_train_ds = lgb.Dataset(X_train_imp, y_train_imp, free_raw_data=True)
lgb_val_ds = lgb.Dataset(X_val_imp, y_val_imp, reference=lgb_train_ds, free_raw_data=True)

model_imp_83 = lgb.train(
    LGB_PARAMS, lgb_train_ds,
    num_boost_round=LGB_MAX_ROUNDS,
    valid_sets=[lgb_val_ds], valid_names=['val'],
    callbacks=[lgb.early_stopping(LGB_EARLY_STOP), lgb.log_evaluation(100)],
)
print(f'  Best iter: {model_imp_83.best_iteration}, '
      f'MAE: {model_imp_83.best_score["val"]["l1"]:.6f}, '
      f'time: {time.time()-t0:.0f}s')

del X_train_imp, y_train_imp, X_val_imp, y_val_imp, lgb_train_ds, lgb_val_ds
gc.collect()

# --- Passo 2: Valutazione su maschere MNAR ---
print('\n  Passo 2: Valutazione su maschere MNAR...')
# Build features for MNAR masked hours
masks_feat = pd.DataFrame({
    'store_id': masks_val['store_id'],
    'product_id': masks_val['product_id'],
    'hour': masks_val['hour'],
})
# Need city_id, dow, cont features for each masked hour
# Merge with df_train_hf to get these
masks_merge = masks_val[['store_id', 'product_id', 'dt', 'hour']].copy()
masks_merge['dow'] = masks_val_dow

# Get city_id per series
city_map = df_train_hf.groupby(['store_id', 'product_id'])['city_id'].first()
masks_merge['city_id'] = masks_merge.set_index(['store_id', 'product_id']).index.map(
    lambda x: city_map.get(x, 0))

# Get cont features from the day
cont_map = df_train_hf.set_index(['store_id', 'product_id', 'dt'])[CONT_FEATURES]
masks_merge_idx = masks_merge.set_index(['store_id', 'product_id', 'dt'])
cont_vals = masks_merge_idx.join(cont_map, how='left')[CONT_FEATURES].values.astype(np.float32)
masks_merge[CONT_FEATURES] = cont_vals

X_mnar = masks_merge[CAT_FEATURES_IMP + CONT_FEATURES].copy()
for c in CAT_FEATURES_IMP:
    X_mnar[c] = X_mnar[c].astype('category')

preds_mnar_lgb = np.clip(model_imp_83.predict(X_mnar), 0, None)
traccia_a_4 = evaluate_on_mnar_batch(preds_mnar_lgb, masks_val, 'LGB imputer (train 83)')

del X_mnar, masks_merge, masks_merge_idx
gc.collect()

# --- Passo 3: Retrain su gg 1-90 ---
print('\n  Passo 3: Retrain LGB su gg 1-90 (in-stock)...')
t0 = time.time()

X_train_90, y_train_90 = build_lgb_imputer_data(1, 90, instock_only=True)
print(f'  Train: {len(X_train_90):,} ore in-stock')

lgb_train_90 = lgb.Dataset(X_train_90, y_train_90, free_raw_data=True)

# Use best_iteration from step 1
model_imp_90 = lgb.train(
    LGB_PARAMS, lgb_train_90,
    num_boost_round=model_imp_83.best_iteration,
)
print(f'  Trained {model_imp_83.best_iteration} rounds, time: {time.time()-t0:.0f}s')

del X_train_90, y_train_90, lgb_train_90
gc.collect()

# --- Passo 4: Imputazione tutte le ore stockout gg 1-90 ---
print('\n  Passo 4: Imputazione ore stockout (gg 1-90)...')

# Build features for all stockout hours
so_mask = stock_flat == 1
feat_so = {
    'store_id': store_flat[so_mask],
    'product_id': product_flat[so_mask],
    'city_id': city_flat[so_mask],
    'dow': dow_flat[so_mask],
    'hour': hours_flat[so_mask],
}
for j, c in enumerate(CONT_FEATURES):
    feat_so[c] = conts_flat[so_mask, j]

X_so = pd.DataFrame(feat_so)
for c in CAT_FEATURES_IMP:
    X_so[c] = X_so[c].astype('category')

preds_so = np.clip(model_imp_90.predict(X_so), 0, None).astype(np.float32)

imputed_flat4 = np.zeros(n_hourly, dtype=np.float32)
imputed_flat4[so_mask] = preds_so

build_completed_sales(imputed_flat4, 'lgb')

del X_so, preds_so, model_imp_83, model_imp_90
gc.collect()


# ===========================================================================
# SUMMARY: Traccia A
# ===========================================================================
print('\n' + '=' * 72)
print('  TRACCIA A — Ranking imputer (maschere MNAR, gg 84-90)')
print('=' * 72)

traccia_a = {
    'Media condizionata': traccia_a_1,
    'Media globale': traccia_a_2,
    'Mediana condizionata': traccia_a_3,
    'LGB imputer': traccia_a_4,
}

print(f'\n  {"Imputer":<24} {"WAPE_recovery":>14} {"WPE_recovery":>13} {"N_masked":>10}')
print('  ' + '-' * 64)

for label, r in traccia_a.items():
    print(f'  {label:<24} {r["wape_recovery"]:>14.4f} {r["wpe_recovery"]:>13.4f} '
          f'{r["n_masked"]:>10,}')

# Save Traccia A results
traccia_a_df = pd.DataFrame([
    {'imputer': k, **v} for k, v in traccia_a.items()
])
traccia_a_df.to_parquet(os.path.join(RESULTS_DIR, 'traccia_a_naive_ml.parquet'), index=False)
print(f'\n  Salvato: traccia_a_naive_ml.parquet')

# Verify completed_sales files
print('\n  File completed_sales generati:')
for f in sorted(os.listdir(COMPLETED_DIR)):
    path = os.path.join(COMPLETED_DIR, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f'    {f}: {size_mb:.1f} MB')

print('\n' + '=' * 72)
print('  DONE — 05_imputation_naive_ml.py')
print('=' * 72)
