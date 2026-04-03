"""
09b_lgb_variant_a.py — LightGBM Variante A (no history) — per-series evaluation only
=====================================================================================
Script minimale per generare lgb_a_{val,test}_per_series.parquet.
Variante A: solo 12 features base (no lag features).

Eseguire con: freshnet/bin/python notebooks/09b_lgb_variant_a.py
"""

import sys
import os
import gc
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import lightgbm as lgb

SEED = 42
np.random.seed(SEED)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CONT_FEATURES = ['discount', 'avg_temperature', 'avg_humidity',
                  'precpt', 'avg_wind_level', 'holiday_flag', 'activity_flag']
CAT_FEATURES = ['store_id', 'product_id', 'city_id', 'dow', 'hour']

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

MAX_BOOST_ROUNDS = 500
EARLY_STOPPING_ROUNDS = 30

# ===========================================================================
print('=' * 72)
print('  LIGHTGBM VARIANTE A — PER-SERIES EVALUATION')
print('=' * 72)

# 1. Load data
print('\n1. Caricamento dati...')
df_train = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))

df_train['dt_parsed'] = pd.to_datetime(df_train['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])

df_full = pd.concat([df_train, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)

all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek

print(f'  Full: {len(df_full):,} righe')
del df_train, df_eval

# Pre-parse hourly arrays
sales_all = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)
stock_all = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)


def build_hourly_dataset_base(df, sales_arr, stock_arr, split):
    """Build flat per-hour dataset with base features only (no lag)."""
    if split == 'train':
        d_min, d_max = 2, 83
    elif split == 'val':
        d_min, d_max = 84, 90
    elif split == 'test':
        d_min, d_max = 91, 97

    mask = (df['day_num'] >= d_min) & (df['day_num'] <= d_max)
    df_split = df[mask]
    idx_split = np.where(mask.values)[0]
    n_days = len(df_split)

    store_ids_day = df_split['store_id'].values
    product_ids_day = df_split['product_id'].values
    city_ids_day = df_split['city_id'].values
    dows_day = df_split['dow'].values
    conts_day = df_split[CONT_FEATURES].values.astype(np.float32)

    sales_day = sales_arr[idx_split]
    stock_day = stock_arr[idx_split]

    hours = np.tile(np.arange(24, dtype=np.int32), n_days)
    store_ids_h = np.repeat(store_ids_day, 24)
    product_ids_h = np.repeat(product_ids_day, 24)
    city_ids_h = np.repeat(city_ids_day, 24)
    dows_h = np.repeat(dows_day, 24)
    conts_h = np.repeat(conts_day, 24, axis=0)

    y = sales_day.ravel().astype(np.float32)
    stock_flat = stock_day.ravel().astype(np.float32)

    feat_dict = {
        'store_id': store_ids_h,
        'product_id': product_ids_h,
        'city_id': city_ids_h,
        'dow': dows_h,
        'hour': hours,
    }
    for j, c in enumerate(CONT_FEATURES):
        feat_dict[c] = conts_h[:, j]

    X = pd.DataFrame(feat_dict)
    for c in CAT_FEATURES:
        X[c] = X[c].astype('category')

    return X, y, stock_flat, store_ids_h, product_ids_h


# 2. Build datasets
print('\n2. Costruzione dataset...')
t0 = time.time()

X_train, y_train, _, _, _ = build_hourly_dataset_base(df_full, sales_all, stock_all, 'train')
print(f'  Train: {len(X_train):,} righe, {X_train.shape[1]} features')

X_val, y_val, stock_val, sids_val, pids_val = \
    build_hourly_dataset_base(df_full, sales_all, stock_all, 'val')
print(f'  Val:   {len(X_val):,} righe')

# 3. Train
print('\n3. Training LightGBM variante A...')
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=True)
lgb_val_ds = lgb.Dataset(X_val, y_val, reference=lgb_train, free_raw_data=True)

callbacks = [
    lgb.early_stopping(EARLY_STOPPING_ROUNDS),
    lgb.log_evaluation(50),
]

model = lgb.train(
    LGB_PARAMS, lgb_train,
    num_boost_round=MAX_BOOST_ROUNDS,
    valid_sets=[lgb_val_ds],
    valid_names=['val'],
    callbacks=callbacks,
)

best_iter = model.best_iteration
best_score = model.best_score['val']['l1']
print(f'  Best iter: {best_iter}, MAE: {best_score:.6f}')

model.save_model(os.path.join(RESULTS_DIR, 'lgb_variant_A.txt'))

del lgb_train, lgb_val_ds, X_train, y_train
gc.collect()

# 4. Evaluate on val + test
print('\n4. Valutazione per-serie...')

for split_name in ['val', 'test']:
    print(f'\n  {split_name}...')
    X_split, y_split, stock_split, sids, pids = \
        build_hourly_dataset_base(df_full, sales_all, stock_all, split_name)
    print(f'    {len(X_split):,} righe')

    preds = np.clip(model.predict(X_split), 0, None)

    # Per-series metrics
    df_eval_flat = pd.DataFrame({
        'store_id': sids,
        'product_id': pids,
        'pred': preds.astype(np.float64),
        'obs': y_split.astype(np.float64),
        'stock': stock_split.astype(np.float64),
    })
    df_eval_flat['abs_err'] = np.abs(df_eval_flat['pred'] - df_eval_flat['obs'])
    df_eval_flat['err'] = df_eval_flat['pred'] - df_eval_flat['obs']
    df_eval_flat['abs_obs'] = np.abs(df_eval_flat['obs'])

    records = []
    for (sid, pid), grp in df_eval_flat.groupby(['store_id', 'product_id'], sort=False):
        m = {}
        for sub, smask_fn in [('overall', lambda g: np.ones(len(g), dtype=bool)),
                               ('instock', lambda g: g['stock'].values == 0),
                               ('stockout', lambda g: g['stock'].values == 1)]:
            smask = smask_fn(grp)
            m[f'n_{sub}'] = int(smask.sum())
            sao = grp['abs_obs'].values[smask].sum()
            so = grp['obs'].values[smask].sum()
            sae = grp['abs_err'].values[smask].sum()
            se = grp['err'].values[smask].sum()
            m[f'wape_{sub}'] = sae / sao if sao > 0 else np.nan
            m[f'wpe_{sub}'] = se / so if so != 0 else np.nan
        m['store_id'] = sid
        m['product_id'] = pid
        records.append(m)

    ps = pd.DataFrame(records)
    out_path = os.path.join(RESULTS_DIR, f'lgb_a_{split_name}_per_series.parquet')
    ps.to_parquet(out_path, index=False)
    print(f'    Salvato: {out_path} ({len(ps):,} serie)')

    # Pooled metrics
    sae = np.abs(preds - y_split).sum()
    sao = np.abs(y_split).sum()
    wape = sae / sao
    wpe = (preds - y_split).sum() / y_split.sum()
    print(f'    WAPE pooled: {wape:.6f}, WPE pooled: {wpe:.6f}')

    # Mediana
    wape_med = ps['wape_instock'].median()
    wpe_med = ps['wpe_instock'].median()
    print(f'    WAPE_in mediana: {wape_med:.4f}, WPE_in mediana: {wpe_med:.4f}')

    del X_split, preds
    gc.collect()

elapsed = time.time() - t0
print(f'\n  Tempo totale: {elapsed:.1f}s')
print('\n' + '=' * 72)
print('  DONE')
print('=' * 72)
