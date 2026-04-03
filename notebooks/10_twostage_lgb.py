"""
10_twostage_lgb.py — Two-Stage: Imputation + LightGBM Forecasting
=================================================================
PINN-Retail: Physics-Informed Neural Networks per Demand Forecasting
di Prodotti Deperibili.

Passo 4: Two-stage approach per gestire il censoring da stockout.

Stage 1 — Imputation:
  Stima la domanda latente D̂ durante le ore di stockout.
  Metodi: (a) Conditional Mean, (b) LightGBM su in-stock hours.

Stage 2 — Forecasting:
  Allena LightGBM con M5-style lag features calcolati dal dataset
  completato (decontaminato dagli zeri da stockout).

Eseguire con: freshnet/bin/python notebooks/10_twostage_lgb.py
"""

import sys
import os
import gc
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lightgbm as lgb

from src.evaluation.metrics import compute_metrics, format_metrics_table

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

CONT_FEATURES = ['discount', 'avg_temperature', 'avg_humidity',
                  'precpt', 'avg_wind_level', 'holiday_flag', 'activity_flag']
CAT_FEATURES = ['store_id', 'product_id', 'city_id', 'dow', 'hour']
ALL_BASE_FEATURES = CAT_FEATURES + CONT_FEATURES

LAG_FEATURES_F = [
    'lag_1d', 'lag_7d', 'lag_14d',           # raw lags (same hour)
    'rmean_7d', 'rmean_14d', 'rstd_7d',      # rolling stats (same hour)
    'lag_dow', 'rmean_dow',                    # day-of-week specific
    'daily_total_lag1', 'daily_total_rmean7',  # daily aggregates
    'momentum_1d_7d',                          # ratio lag_1d / rmean_7d
]

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

IMPUTATION_METHODS = ['cm', 'lgb']
IMPUTATION_LABELS = {
    'cm': 'Conditional Mean',
    'lgb': 'LGB Imputation',
}

# ===========================================================================
print('=' * 72)
print('  TWO-STAGE: IMPUTATION + LIGHTGBM FORECASTING')
print('=' * 72)

# ---------------------------------------------------------------------------
# 1. Caricamento dati
# ---------------------------------------------------------------------------
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

n_series = df_full.groupby(['store_id', 'product_id']).ngroups
print(f'  Train: {len(df_train):,} righe, giorni 1-90')
print(f'  Eval:  {len(df_eval):,} righe, giorni 91-97')
print(f'  Full:  {len(df_full):,} righe, {len(all_dates)} giorni, {n_series:,} serie')

del df_train, df_eval

# Pre-parse hourly arrays
sales_all = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)    # (N_full, 24)
stock_all = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)  # (N_full, 24)

# ---------------------------------------------------------------------------
# 2. Build series cache (una volta sola, riusata ovunque)
# ---------------------------------------------------------------------------
print('\n2. Costruzione series_cache...')
t0_cache = time.time()

series_cache = {}
for (sid, pid), grp in df_full.groupby(['store_id', 'product_id'], sort=False):
    grp_s = grp.sort_values('day_num')
    idx = grp_s.index.values
    series_cache[(sid, pid)] = {
        'days': grp_s['day_num'].values,
        'dows': grp_s['dow'].values,
        'sales': sales_all[idx],              # (N_days, 24) — raw S_obs
        'stock': stock_all[idx],              # (N_days, 24) — 0=in-stock, 1=stockout
        'city_id': grp_s['city_id'].values[0],
        'conts': grp_s[CONT_FEATURES].values.astype(np.float32),  # (N_days, 7)
    }

print(f'  {len(series_cache):,} serie, '
      f'tempo: {time.time() - t0_cache:.1f}s')

# ===========================================================================
# STAGE 1: IMPUTATION
# ===========================================================================
print('\n' + '=' * 72)
print('  STAGE 1: IMPUTATION')
print('=' * 72)


# ---------------------------------------------------------------------------
# 3a. Conditional Mean imputation
# ---------------------------------------------------------------------------
def impute_conditional_mean(series_cache, anchor_day):
    """Impute stockout hours using conditional mean per (series, dow, hour).

    For each (store, product, dow, hour), D̂ = mean(S_obs | stock=0, day ≤ anchor_day).
    Fallback hierarchy:
      1. (store, product, dow, hour) if ≥ 3 in-stock observations
      2. (store, product, hour) across all DOWs
      3. (store, product) overall mean / 24

    Returns: dict {(sid,pid): completed_sales array (N_days, 24)}
    """
    completed = {}

    for (sid, pid), sc in series_cache.items():
        days = sc['days']
        dows = sc['dows']
        sales = sc['sales']   # (N_days, 24)
        stock = sc['stock']   # (N_days, 24)
        n_days = len(days)

        # Use only days ≤ anchor_day for computing profiles
        train_mask = days <= anchor_day
        train_sales = sales[train_mask]     # (K, 24)
        train_stock = stock[train_mask]     # (K, 24)
        train_dows = dows[train_mask]       # (K,)

        # Compute per-(dow, hour) mean from in-stock observations
        # dow_hour_profiles[dow, hour] = mean(S_obs | stock=0)
        dow_hour_sum = np.zeros((7, 24), dtype=np.float64)
        dow_hour_cnt = np.zeros((7, 24), dtype=np.int32)

        for di in range(len(train_sales)):
            d = train_dows[di]
            for h in range(24):
                if train_stock[di, h] == 0:  # in-stock
                    dow_hour_sum[d, h] += train_sales[di, h]
                    dow_hour_cnt[d, h] += 1

        # Fallback 2: (store, product, hour) across all DOWs
        hour_sum = dow_hour_sum.sum(axis=0)  # (24,)
        hour_cnt = dow_hour_cnt.sum(axis=0)  # (24,)

        # Fallback 3: (store, product) overall mean
        total_sum = hour_sum.sum()
        total_cnt = hour_cnt.sum()
        overall_mean_per_hour = (total_sum / total_cnt / 24.0) if total_cnt > 0 else 0.0

        # Build completed sales
        completed_sales = sales.copy()  # start with raw S_obs

        for di in range(n_days):
            d = dows[di]
            for h in range(24):
                if stock[di, h] == 1:  # stockout → impute
                    if dow_hour_cnt[d, h] >= 3:
                        completed_sales[di, h] = dow_hour_sum[d, h] / dow_hour_cnt[d, h]
                    elif hour_cnt[h] >= 1:
                        completed_sales[di, h] = hour_sum[h] / hour_cnt[h]
                    else:
                        completed_sales[di, h] = overall_mean_per_hour

        completed[(sid, pid)] = completed_sales

    return completed


print('\n3a. Conditional Mean imputation...')
t0 = time.time()

# For train+val imputation: anchor_day = 83 (use training data only)
completed_cm_83 = impute_conditional_mean(series_cache, anchor_day=83)
# For test imputation: anchor_day = 90 (use training + val data)
completed_cm_90 = impute_conditional_mean(series_cache, anchor_day=90)

# Merge: for each series, use anchor=83 profile for days ≤ 90, anchor=90 for days 91-97
# Actually simpler: build one completed_sales dict using appropriate profile per day
completed_cm = {}
for (sid, pid), sc in series_cache.items():
    days = sc['days']
    c83 = completed_cm_83[(sid, pid)]
    c90 = completed_cm_90[(sid, pid)]
    result = np.empty_like(c83)
    test_mask = days >= 91
    result[~test_mask] = c83[~test_mask]
    result[test_mask] = c90[test_mask]
    completed_cm[(sid, pid)] = result

del completed_cm_83, completed_cm_90
gc.collect()

elapsed_cm = time.time() - t0
print(f'  Completato in {elapsed_cm:.1f}s')

# Diagnostica CM
n_imputed_cm = 0
sum_imputed_cm = 0.0
n_stockout_total = 0
for (sid, pid), sc in series_cache.items():
    so_mask = sc['stock'] == 1
    n_so = so_mask.sum()
    n_stockout_total += n_so
    imp_vals = completed_cm[(sid, pid)][so_mask]
    n_imputed_cm += n_so
    sum_imputed_cm += imp_vals.sum()

mean_imputed_cm = sum_imputed_cm / n_imputed_cm if n_imputed_cm > 0 else 0
print(f'  Ore stockout imputate: {n_imputed_cm:,}')
print(f'  Media D̂ imputata (CM): {mean_imputed_cm:.6f}')

# Sanity check: media in-stock
n_instock = 0
sum_instock = 0.0
for (sid, pid), sc in series_cache.items():
    in_mask = sc['stock'] == 0
    n_instock += in_mask.sum()
    sum_instock += sc['sales'][in_mask].sum()
mean_instock = sum_instock / n_instock if n_instock > 0 else 0
print(f'  Media S_obs in-stock:   {mean_instock:.6f}')
print(f'  Ratio D̂/S_obs_in:      {mean_imputed_cm / mean_instock:.4f}' if mean_instock > 0 else '')


# ---------------------------------------------------------------------------
# 3b. LGB Imputation (model-based)
# ---------------------------------------------------------------------------
def build_hourly_dataset_instock(df, sales_arr, stock_arr, split):
    """Build hourly dataset from IN-STOCK hours only, base features (no lags).

    Used for Stage 1b: training the imputation model.
    """
    if split == 'train':
        d_min, d_max = 2, 83
    elif split == 'val':
        d_min, d_max = 84, 90
    elif split == 'test':
        d_min, d_max = 91, 97

    mask_split = (df['day_num'] >= d_min) & (df['day_num'] <= d_max)
    df_split = df[mask_split]
    idx_split = np.where(mask_split.values)[0]
    n_days = len(df_split)

    store_ids_day = df_split['store_id'].values
    product_ids_day = df_split['product_id'].values
    city_ids_day = df_split['city_id'].values
    dows_day = df_split['dow'].values
    conts_day = df_split[CONT_FEATURES].values.astype(np.float32)
    sales_day = sales_arr[idx_split]   # (n_days, 24)
    stock_day = stock_arr[idx_split]   # (n_days, 24)

    # Expand to hourly
    n_hourly = n_days * 24
    hours = np.tile(np.arange(24, dtype=np.int32), n_days)
    store_ids_h = np.repeat(store_ids_day, 24)
    product_ids_h = np.repeat(product_ids_day, 24)
    city_ids_h = np.repeat(city_ids_day, 24)
    dows_h = np.repeat(dows_day, 24)
    conts_h = np.repeat(conts_day, 24, axis=0)
    y = sales_day.ravel().astype(np.float32)
    stock_flat = stock_day.ravel().astype(np.float32)

    # Filter: keep only in-stock hours (stock_status == 0)
    instock_mask = stock_flat == 0
    n_instock = instock_mask.sum()

    feat_dict = {
        'store_id': store_ids_h[instock_mask],
        'product_id': product_ids_h[instock_mask],
        'city_id': city_ids_h[instock_mask],
        'dow': dows_h[instock_mask],
        'hour': hours[instock_mask],
    }
    for j, c in enumerate(CONT_FEATURES):
        feat_dict[c] = conts_h[instock_mask, j]

    X = pd.DataFrame(feat_dict)
    del feat_dict
    gc.collect()

    for c in CAT_FEATURES:
        X[c] = X[c].astype('category')

    return X, y[instock_mask], n_instock


def build_hourly_dataset_all_base(df, sales_arr, stock_arr, split):
    """Build hourly dataset for ALL hours, base features only (no lags).

    Used for predicting D̂ on stockout hours.
    Returns X, y, stock_flat, store_ids, product_ids.
    """
    if split == 'train':
        d_min, d_max = 2, 83
    elif split == 'val':
        d_min, d_max = 84, 90
    elif split == 'test':
        d_min, d_max = 91, 97

    mask_split = (df['day_num'] >= d_min) & (df['day_num'] <= d_max)
    df_split = df[mask_split]
    idx_split = np.where(mask_split.values)[0]
    n_days = len(df_split)

    store_ids_day = df_split['store_id'].values
    product_ids_day = df_split['product_id'].values
    city_ids_day = df_split['city_id'].values
    dows_day = df_split['dow'].values
    conts_day = df_split[CONT_FEATURES].values.astype(np.float32)
    sales_day = sales_arr[idx_split]
    stock_day = stock_arr[idx_split]

    n_hourly = n_days * 24
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
    del feat_dict
    gc.collect()

    for c in CAT_FEATURES:
        X[c] = X[c].astype('category')

    return X, y, stock_flat, store_ids_h, product_ids_h


print('\n3b. LGB Imputation (train on in-stock hours)...')
t0 = time.time()

# Build in-stock training dataset
print('    Costruzione dataset in-stock train...')
X_imp_train, y_imp_train, n_imp_train = \
    build_hourly_dataset_instock(df_full, sales_all, stock_all, 'train')
print(f'    In-stock train: {n_imp_train:,} righe, {X_imp_train.shape[1]} features')

# Build in-stock val dataset for early stopping
print('    Costruzione dataset in-stock val...')
X_imp_val, y_imp_val, n_imp_val = \
    build_hourly_dataset_instock(df_full, sales_all, stock_all, 'val')
print(f'    In-stock val:   {n_imp_val:,} righe')

# Train imputation LGB
print('    Training LGB imputation...')
lgb_imp_train = lgb.Dataset(X_imp_train, y_imp_train, free_raw_data=True)
lgb_imp_val = lgb.Dataset(X_imp_val, y_imp_val, reference=lgb_imp_train, free_raw_data=True)

callbacks_imp = [
    lgb.early_stopping(EARLY_STOPPING_ROUNDS),
    lgb.log_evaluation(50),
]

imp_model = lgb.train(
    LGB_PARAMS, lgb_imp_train,
    num_boost_round=MAX_BOOST_ROUNDS,
    valid_sets=[lgb_imp_val],
    valid_names=['val'],
    callbacks=callbacks_imp,
)

print(f'    Best iter: {imp_model.best_iteration}, '
      f'MAE: {imp_model.best_score["val"]["l1"]:.6f}')

del X_imp_train, y_imp_train, lgb_imp_train
del X_imp_val, y_imp_val, lgb_imp_val
gc.collect()

# Save imputation model
imp_model.save_model(os.path.join(RESULTS_DIR, 'twostage_imputation_lgb.txt'))

# Predict D̂ for ALL hours (train + val + test), then replace stockout hours
print('    Predicting D̂ per tutte le ore...')
completed_lgb = {}

for split_name in ['train', 'val', 'test']:
    print(f'      Predicting {split_name}...')
    X_all, y_all, stock_flat, sids, pids = \
        build_hourly_dataset_all_base(df_full, sales_all, stock_all, split_name)

    preds = imp_model.predict(X_all)
    preds = np.clip(preds, 0, None)  # ensure non-negative

    # Store predictions for stockout hours into completed_lgb
    if split_name == 'train':
        d_min, d_max = 2, 83
    elif split_name == 'val':
        d_min, d_max = 84, 90
    else:
        d_min, d_max = 91, 97

    mask_split = (df_full['day_num'] >= d_min) & (df_full['day_num'] <= d_max)
    df_split = df_full[mask_split]
    sids_day = df_split['store_id'].values
    pids_day = df_split['product_id'].values
    day_nums = df_split['day_num'].values

    # Reshape preds to (n_days, 24)
    n_days_split = len(df_split)
    preds_2d = preds.reshape(n_days_split, 24)
    stock_2d = stock_flat.reshape(n_days_split, 24)
    y_2d = y_all.reshape(n_days_split, 24)

    for di in range(n_days_split):
        sid = sids_day[di]
        pid = pids_day[di]
        key = (sid, pid)
        sc = series_cache[key]
        # Find the index of this day in the series
        day_val = day_nums[di]
        day_idx = np.searchsorted(sc['days'], day_val)

        if key not in completed_lgb:
            completed_lgb[key] = sc['sales'].copy()

        for h in range(24):
            if stock_2d[di, h] == 1:  # stockout → use imputed
                completed_lgb[key][day_idx, h] = preds_2d[di, h]

    del X_all, y_all, preds, stock_flat, sids, pids
    gc.collect()

del imp_model
gc.collect()

elapsed_lgb_imp = time.time() - t0
print(f'  LGB imputation completata in {elapsed_lgb_imp:.1f}s')

# Diagnostica LGB imputation
n_imputed_lgb = 0
sum_imputed_lgb = 0.0
for (sid, pid), sc in series_cache.items():
    so_mask = sc['stock'] == 1
    n_so = so_mask.sum()
    n_imputed_lgb += n_so
    imp_vals = completed_lgb[(sid, pid)][so_mask]
    sum_imputed_lgb += imp_vals.sum()

mean_imputed_lgb = sum_imputed_lgb / n_imputed_lgb if n_imputed_lgb > 0 else 0
print(f'  Media D̂ imputata (LGB): {mean_imputed_lgb:.6f}')
print(f'  Media S_obs in-stock:    {mean_instock:.6f}')
print(f'  Ratio D̂/S_obs_in:       {mean_imputed_lgb / mean_instock:.4f}' if mean_instock > 0 else '')


# ===========================================================================
# STAGE 2: LGB FORECASTING ON COMPLETED DATA
# ===========================================================================
print('\n' + '=' * 72)
print('  STAGE 2: LGB FORECASTING ON COMPLETED DATA')
print('=' * 72)


def build_hourly_dataset_completed(df, completed_sales, stock_arr, series_cache,
                                    split):
    """Build flat per-hour dataset with M5 lag features from completed sales.

    Like build_hourly_dataset variant F from notebook 09, but:
    - Lag features computed from completed_sales (not raw S_obs)
    - Target = completed Y(t) (not S_obs)
    - stock_flat from original data (for stratified evaluation)
    """
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
    day_nums_day = df_split['day_num'].values

    # Target = completed Y(t), stock from original
    stock_day = stock_arr[idx_split]    # (n_days, 24) — original stock_status

    # Expand to hourly
    n_hourly = n_days * 24
    hours = np.tile(np.arange(24, dtype=np.int32), n_days)
    store_ids_h = np.repeat(store_ids_day, 24)
    product_ids_h = np.repeat(product_ids_day, 24)
    city_ids_h = np.repeat(city_ids_day, 24)
    dows_h = np.repeat(dows_day, 24)
    conts_h = np.repeat(conts_day, 24, axis=0)
    stock_flat = stock_day.ravel().astype(np.float32)

    # Build target from completed_sales
    y = np.empty(n_hourly, dtype=np.float32)
    for di in range(n_days):
        sid = store_ids_day[di]
        pid = product_ids_day[di]
        d = day_nums_day[di]
        sc = series_cache[(sid, pid)]
        day_idx = np.searchsorted(sc['days'], d)
        hs = di * 24
        y[hs:hs + 24] = completed_sales[(sid, pid)][day_idx]

    # Base features
    feat_dict = {
        'store_id': store_ids_h,
        'product_id': product_ids_h,
        'city_id': city_ids_h,
        'dow': dows_h,
        'hour': hours,
    }
    for j, c in enumerate(CONT_FEATURES):
        feat_dict[c] = conts_h[:, j]

    # M5-style lag features from completed_sales
    lag_arrays = {name: np.full(n_hourly, np.nan, dtype=np.float32)
                  for name in LAG_FEATURES_F}

    print(f'      Computing lag features from completed sales ({n_days:,} days)...')
    for row_i in range(n_days):
        if (row_i + 1) % 500000 == 0:
            print(f'        ... {row_i+1:,}/{n_days:,}')

        sid = store_ids_day[row_i]
        pid = product_ids_day[row_i]
        d = day_nums_day[row_i]
        dow_val = dows_day[row_i]

        sc = series_cache[(sid, pid)]
        s_days = sc['days']
        s_dows = sc['dows']
        # Use COMPLETED sales for lag computation (key difference from 09)
        s_sales = completed_sales[(sid, pid)]

        # Anchor: train=rolling(d-1), val=fixed(83), test=fixed(90)
        if split == 'train':
            a_day = d - 1
        elif split == 'val':
            a_day = 83
        else:
            a_day = 90

        avail_mask = s_days <= a_day
        K = int(avail_mask.sum())
        hs = row_i * 24
        he = hs + 24

        if K > 0:
            avail_sales = s_sales[avail_mask]
            avail_dows = s_dows[avail_mask]

            # Raw lags
            lag_arrays['lag_1d'][hs:he] = avail_sales[-1]
            if K >= 7:
                lag_arrays['lag_7d'][hs:he] = avail_sales[-7]
            if K >= 14:
                lag_arrays['lag_14d'][hs:he] = avail_sales[-14]

            # Rolling means
            if K >= 7:
                lag_arrays['rmean_7d'][hs:he] = avail_sales[-7:].mean(axis=0)
            if K >= 14:
                lag_arrays['rmean_14d'][hs:he] = avail_sales[-14:].mean(axis=0)

            # Rolling std
            if K >= 2:
                w = min(7, K)
                lag_arrays['rstd_7d'][hs:he] = avail_sales[-w:].std(axis=0)

            # DoW-specific
            same_dow = avail_dows == dow_val
            if same_dow.any():
                dow_sales = avail_sales[same_dow]
                lag_arrays['lag_dow'][hs:he] = dow_sales[-1]
                lag_arrays['rmean_dow'][hs:he] = dow_sales.mean(axis=0)

            # Daily aggregates
            daily_totals = avail_sales.sum(axis=1)
            lag_arrays['daily_total_lag1'][hs:he] = daily_totals[-1]
            if K >= 7:
                lag_arrays['daily_total_rmean7'][hs:he] = daily_totals[-7:].mean()

    # Momentum
    l1 = lag_arrays['lag_1d']
    rm7 = lag_arrays['rmean_7d']
    valid_mom = (~np.isnan(l1)) & (~np.isnan(rm7)) & (rm7 > 0)
    lag_arrays['momentum_1d_7d'][valid_mom] = l1[valid_mom] / rm7[valid_mom]

    for name in LAG_FEATURES_F:
        feat_dict[name] = lag_arrays[name]

    # Print NaN stats
    print('      NaN counts per lag feature:')
    for name in LAG_FEATURES_F:
        nan_count = np.isnan(lag_arrays[name]).sum()
        pct = 100.0 * nan_count / n_hourly
        print(f'        {name:<22} {nan_count:>12,} ({pct:.1f}%)')

    del lag_arrays
    gc.collect()

    X = pd.DataFrame(feat_dict)
    del feat_dict
    gc.collect()

    for c in CAT_FEATURES:
        X[c] = X[c].astype('category')

    return X, y, stock_flat, store_ids_h, product_ids_h


# ---------------------------------------------------------------------------
# 4. Train Stage 2 LGB for each imputation method
# ---------------------------------------------------------------------------
completed_dicts = {
    'cm': completed_cm,
    'lgb': completed_lgb,
}

stage2_results = {}

for imp_method in IMPUTATION_METHODS:
    imp_label = IMPUTATION_LABELS[imp_method]
    print(f'\n  --- Stage 2: {imp_label} ---')
    t0 = time.time()

    completed = completed_dicts[imp_method]

    # Build train dataset
    print(f'    Costruzione dataset train (lags da {imp_label})...')
    X_train, y_train, _, _, _ = \
        build_hourly_dataset_completed(df_full, completed, stock_all,
                                        series_cache, 'train')
    print(f'    Train: {len(X_train):,} righe, {X_train.shape[1]} features')

    # Build val dataset
    print(f'    Costruzione dataset val...')
    X_val, y_val_completed, stock_val, sids_val, pids_val = \
        build_hourly_dataset_completed(df_full, completed, stock_all,
                                        series_cache, 'val')
    print(f'    Val:   {len(X_val):,} righe')

    # For early stopping, we need S_obs on val (not completed Y)
    # Reconstruct y_val_obs from original data
    mask_val = (df_full['day_num'] >= 84) & (df_full['day_num'] <= 90)
    y_val_obs = sales_all[mask_val.values].ravel().astype(np.float32)

    # Custom evaluation: WAPE on in-stock hours only using S_obs as ground truth
    # LGB early stopping uses its built-in MAE metric on completed Y (close enough)
    # But we also track WAPE_instock manually for selection

    # Train LGB
    print(f'    Training LightGBM Stage 2...')
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=True)
    lgb_val_ds = lgb.Dataset(X_val, y_val_completed, reference=lgb_train,
                              free_raw_data=True)

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

    # Predict on val
    preds_val = np.clip(model.predict(X_val), 0, None)

    # Compute val WAPE_instock against S_obs (ground truth)
    instock_mask_val = stock_val == 0
    sae_in = np.abs(preds_val[instock_mask_val] - y_val_obs[instock_mask_val]).sum()
    sao_in = np.abs(y_val_obs[instock_mask_val]).sum()
    val_wape_instock = sae_in / sao_in if sao_in > 0 else float('inf')

    # Compute val WAPE overall against S_obs
    sae_all = np.abs(preds_val - y_val_obs).sum()
    sao_all = np.abs(y_val_obs).sum()
    val_wape_overall = sae_all / sao_all if sao_all > 0 else float('inf')

    # Val WAPE_instock median per-serie
    df_tmp = pd.DataFrame({
        'sid': sids_val, 'pid': pids_val,
        'abs_err': np.abs(preds_val - y_val_obs) * (stock_val == 0).astype(np.float32),
        'abs_obs': np.abs(y_val_obs) * (stock_val == 0).astype(np.float32),
    })
    grp_sums = df_tmp.groupby(['sid', 'pid'], sort=False)[['abs_err', 'abs_obs']].sum()
    valid = grp_sums['abs_obs'] > 0
    ps_wapes = (grp_sums.loc[valid, 'abs_err'] / grp_sums.loc[valid, 'abs_obs']).values
    med_wape_instock = np.median(ps_wapes) if len(ps_wapes) > 0 else np.nan
    del df_tmp, grp_sums

    elapsed = time.time() - t0
    stage2_results[imp_method] = {
        'wape_instock_pooled': val_wape_instock,
        'wape_instock_median': med_wape_instock,
        'wape_overall_pooled': val_wape_overall,
        'best_iter': best_iter,
        'best_mae': best_score,
        'elapsed': elapsed,
    }

    print(f'    Best iter: {best_iter}, MAE (completed): {best_score:.6f}')
    print(f'    Val WAPE_instock pooled: {val_wape_instock:.6f}, '
          f'median: {med_wape_instock:.6f}')
    print(f'    Val WAPE_overall pooled: {val_wape_overall:.6f}')
    print(f'    Tempo: {elapsed:.1f}s')

    # Save model
    model.save_model(os.path.join(RESULTS_DIR,
                                   f'twostage_{imp_method}_forecaster.txt'))

    del lgb_train, lgb_val_ds, model, X_train, y_train
    del X_val, y_val_completed, preds_val
    gc.collect()


# ---------------------------------------------------------------------------
# 5. Selection table
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  STAGE 2: SELEZIONE METODO IMPUTATION')
print('=' * 72)

print(f'\n  {"Method":<18} {"WAPE_in pool":>14} {"WAPE_in med":>14} '
      f'{"WAPE_all pool":>14} {"Iter":>6} {"Time":>8}')
print('  ' + '-' * 78)

for m in IMPUTATION_METHODS:
    r = stage2_results[m]
    label = IMPUTATION_LABELS[m]
    print(f'  {label:<18} {r["wape_instock_pooled"]:>14.6f} '
          f'{r["wape_instock_median"]:>14.6f} '
          f'{r["wape_overall_pooled"]:>14.6f} '
          f'{r["best_iter"]:>6d} {r["elapsed"]:>7.1f}s')

# Reference: single-stage LGB var F
print(f'\n  {"LGB var F (ref)":<18} {"0.883800":>14} {"—":>14} '
      f'{"0.998100":>14} {"497":>6}')

best_imp = min(IMPUTATION_METHODS,
               key=lambda m: stage2_results[m]['wape_instock_pooled'])
print(f'\n  Best imputation: {IMPUTATION_LABELS[best_imp]} '
      f'(WAPE_instock pooled: {stage2_results[best_imp]["wape_instock_pooled"]:.6f})')


# ---------------------------------------------------------------------------
# 6. Full evaluation on val + test with best imputation method
# ---------------------------------------------------------------------------
print(f'\n' + '=' * 72)
print(f'  EVALUAZIONE COMPLETA — Two-Stage ({IMPUTATION_LABELS[best_imp]})')
print('=' * 72)

best_forecaster = lgb.Booster(
    model_file=os.path.join(RESULTS_DIR,
                             f'twostage_{best_imp}_forecaster.txt'))
best_completed = completed_dicts[best_imp]

pooled_results = {}
per_series_dfs = {}

for split_name in ['val', 'test']:
    print(f'\n  Valutazione {split_name}...')
    X_split, y_completed, stock_split, sids, pids = \
        build_hourly_dataset_completed(df_full, best_completed, stock_all,
                                        series_cache, split_name)
    print(f'    {len(X_split):,} righe')

    # Get original S_obs for evaluation
    if split_name == 'val':
        d_min, d_max = 84, 90
    else:
        d_min, d_max = 91, 97
    mask_split = (df_full['day_num'] >= d_min) & (df_full['day_num'] <= d_max)
    y_obs = sales_all[mask_split.values].ravel().astype(np.float32)

    preds = np.clip(best_forecaster.predict(X_split), 0, None)

    # Pooled metrics (against S_obs)
    r = {}
    for sub, smask in [('overall', np.ones(len(preds), dtype=bool)),
                       ('instock', stock_split == 0),
                       ('stockout', stock_split == 1)]:
        ef = (preds - y_obs)[smask]
        of = y_obs[smask]
        sae = np.abs(ef).sum()
        sao = np.abs(of).sum()
        r[f'wape_{sub}'] = sae / sao if sao > 0 else np.nan
        r[f'wpe_{sub}'] = ef.sum() / of.sum() if of.sum() != 0 else np.nan
        r[f'n_{sub}'] = int(smask.sum())
    pooled_results[split_name] = r

    # Per-series metrics
    print('    Calcolo metriche per-serie...')
    df_eval_flat = pd.DataFrame({
        'store_id': sids,
        'product_id': pids,
        'pred': preds.astype(np.float64),
        'obs': y_obs.astype(np.float64),
        'stock': stock_split.astype(np.float64),
    })
    df_eval_flat['abs_err'] = np.abs(df_eval_flat['pred'] - df_eval_flat['obs'])
    df_eval_flat['err'] = df_eval_flat['pred'] - df_eval_flat['obs']
    df_eval_flat['abs_obs'] = np.abs(df_eval_flat['obs'])

    records = []
    for (sid, pid), grp in df_eval_flat.groupby(['store_id', 'product_id'],
                                                  sort=False):
        m = {}
        for sub, smask_fn in [
            ('overall', lambda g: np.ones(len(g), dtype=bool)),
            ('instock', lambda g: g['stock'].values == 0),
            ('stockout', lambda g: g['stock'].values == 1),
        ]:
            smask = smask_fn(grp)
            n = int(smask.sum())
            m[f'n_{sub}'] = n
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
    per_series_dfs[split_name] = ps
    out_path = os.path.join(RESULTS_DIR,
                             f'twostage_{best_imp}_{split_name}_per_series.parquet')
    ps.to_parquet(out_path, index=False)
    print(f'    Salvato: {out_path} ({len(ps):,} serie)')

    del X_split, preds, df_eval_flat
    gc.collect()

# Also save per-series for the other imputation method (if useful for comparison)
# We skip this for now to save memory/time

# ---------------------------------------------------------------------------
# 7. Tabella risultati pooled
# ---------------------------------------------------------------------------
print(format_metrics_table(pooled_results,
                            model_name=f'Two-Stage ({IMPUTATION_LABELS[best_imp]})'))

# ---------------------------------------------------------------------------
# 8. Distribuzione per-serie
# ---------------------------------------------------------------------------
METRIC_COLS = ['wape_overall', 'wape_instock', 'wape_stockout',
               'wpe_overall', 'wpe_instock', 'wpe_stockout']

print('\n' + '=' * 72)
print('  DISTRIBUZIONE METRICHE PER-SERIE')
print('=' * 72)

print(f'\n  {"Split":<8} {"Metric":<16} {"Mean":>8} {"Median":>8} '
      f'{"Std":>8} {"Q5":>8} {"Q95":>8} {"Valid":>7}')
print('  ' + '-' * 80)

for split_name, ps in per_series_dfs.items():
    for col in METRIC_COLS:
        vals = ps[col].dropna()
        if len(vals) == 0:
            continue
        q5, q95 = np.quantile(vals, [0.05, 0.95])
        print(f'  {split_name:<8} {col:<16} {vals.mean():>8.4f} '
              f'{vals.median():>8.4f} {vals.std():>8.4f} '
              f'{q5:>8.4f} {q95:>8.4f} {len(vals):>7,}')

# ---------------------------------------------------------------------------
# 9. Confronto con tutti i baseline
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  CONFRONTO CON TUTTI I BASELINE (in-stock, test)')
print('=' * 72)

all_baselines = {
    'Naive (direct)': 'naive_direct',
    'MA K=14 (direct)': 'ma_direct_K14',
    'Global Mean': 'global_mean',
    'DoW Mean': 'dow_mean',
    'MLP (var A)': 'mlp',
    'LGB (var F)': 'lgb',
    f'2-Stage ({best_imp})': f'twostage_{best_imp}',
}

print(f'\n  {"Model":<24} {"WAPE_in pool":>14} {"WAPE_in med":>14} '
      f'{"WPE_in med":>12} {"WAPE_all pool":>14}')
print('  ' + '-' * 82)

for label, prefix in all_baselines.items():
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if not os.path.exists(path):
        continue
    ps_bl = pd.read_parquet(path)
    wape_in_pool_bl = None
    # For pooled, recompute from per-series or use stored pooled_results
    if prefix == f'twostage_{best_imp}' and 'test' in pooled_results:
        wape_in_pool_val = pooled_results['test']['wape_instock']
    else:
        wape_in_pool_val = None
    wape_in_med = ps_bl['wape_instock'].median()
    wpe_in_med = ps_bl['wpe_instock'].median()
    wape_all_med = ps_bl['wape_overall'].median()
    # Print without pooled for other baselines (don't have it stored)
    if wape_in_pool_val is not None:
        print(f'  {label:<24} {wape_in_pool_val:>14.4f} {wape_in_med:>14.4f} '
              f'{wpe_in_med:>12.4f} {pooled_results["test"]["wape_overall"]:>14.4f}')
    else:
        print(f'  {label:<24} {"—":>14} {wape_in_med:>14.4f} '
              f'{wpe_in_med:>12.4f} {"—":>14}')


# ---------------------------------------------------------------------------
# 10. Figure
# ---------------------------------------------------------------------------
print('\n  Generazione figure...')

# Fig 37: Boxplot confronto in-stock (tutti i modelli incluso two-stage)
colors_all = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#DD8452',
              '#E24A33']

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle('Confronto Tutti i Modelli — Metriche In-Stock (per-serie)',
             fontsize=15, y=0.98)

for j, split in enumerate(['val', 'test']):
    for row, (metric, ylabel) in enumerate([('wape_instock', 'WAPE in-stock'),
                                              ('wpe_instock', 'WPE in-stock')]):
        ax = axes[row, j]
        box_data = []
        box_labels = []
        box_colors = []
        medians = []

        for k, (label, prefix) in enumerate(all_baselines.items()):
            path = os.path.join(RESULTS_DIR,
                                f'{prefix}_{split}_per_series.parquet')
            if not os.path.exists(path):
                continue
            ps_bl = pd.read_parquet(path)
            vals = ps_bl[metric].dropna()
            if metric.startswith('wape'):
                q99 = vals.quantile(0.99)
                box_data.append(vals.clip(upper=q99).values)
            else:
                q01, q99 = vals.quantile(0.01), vals.quantile(0.99)
                box_data.append(vals.clip(lower=q01, upper=q99).values)
            box_labels.append(label)
            box_colors.append(colors_all[k % len(colors_all)])
            medians.append(vals.median())

        if not box_data:
            continue

        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                        widths=0.6)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for ml in bp['medians']:
            ml.set_color('red')
            ml.set_linewidth(2)

        if metric.startswith('wpe'):
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

        for k, med in enumerate(medians):
            if metric.startswith('wape'):
                ax.text(k + 1, med + 0.01, f'{med:.3f}', ha='center',
                        va='bottom', fontsize=7, fontweight='bold', color='red')
            else:
                offset = 0.005 if med >= 0 else -0.005
                va = 'bottom' if med >= 0 else 'top'
                ax.text(k + 1, med + offset, f'{med:.4f}', ha='center',
                        va=va, fontsize=7, fontweight='bold', color='red')

        ax.set_title(f'{ylabel} — {split}', fontsize=13)
        ax.set_ylabel(ylabel if j == 0 else '')
        ax.tick_params(axis='x', rotation=30)

fig.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(FIG_DIR, 'fig37_compare_instock_all_with_twostage.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Salvata: {out_path}')

# Feature importance (Stage 2 forecaster)
print('\n  Feature importance (Stage 2)...')
fi_model = lgb.Booster(
    model_file=os.path.join(RESULTS_DIR,
                             f'twostage_{best_imp}_forecaster.txt'))
importance = fi_model.feature_importance(importance_type='gain')
feat_names = fi_model.feature_name()

fi = sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)
print(f'\n  {"Feature":<22} {"Importance (gain)":>18}')
print('  ' + '-' * 44)
for name, imp in fi[:15]:
    print(f'  {name:<22} {imp:>18,.0f}')


print('\n' + '=' * 72)
print('  DONE')
print('=' * 72)
