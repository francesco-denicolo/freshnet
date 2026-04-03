"""
05_fase_b_imputation.py — Fase B Step 0.3: Imputation Candidates
=================================================================
Piano: CLAUDE_SEQUENTIAL-2.md, Fase B, punto (3)

Candidati di imputation per la Traccia A (demand recovery):

Famiglia 1 — Naive:
  1. Global Mean: per (store, product, hour), media S_obs su ore in-stock
  2. Conditional Mean: per (store, product, dow, hour), media S_obs su ore in-stock
     Fallback: hour → overall

Famiglia 2 — ML:
  3. LGB Imputer: LightGBM trainato su ore in-stock (base features, no lags)

Workflow:
  1. TRAIN:    Allena su gg 1-83 (ore in-stock, senza maschere MNAR)
  2. APPLY:    Predici D_hat alle posizioni mascherate MNAR val (gg 84-90, seed=42)
  3. EVALUATE: WAPE_recovery e WPE_recovery vs ground_truth dalle maschere
  4. COMPARE:  Classifica tutti i candidati

Metriche:
  WAPE_recovery = sum|D_hat - GT| / sum(GT)   (pooled e per-serie mediana)
  WPE_recovery  = sum(D_hat - GT) / sum(GT)   (pooled e per-serie mediana)

Output:
  notebooks/v2/results/imputation_{method}_val_per_series.parquet
  notebooks/v2/results/imputation_comparison.txt

Eseguire con: freshnet/bin/python notebooks/v2/05_fase_b_imputation.py
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import functools
import lightgbm as lgb

print = functools.partial(print, flush=True)

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Config ----
CONT_COLS = ['discount', 'avg_temperature', 'avg_humidity', 'precpt',
             'avg_wind_level', 'holiday_flag', 'activity_flag']

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
    'seed': 42,
}
MAX_BOOST_ROUNDS = 500
EARLY_STOPPING_ROUNDS = 30

print("=" * 72)
print("  FASE B — IMPUTATION CANDIDATES (Traccia A: demand recovery)")
print("=" * 72)

# =========================================================================
# 1. Caricamento dati + maschere MNAR
# =========================================================================
t0 = time.time()
print("\n1. Caricamento dati...")

df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_train_hf['dt_parsed'] = pd.to_datetime(df_train_hf['dt'])

all_dates_train = sorted(df_train_hf['dt_parsed'].unique())
date_to_day = {d: i + 1 for i, d in enumerate(all_dates_train)}
df_train_hf['day_num'] = df_train_hf['dt_parsed'].map(date_to_day)
df_train_hf['dow'] = df_train_hf['dt_parsed'].dt.dayofweek

print(f"  Train HF: {len(df_train_hf):,} righe, giorni 1-90")

# Load MNAR val masks
masks_val = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_val.parquet'))
print(f"  MNAR masks val: {len(masks_val):,} posizioni (gg 84-90, seed=42)")
print(f"  GT mean: {masks_val['ground_truth'].mean():.4f}")
print(f"  GT>0: {(masks_val['ground_truth'] > 0).mean()*100:.1f}%")

# Pre-parse hourly arrays for full train HF
print("  Parsing hourly arrays...")
sales_all = np.array(df_train_hf['hours_sale'].tolist(), dtype=np.float32)
stock_all = np.array(df_train_hf['hours_stock_status'].tolist(), dtype=np.float32)

print(f"  Tempo loading: {time.time()-t0:.1f}s")

# =========================================================================
# 2. Build series cache
# =========================================================================
print("\n2. Building series cache...")
t1 = time.time()

series_cache = {}
for (sid, pid), grp in df_train_hf.groupby(['store_id', 'product_id'], sort=False):
    grp_s = grp.sort_values('day_num')
    idx = grp_s.index.values
    series_cache[(sid, pid)] = {
        'days': grp_s['day_num'].values,
        'dows': grp_s['dow'].values,
        'sales': sales_all[idx],
        'stock': stock_all[idx],
        'city_id': grp_s['city_id'].values[0],
        'conts': grp_s[CONT_COLS].values.astype(np.float32),
    }

print(f"  {len(series_cache):,} serie in {time.time()-t1:.0f}s")

# =========================================================================
# 3. Build MNAR lookup: {(store_id, product_id, dt, hour): ground_truth}
# =========================================================================
print("\n3. Building MNAR lookup...")
t2 = time.time()

# Group masks by (store_id, product_id) for efficient per-series lookup
mnar_by_series = masks_val.groupby(['store_id', 'product_id'])
mnar_grouped = {}
for (sid, pid), grp in mnar_by_series:
    mnar_grouped[(sid, pid)] = grp[['dt', 'hour', 'ground_truth']].copy()

print(f"  {len(mnar_grouped):,} serie con maschere MNAR")
print(f"  Tempo: {time.time()-t2:.1f}s")


# =========================================================================
# 4. Evaluation function
# =========================================================================
def evaluate_imputer_on_mnar(predictions_fn, mnar_grouped, series_cache):
    """Evaluate an imputer on MNAR val positions.

    Args:
        predictions_fn: callable(sid, pid, dt, hour) -> D_hat
            or callable(sid, pid) -> dict {(dt, hour): D_hat}
        mnar_grouped: {(sid, pid): DataFrame with dt, hour, ground_truth}
        series_cache: series data dict

    Returns:
        pooled_metrics: dict with wape_recovery, wpe_recovery
        per_series_df: DataFrame with per-series metrics
    """
    # Pooled accumulators
    sum_abs_err = 0.0
    sum_err = 0.0
    sum_gt = 0.0
    n_total = 0

    ps_records = []

    for (sid, pid), mnar_df in mnar_grouped.items():
        sc = series_cache.get((sid, pid))
        if sc is None:
            continue

        gts = mnar_df['ground_truth'].values
        dts = mnar_df['dt'].values
        hours = mnar_df['hour'].values

        # Get predictions for this series
        preds = predictions_fn(sid, pid, dts, hours)

        # Per-series metrics
        errs = preds - gts
        abs_errs = np.abs(errs)
        abs_gt = np.abs(gts)
        sum_gt_s = abs_gt.sum()
        sum_err_s = errs.sum()
        sum_abs_err_s = abs_errs.sum()
        n_s = len(gts)

        # Pooled
        sum_abs_err += sum_abs_err_s
        sum_err += sum_err_s
        sum_gt += sum_gt_s
        n_total += n_s

        wape_s = sum_abs_err_s / sum_gt_s if sum_gt_s > 0 else np.nan
        wpe_s = sum_err_s / gts.sum() if gts.sum() != 0 else np.nan

        ps_records.append({
            'store_id': sid,
            'product_id': pid,
            'n_masked': n_s,
            'wape_recovery': wape_s,
            'wpe_recovery': wpe_s,
            'gt_sum': float(gts.sum()),
            'pred_sum': float(preds.sum()),
        })

    pooled = {
        'wape_recovery': sum_abs_err / sum_gt if sum_gt > 0 else np.nan,
        'wpe_recovery': sum_err / sum_gt if sum_gt > 0 else np.nan,
        'n_positions': n_total,
    }

    return pooled, pd.DataFrame(ps_records)


# =========================================================================
# 5. Imputer 1: Global Mean (per store, product, hour)
# =========================================================================
print("\n" + "=" * 72)
print("  IMPUTER 1: GLOBAL MEAN (per series, hour)")
print("=" * 72)
t3 = time.time()

# Compute profiles: mean S_obs for in-stock hours of days 1-83
global_profiles = {}  # {(sid, pid): (24,) array}
for (sid, pid), sc in series_cache.items():
    days = sc['days']
    sales = sc['sales']
    stock = sc['stock']
    train_mask = days <= 83
    if not train_mask.any():
        global_profiles[(sid, pid)] = np.zeros(24, dtype=np.float32)
        continue

    train_sales = sales[train_mask]
    train_stock = stock[train_mask]
    instock_mask = train_stock == 0

    profile = np.zeros(24, dtype=np.float64)
    for h in range(24):
        vals = train_sales[:, h][instock_mask[:, h]]
        profile[h] = vals.mean() if len(vals) > 0 else 0.0

    global_profiles[(sid, pid)] = profile.astype(np.float32)


def predict_global_mean(sid, pid, dts, hours):
    prof = global_profiles.get((sid, pid), np.zeros(24, dtype=np.float32))
    return prof[hours]


pooled_gm, ps_gm = evaluate_imputer_on_mnar(predict_global_mean, mnar_grouped, series_cache)
ps_gm.to_parquet(os.path.join(RESULTS_DIR, 'imputation_global_mean_val_per_series.parquet'),
                  index=False)
print(f"  WAPE_recovery pooled: {pooled_gm['wape_recovery']:.4f}")
print(f"  WPE_recovery pooled:  {pooled_gm['wpe_recovery']:.4f}")
print(f"  WAPE_recovery median: {ps_gm['wape_recovery'].median():.4f}")
print(f"  WPE_recovery median:  {ps_gm['wpe_recovery'].median():.4f}")
print(f"  Tempo: {time.time()-t3:.1f}s")


# =========================================================================
# 6. Imputer 2: Conditional Mean (per series, dow, hour)
# =========================================================================
print("\n" + "=" * 72)
print("  IMPUTER 2: CONDITIONAL MEAN (per series, dow, hour)")
print("=" * 72)
t4 = time.time()

# Compute profiles: mean S_obs for in-stock hours of days 1-83, per dow
cond_profiles = {}  # {(sid, pid): (7, 24) array}
for (sid, pid), sc in series_cache.items():
    days = sc['days']
    dows = sc['dows']
    sales = sc['sales']
    stock = sc['stock']
    train_mask = days <= 83

    if not train_mask.any():
        cond_profiles[(sid, pid)] = np.zeros((7, 24), dtype=np.float32)
        continue

    train_sales = sales[train_mask]
    train_stock = stock[train_mask]
    train_dows = dows[train_mask]

    # per (dow, hour) profile
    dow_hour_sum = np.zeros((7, 24), dtype=np.float64)
    dow_hour_cnt = np.zeros((7, 24), dtype=np.int32)

    for di in range(len(train_sales)):
        d = train_dows[di]
        for h in range(24):
            if train_stock[di, h] == 0:
                dow_hour_sum[d, h] += train_sales[di, h]
                dow_hour_cnt[d, h] += 1

    # Fallback: hour-level mean
    hour_sum = dow_hour_sum.sum(axis=0)
    hour_cnt = dow_hour_cnt.sum(axis=0)

    profile = np.zeros((7, 24), dtype=np.float32)
    for d in range(7):
        for h in range(24):
            if dow_hour_cnt[d, h] >= 3:
                profile[d, h] = dow_hour_sum[d, h] / dow_hour_cnt[d, h]
            elif hour_cnt[h] >= 1:
                profile[d, h] = hour_sum[h] / hour_cnt[h]
            # else: 0

    cond_profiles[(sid, pid)] = profile

# Build dt → dow lookup from training data
dt_to_dow = {}
for _, row in df_train_hf[['dt', 'dow']].drop_duplicates().iterrows():
    dt_to_dow[row['dt']] = row['dow']


def predict_cond_mean(sid, pid, dts, hours):
    prof = cond_profiles.get((sid, pid), np.zeros((7, 24), dtype=np.float32))
    preds = np.zeros(len(dts), dtype=np.float32)
    for i in range(len(dts)):
        dow = dt_to_dow.get(dts[i], 0)
        preds[i] = prof[dow, hours[i]]
    return preds


pooled_cm, ps_cm = evaluate_imputer_on_mnar(predict_cond_mean, mnar_grouped, series_cache)
ps_cm.to_parquet(os.path.join(RESULTS_DIR, 'imputation_cond_mean_val_per_series.parquet'),
                  index=False)
print(f"  WAPE_recovery pooled: {pooled_cm['wape_recovery']:.4f}")
print(f"  WPE_recovery pooled:  {pooled_cm['wpe_recovery']:.4f}")
print(f"  WAPE_recovery median: {ps_cm['wape_recovery'].median():.4f}")
print(f"  WPE_recovery median:  {ps_cm['wpe_recovery'].median():.4f}")
print(f"  Tempo: {time.time()-t4:.1f}s")


# =========================================================================
# 7. Imputer 3: LGB Imputer (trained on in-stock hours)
# =========================================================================
print("\n" + "=" * 72)
print("  IMPUTER 3: LGB IMPUTER (trained on in-stock hours)")
print("=" * 72)

# 7a. Build training dataset: in-stock hours of days 1-83
print("\n  7a. Building training dataset (in-stock hours, gg 1-83)...")
t5 = time.time()

df_train_sub = df_train_hf[(df_train_hf['day_num'] >= 1) & (df_train_hf['day_num'] <= 83)].copy()
N_train = len(df_train_sub)

# Expand to hourly
store_ids_h = np.repeat(df_train_sub['store_id'].values, 24)
product_ids_h = np.repeat(df_train_sub['product_id'].values, 24)
city_ids_h = np.repeat(df_train_sub['city_id'].values, 24)
dows_h = np.repeat(df_train_sub['dow'].values, 24)
hour_h = np.tile(np.arange(24, dtype=np.int32), N_train)
conts_h = np.repeat(df_train_sub[CONT_COLS].values.astype(np.float32), 24, axis=0)

idx_train = df_train_sub.index.values
sales_h = sales_all[idx_train].ravel()
stock_h = stock_all[idx_train].ravel()

# Filter to in-stock only
instock = stock_h == 0
print(f"    Total hourly slots: {len(sales_h):,}")
print(f"    In-stock slots: {instock.sum():,} ({instock.mean()*100:.1f}%)")

X_lgb_train = pd.DataFrame({
    'store_id': store_ids_h[instock],
    'product_id': product_ids_h[instock],
    'city_id': city_ids_h[instock],
    'dow': dows_h[instock],
    'hour': hour_h[instock],
})
for j, col in enumerate(CONT_COLS):
    X_lgb_train[col] = conts_h[instock, j]

for col in ['store_id', 'product_id', 'city_id', 'dow', 'hour']:
    X_lgb_train[col] = X_lgb_train[col].astype('category')

y_lgb_train = sales_h[instock]
print(f"    Training rows: {len(X_lgb_train):,}")
print(f"    Build time: {time.time()-t5:.0f}s")

# 7b. Build val dataset for early stopping: in-stock hours of days 84-90
print("\n  7b. Building val dataset (in-stock hours, gg 84-90)...")
df_val_sub = df_train_hf[(df_train_hf['day_num'] >= 84) & (df_train_hf['day_num'] <= 90)].copy()
N_val = len(df_val_sub)

store_ids_v = np.repeat(df_val_sub['store_id'].values, 24)
product_ids_v = np.repeat(df_val_sub['product_id'].values, 24)
city_ids_v = np.repeat(df_val_sub['city_id'].values, 24)
dows_v = np.repeat(df_val_sub['dow'].values, 24)
hour_v = np.tile(np.arange(24, dtype=np.int32), N_val)
conts_v = np.repeat(df_val_sub[CONT_COLS].values.astype(np.float32), 24, axis=0)

idx_val = df_val_sub.index.values
sales_v = sales_all[idx_val].ravel()
stock_v = stock_all[idx_val].ravel()

instock_v = stock_v == 0
X_lgb_val = pd.DataFrame({
    'store_id': store_ids_v[instock_v],
    'product_id': product_ids_v[instock_v],
    'city_id': city_ids_v[instock_v],
    'dow': dows_v[instock_v],
    'hour': hour_v[instock_v],
})
for j, col in enumerate(CONT_COLS):
    X_lgb_val[col] = conts_v[instock_v, j]

for col in ['store_id', 'product_id', 'city_id', 'dow', 'hour']:
    X_lgb_val[col] = X_lgb_val[col].astype('category')

y_lgb_val = sales_v[instock_v]
print(f"    Val rows (in-stock): {len(X_lgb_val):,}")

# 7c. Train LGB imputer
print("\n  7c. Training LGB imputer...")
t6 = time.time()

lgb_train_ds = lgb.Dataset(X_lgb_train, y_lgb_train, free_raw_data=True)
lgb_val_ds = lgb.Dataset(X_lgb_val, y_lgb_val, reference=lgb_train_ds, free_raw_data=True)

callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(50)]

lgb_model = lgb.train(
    LGB_PARAMS, lgb_train_ds,
    num_boost_round=MAX_BOOST_ROUNDS,
    valid_sets=[lgb_val_ds],
    valid_names=['val'],
    callbacks=callbacks,
)

best_iter = lgb_model.best_iteration
best_mae = lgb_model.best_score['val']['l1']
print(f"    Best iteration: {best_iter}, Val MAE: {best_mae:.6f}")
print(f"    Training time: {time.time()-t6:.0f}s")

lgb_model.save_model(os.path.join(RESULTS_DIR, 'lgb_imputer.txt'))

# 7d. Predict at MNAR positions
print("\n  7d. Predicting at MNAR val positions...")
t7 = time.time()

# Build feature DataFrame for ALL MNAR positions
mnar_features = masks_val.copy()
mnar_features['dt_parsed'] = pd.to_datetime(mnar_features['dt'])
mnar_features['day_num'] = mnar_features['dt_parsed'].map(date_to_day)
mnar_features['dow'] = mnar_features['dt_parsed'].dt.dayofweek

# Need city_id and continuous features for each (store_id, product_id, dt)
# Merge from df_train_hf
mnar_features = mnar_features.merge(
    df_train_hf[['store_id', 'product_id', 'dt', 'city_id'] + CONT_COLS],
    on=['store_id', 'product_id', 'dt'],
    how='left'
)

X_mnar = pd.DataFrame({
    'store_id': mnar_features['store_id'].values,
    'product_id': mnar_features['product_id'].values,
    'city_id': mnar_features['city_id'].values,
    'dow': mnar_features['dow'].values,
    'hour': mnar_features['hour'].values,
})
for col in CONT_COLS:
    X_mnar[col] = mnar_features[col].values

for col in ['store_id', 'product_id', 'city_id', 'dow', 'hour']:
    X_mnar[col] = X_mnar[col].astype('category')

lgb_preds_mnar = np.clip(lgb_model.predict(X_mnar), 0, None).astype(np.float32)
print(f"    Predicted {len(lgb_preds_mnar):,} positions")
print(f"    Pred mean: {lgb_preds_mnar.mean():.4f}")
print(f"    GT mean: {masks_val['ground_truth'].mean():.4f}")
print(f"    Tempo: {time.time()-t7:.1f}s")

# 7e. Evaluate LGB imputer
# Store predictions indexed by (store_id, product_id, dt, hour) for the evaluate function
lgb_pred_series = {}
mnar_features['lgb_pred'] = lgb_preds_mnar
for (sid, pid), grp in mnar_features.groupby(['store_id', 'product_id']):
    lgb_pred_series[(sid, pid)] = dict(zip(
        zip(grp['dt'].values, grp['hour'].values),
        grp['lgb_pred'].values
    ))


def predict_lgb(sid, pid, dts, hours):
    lookup = lgb_pred_series.get((sid, pid), {})
    preds = np.zeros(len(dts), dtype=np.float32)
    for i in range(len(dts)):
        preds[i] = lookup.get((dts[i], hours[i]), 0.0)
    return preds


pooled_lgb, ps_lgb = evaluate_imputer_on_mnar(predict_lgb, mnar_grouped, series_cache)
ps_lgb.to_parquet(os.path.join(RESULTS_DIR, 'imputation_lgb_val_per_series.parquet'),
                   index=False)
print(f"\n  WAPE_recovery pooled: {pooled_lgb['wape_recovery']:.4f}")
print(f"  WPE_recovery pooled:  {pooled_lgb['wpe_recovery']:.4f}")
print(f"  WAPE_recovery median: {ps_lgb['wape_recovery'].median():.4f}")
print(f"  WPE_recovery median:  {ps_lgb['wpe_recovery'].median():.4f}")


# =========================================================================
# 8. Confronto candidati
# =========================================================================
print("\n" + "=" * 72)
print("  8. CONFRONTO CANDIDATI IMPUTATION (val MNAR, gg 84-90)")
print("=" * 72)

results_all = {
    'Global Mean': (pooled_gm, ps_gm),
    'Cond Mean': (pooled_cm, ps_cm),
    'LGB Imputer': (pooled_lgb, ps_lgb),
}

print(f"\n  {'Metodo':<20} {'WAPE_rec_pool':>14} {'WAPE_rec_med':>14} "
      f"{'WPE_rec_pool':>14} {'WPE_rec_med':>14}")
print("  " + "-" * 80)

for label, (pooled, ps) in results_all.items():
    print(f"  {label:<20} {pooled['wape_recovery']:>14.4f} "
          f"{ps['wape_recovery'].median():>14.4f} "
          f"{pooled['wpe_recovery']:>14.4f} "
          f"{ps['wpe_recovery'].median():>14.4f}")

# Select best
best_method = min(results_all.keys(),
                  key=lambda k: results_all[k][0]['wape_recovery'])
print(f"\n  Miglior metodo (WAPE_recovery pooled): {best_method}")
print(f"  WAPE_recovery = {results_all[best_method][0]['wape_recovery']:.4f}")

print(f"\n  Tempo totale: {time.time()-t0:.0f}s")
print("=" * 72)
