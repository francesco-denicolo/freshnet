"""
06_fase_b_select_and_complete.py — Fase B Step 0.4: Select Winner & Produce completed_sales
============================================================================================
Piano: CLAUDE_SEQUENTIAL-2.md, Fase B, punto (3) finale + (4) prep

Workflow:
  1. RETRAIN:   Riallena LGB Imputer (vincitore) su TUTTI i 90 gg (in-stock hours)
  2. IMPUTE:    Predici D_hat per TUTTE le ore di stockout → completed_sales
  3. MNAR TEST: Valuta recovery su maschere MNAR test (seed=123, gg 1-90)
  4. SAVE:      Salva data/completed_sales.parquet per Fase B punti (4)-(5)

completed_sales:
  - Dove in-stock: completed_sale = S_obs (vendite osservate)
  - Dove stockout: completed_sale = max(0, lgb.predict(features))

Eseguire con: freshnet/bin/python notebooks/v2/06_fase_b_select_and_complete.py
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
# Best iteration from Step 0.3 (val on gg 84-90 in-stock)
BEST_ITER = 500

print("=" * 72)
print("  FASE B — STEP 0.4: SELECT WINNER & PRODUCE completed_sales")
print("=" * 72)
print(f"  Winner: LGB Imputer (WAPE_rec pool=0.9820, best_iter={BEST_ITER})")
print(f"  Retrain su gg 1-90, poi imputa TUTTE le ore stockout")

# =========================================================================
# 1. Caricamento dati
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

# Parse hourly arrays
print("  Parsing hourly arrays...")
sales_all = np.array(df_train_hf['hours_sale'].tolist(), dtype=np.float32)
stock_all = np.array(df_train_hf['hours_stock_status'].tolist(), dtype=np.float32)

print(f"  Tempo loading: {time.time()-t0:.1f}s")

# =========================================================================
# 2. Retrain LGB Imputer su TUTTI i 90 giorni (in-stock hours)
# =========================================================================
print("\n2. Retrain LGB Imputer su gg 1-90 (in-stock hours)...")
t1 = time.time()

N_all = len(df_train_hf)

# Expand to hourly
store_ids_h = np.repeat(df_train_hf['store_id'].values, 24)
product_ids_h = np.repeat(df_train_hf['product_id'].values, 24)
city_ids_h = np.repeat(df_train_hf['city_id'].values, 24)
dows_h = np.repeat(df_train_hf['dow'].values, 24)
hour_h = np.tile(np.arange(24, dtype=np.int32), N_all)
conts_h = np.repeat(df_train_hf[CONT_COLS].values.astype(np.float32), 24, axis=0)

sales_h = sales_all.ravel()
stock_h = stock_all.ravel()

# Filter to in-stock only
instock = stock_h == 0
print(f"  Total hourly slots: {len(sales_h):,}")
print(f"  In-stock slots: {instock.sum():,} ({instock.mean()*100:.1f}%)")
print(f"  Stockout slots: {(~instock).sum():,} ({(~instock).mean()*100:.1f}%)")

X_lgb = pd.DataFrame({
    'store_id': store_ids_h[instock],
    'product_id': product_ids_h[instock],
    'city_id': city_ids_h[instock],
    'dow': dows_h[instock],
    'hour': hour_h[instock],
})
for j, col in enumerate(CONT_COLS):
    X_lgb[col] = conts_h[instock, j]

for col in ['store_id', 'product_id', 'city_id', 'dow', 'hour']:
    X_lgb[col] = X_lgb[col].astype('category')

y_lgb = sales_h[instock]
print(f"  Training rows: {len(X_lgb):,}")

lgb_ds = lgb.Dataset(X_lgb, y_lgb, free_raw_data=True)

print(f"  Training {BEST_ITER} rounds (no early stopping)...")
t2 = time.time()
lgb_model = lgb.train(
    LGB_PARAMS, lgb_ds,
    num_boost_round=BEST_ITER,
    callbacks=[lgb.log_evaluation(100)],
)
print(f"  Training time: {time.time()-t2:.0f}s")

lgb_model.save_model(os.path.join(RESULTS_DIR, 'lgb_imputer_retrained.txt'))
print(f"  Modello salvato: {os.path.join(RESULTS_DIR, 'lgb_imputer_retrained.txt')}")

# =========================================================================
# 3. Impute ALL stockout hours → completed_sales
# =========================================================================
print("\n3. Imputing stockout hours...")
t3 = time.time()

stockout = stock_h == 1
n_stockout = stockout.sum()
print(f"  Stockout slots to impute: {n_stockout:,}")

# Build features for stockout slots
X_stockout = pd.DataFrame({
    'store_id': store_ids_h[stockout],
    'product_id': product_ids_h[stockout],
    'city_id': city_ids_h[stockout],
    'dow': dows_h[stockout],
    'hour': hour_h[stockout],
})
for j, col in enumerate(CONT_COLS):
    X_stockout[col] = conts_h[stockout, j]

for col in ['store_id', 'product_id', 'city_id', 'dow', 'hour']:
    X_stockout[col] = X_stockout[col].astype('category')

# Predict
d_hat = np.clip(lgb_model.predict(X_stockout), 0, None).astype(np.float32)
print(f"  D_hat mean: {d_hat.mean():.4f}")
print(f"  D_hat median: {np.median(d_hat):.4f}")
print(f"  D_hat > 0: {(d_hat > 0).mean()*100:.1f}%")
print(f"  D_hat max: {d_hat.max():.4f}")
print(f"  Tempo predict: {time.time()-t3:.1f}s")

# Build completed_sales array (same shape as sales_all, (N_all, 24))
completed_sales = sales_all.copy()
stock_flat = stock_all.ravel()
stockout_flat = stock_flat == 1
completed_flat = completed_sales.ravel()
completed_flat[stockout_flat] = d_hat
completed_sales = completed_flat.reshape(sales_all.shape)

print(f"\n  Completed_sales stats:")
print(f"    Mean (all):      {completed_sales.mean():.4f}")
print(f"    Mean (in-stock): {sales_all[stock_all == 0].mean():.4f}")
print(f"    Mean (stockout): {completed_sales[stock_all == 1].mean():.4f}")
print(f"    S_obs mean:      {sales_all.mean():.4f}")

# =========================================================================
# 4. Save completed_sales.parquet
# =========================================================================
print("\n4. Saving completed_sales.parquet...")
t4 = time.time()

# Build DataFrame with completed sales as hourly arrays
completed_df = df_train_hf[['store_id', 'product_id', 'dt']].copy()
completed_df['hours_completed_sale'] = list(completed_sales)
completed_df['hours_sale_original'] = list(sales_all)
completed_df['hours_stock_status'] = list(stock_all)

out_path = os.path.join(DATA_DIR, 'completed_sales.parquet')
completed_df.to_parquet(out_path, index=False, engine='pyarrow')
size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"  Salvato: {out_path}")
print(f"  Size: {size_mb:.1f} MB")
print(f"  Righe: {len(completed_df):,}")
print(f"  Tempo: {time.time()-t4:.1f}s")

# =========================================================================
# 5. Evaluate on MNAR TEST masks (seed=123, gg 1-90)
# =========================================================================
print("\n5. Evaluation su maschere MNAR TEST (seed=123, gg 1-90)...")
t5 = time.time()

masks_test = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_test.parquet'))
print(f"  MNAR masks test: {len(masks_test):,} posizioni")
print(f"  GT mean: {masks_test['ground_truth'].mean():.4f}")
print(f"  GT>0: {(masks_test['ground_truth'] > 0).mean()*100:.1f}%")

# Build features for MNAR test positions
masks_test['dt_parsed'] = pd.to_datetime(masks_test['dt'])
masks_test['day_num'] = masks_test['dt_parsed'].map(date_to_day)
masks_test['dow'] = masks_test['dt_parsed'].dt.dayofweek

# Merge features from df_train_hf
masks_test = masks_test.merge(
    df_train_hf[['store_id', 'product_id', 'dt', 'city_id'] + CONT_COLS],
    on=['store_id', 'product_id', 'dt'],
    how='left'
)

X_mnar_test = pd.DataFrame({
    'store_id': masks_test['store_id'].values,
    'product_id': masks_test['product_id'].values,
    'city_id': masks_test['city_id'].values,
    'dow': masks_test['dow'].values,
    'hour': masks_test['hour'].values,
})
for col in CONT_COLS:
    X_mnar_test[col] = masks_test[col].values

for col in ['store_id', 'product_id', 'city_id', 'dow', 'hour']:
    X_mnar_test[col] = X_mnar_test[col].astype('category')

# Predict
preds_mnar_test = np.clip(lgb_model.predict(X_mnar_test), 0, None).astype(np.float32)
gt_test = masks_test['ground_truth'].values.astype(np.float32)

# Pooled metrics
abs_err = np.abs(preds_mnar_test - gt_test)
err = preds_mnar_test - gt_test
sum_gt = np.abs(gt_test).sum()

wape_pool = abs_err.sum() / sum_gt if sum_gt > 0 else np.nan
wpe_pool = err.sum() / gt_test.sum() if gt_test.sum() > 0 else np.nan

print(f"\n  POOLED metrics:")
print(f"    WAPE_recovery: {wape_pool:.4f}")
print(f"    WPE_recovery:  {wpe_pool:.4f}")
print(f"    Pred mean: {preds_mnar_test.mean():.4f}")
print(f"    GT mean:   {gt_test.mean():.4f}")

# Per-series metrics
masks_test['pred'] = preds_mnar_test
ps_records = []
for (sid, pid), grp in masks_test.groupby(['store_id', 'product_id']):
    gts = grp['ground_truth'].values
    preds = grp['pred'].values

    s_ae = np.abs(preds - gts).sum()
    s_e = (preds - gts).sum()
    s_gt = np.abs(gts).sum()

    wape_s = s_ae / s_gt if s_gt > 0 else np.nan
    wpe_s = s_e / gts.sum() if gts.sum() > 0 else np.nan

    ps_records.append({
        'store_id': sid,
        'product_id': pid,
        'n_masked': len(gts),
        'wape_recovery': wape_s,
        'wpe_recovery': wpe_s,
        'gt_sum': float(gts.sum()),
        'pred_sum': float(preds.sum()),
    })

ps_test = pd.DataFrame(ps_records)
ps_test.to_parquet(os.path.join(RESULTS_DIR, 'imputation_lgb_test_per_series.parquet'),
                    index=False)

wape_med = ps_test['wape_recovery'].median()
wpe_med = ps_test['wpe_recovery'].median()

print(f"\n  PER-SERIES MEDIAN metrics:")
print(f"    WAPE_recovery: {wape_med:.4f}")
print(f"    WPE_recovery:  {wpe_med:.4f}")
print(f"  Tempo: {time.time()-t5:.1f}s")

# =========================================================================
# 6. Also evaluate naive imputers on MNAR TEST (for comparison table)
# =========================================================================
print("\n6. Naive imputers on MNAR TEST (for comparison)...")
t6 = time.time()

# Build series lookup for all 90 days
series_cache = {}
for (sid, pid), grp in df_train_hf.groupby(['store_id', 'product_id'], sort=False):
    grp_s = grp.sort_values('day_num')
    idx = grp_s.index.values
    series_cache[(sid, pid)] = {
        'days': grp_s['day_num'].values,
        'dows': grp_s['dow'].values,
        'sales': sales_all[idx],
        'stock': stock_all[idx],
    }

# Global Mean: per (store, product, hour), mean S_obs on in-stock of ALL 90 days
global_profiles = {}
for (sid, pid), sc in series_cache.items():
    train_sales = sc['sales']
    train_stock = sc['stock']
    instock_mask = train_stock == 0

    profile = np.zeros(24, dtype=np.float64)
    for h in range(24):
        vals = train_sales[:, h][instock_mask[:, h]]
        profile[h] = vals.mean() if len(vals) > 0 else 0.0
    global_profiles[(sid, pid)] = profile.astype(np.float32)

# Conditional Mean: per (store, product, dow, hour), fallback to hour
cond_profiles = {}
for (sid, pid), sc in series_cache.items():
    train_sales = sc['sales']
    train_stock = sc['stock']
    train_dows = sc['dows']

    dow_hour_sum = np.zeros((7, 24), dtype=np.float64)
    dow_hour_cnt = np.zeros((7, 24), dtype=np.int32)

    for di in range(len(train_sales)):
        d = train_dows[di]
        for h in range(24):
            if train_stock[di, h] == 0:
                dow_hour_sum[d, h] += train_sales[di, h]
                dow_hour_cnt[d, h] += 1

    hour_sum = dow_hour_sum.sum(axis=0)
    hour_cnt = dow_hour_cnt.sum(axis=0)

    profile = np.zeros((7, 24), dtype=np.float32)
    for d in range(7):
        for h in range(24):
            if dow_hour_cnt[d, h] >= 3:
                profile[d, h] = dow_hour_sum[d, h] / dow_hour_cnt[d, h]
            elif hour_cnt[h] >= 1:
                profile[d, h] = hour_sum[h] / hour_cnt[h]
    cond_profiles[(sid, pid)] = profile

dt_to_dow = {}
for _, row in df_train_hf[['dt', 'dow']].drop_duplicates().iterrows():
    dt_to_dow[row['dt']] = row['dow']


def eval_naive_on_test(predict_fn, label):
    """Evaluate a naive imputer on MNAR test masks."""
    sum_ae, sum_e, sum_gt_p = 0.0, 0.0, 0.0
    ps_recs = []

    mnar_by_series = masks_test.groupby(['store_id', 'product_id'])
    for (sid, pid), grp in mnar_by_series:
        gts = grp['ground_truth'].values
        dts = grp['dt'].values
        hours = grp['hour'].values

        preds = predict_fn(sid, pid, dts, hours)

        ae = np.abs(preds - gts).sum()
        e = (preds - gts).sum()
        gt_s = np.abs(gts).sum()

        sum_ae += ae
        sum_e += e
        sum_gt_p += gt_s

        wape_s = ae / gt_s if gt_s > 0 else np.nan
        wpe_s = e / gts.sum() if gts.sum() > 0 else np.nan
        ps_recs.append({
            'store_id': sid, 'product_id': pid,
            'wape_recovery': wape_s, 'wpe_recovery': wpe_s,
        })

    ps_df = pd.DataFrame(ps_recs)
    wape_p = sum_ae / sum_gt_p if sum_gt_p > 0 else np.nan
    wpe_p = sum_e / sum_gt_p if sum_gt_p > 0 else np.nan

    print(f"  {label:<20} pool: WAPE={wape_p:.4f}, WPE={wpe_p:.4f}  "
          f"med: WAPE={ps_df['wape_recovery'].median():.4f}, "
          f"WPE={ps_df['wpe_recovery'].median():.4f}")

    return {'wape_pool': wape_p, 'wpe_pool': wpe_p,
            'wape_med': ps_df['wape_recovery'].median(),
            'wpe_med': ps_df['wpe_recovery'].median()}


def predict_gm(sid, pid, dts, hours):
    prof = global_profiles.get((sid, pid), np.zeros(24, dtype=np.float32))
    return prof[hours]


def predict_cm(sid, pid, dts, hours):
    prof = cond_profiles.get((sid, pid), np.zeros((7, 24), dtype=np.float32))
    preds = np.zeros(len(dts), dtype=np.float32)
    for i in range(len(dts)):
        dow = dt_to_dow.get(dts[i], 0)
        preds[i] = prof[dow, hours[i]]
    return preds


res_gm = eval_naive_on_test(predict_gm, "Global Mean")
res_cm = eval_naive_on_test(predict_cm, "Cond Mean")

print(f"  Tempo: {time.time()-t6:.1f}s")

# =========================================================================
# 7. Final comparison table
# =========================================================================
print("\n" + "=" * 72)
print("  7. CONFRONTO FINALE IMPUTATION (test MNAR, seed=123, gg 1-90)")
print("=" * 72)

all_results = {
    'Global Mean': res_gm,
    'Cond Mean': res_cm,
    'LGB Imputer': {
        'wape_pool': wape_pool, 'wpe_pool': wpe_pool,
        'wape_med': wape_med, 'wpe_med': wpe_med,
    },
}

print(f"\n  {'Metodo':<20} {'WAPE_pool':>12} {'WAPE_med':>12} "
      f"{'WPE_pool':>12} {'WPE_med':>12}")
print("  " + "-" * 72)

for label, r in all_results.items():
    print(f"  {label:<20} {r['wape_pool']:>12.4f} {r['wape_med']:>12.4f} "
          f"{r['wpe_pool']:>12.4f} {r['wpe_med']:>12.4f}")

best_label = min(all_results.keys(), key=lambda k: all_results[k]['wape_pool'])
print(f"\n  Vincitore (WAPE_recovery pooled): {best_label}")

# =========================================================================
# 8. Summary
# =========================================================================
print("\n" + "=" * 72)
print("  RIEPILOGO")
print("=" * 72)
print(f"  Vincitore imputation:   LGB Imputer")
print(f"  Retrained su:           gg 1-90, {BEST_ITER} rounds")
print(f"  completed_sales:        {out_path}")
print(f"  Size:                   {size_mb:.1f} MB")
print(f"  Test MNAR WAPE (pool):  {wape_pool:.4f}")
print(f"  Test MNAR WPE (pool):   {wpe_pool:.4f}")
print(f"  Test MNAR WAPE (med):   {wape_med:.4f}")
print(f"  Test MNAR WPE (med):    {wpe_med:.4f}")
print(f"\n  Tempo totale: {time.time()-t0:.0f}s")
print("=" * 72)
