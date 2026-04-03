"""
03_fase_a_lgb.py — Fase A: LightGBM Forecasting su dati sporchi
================================================================
Piano: CLAUDE_SEQUENTIAL-2.md, Fase A, punto (2)

LightGBM su dati sporchi (S_obs con stockout), due varianti:
  A: solo base features (store, product, city, dow, hour, 7 cont)
  F: base + 11 M5-style lag features (raw lags, rolling stats, DoW, momentum)

Workflow:
  1. TUNING:     Train gg 2-83, val gg 84-90 → best_iteration per variante
  2. SELECT:     Scelta miglior variante (pooled WAPE)
  3. RETRAINING: Retrain su gg 2-90 con best_iteration (no early stopping)
  4. TEST:       Eval su gg 91-97 (eval HF)

Output:
  notebooks/v2/results/lgb_{a|f}_{val|test}_per_series.parquet
  notebooks/v2/results/lgb_best.txt

Eseguire con: freshnet/bin/python notebooks/v2/03_fase_a_lgb.py
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

from src.evaluation.metrics import compute_metrics

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
print("  FASE A — LightGBM FORECASTING (dati sporchi)")
print("=" * 72)

# =========================================================================
# 1. Caricamento dati
# =========================================================================
t0 = time.time()
print("\n1. Caricamento dati...")
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
print(f"  Train: {len(df_train):,} righe, Eval: {len(df_eval):,} righe")
print(f"  Full: {len(df_full):,} righe, {len(all_dates)} giorni, {n_series:,} serie")
print(f"  Tempo loading: {time.time()-t0:.1f}s")

del df_train, df_eval

# =========================================================================
# 2. Build series cache (for lag computation)
# =========================================================================
print("\n2. Building series cache...")
t1 = time.time()

series_cache = {}
for (sid, pid), grp in df_full.groupby(['store_id', 'product_id'], sort=False):
    grp_s = grp.sort_values('day_num')
    series_cache[(sid, pid)] = {
        'days': grp_s['day_num'].values,
        'dows': grp_s['dow'].values,
        'sales': np.array(grp_s['hours_sale'].tolist(), dtype=np.float32),
        'stock': np.array(grp_s['hours_stock_status'].tolist(), dtype=np.float32),
    }

print(f"  {len(series_cache):,} serie cached in {time.time()-t1:.0f}s")


# =========================================================================
# 3. Build hourly dataset function
# =========================================================================
def build_hourly_dataset(df_subset, anchor_mode='rolling', anchor_day=None,
                         with_lags=False):
    """Build flat per-hour dataset for LightGBM.

    Args:
        df_subset: DataFrame filtered to desired day range
        anchor_mode: 'rolling' for train (anchor=d-1), 'fixed' for val/test
        anchor_day: fixed anchor day (for val/test)
        with_lags: whether to compute M5-style lag features

    Returns:
        X: pd.DataFrame with features
        y: np.array targets (hourly sales)
        stock: np.array stock status (hourly)
        sids: store_id per hourly row
        pids: product_id per hourly row
    """
    N = len(df_subset)

    # Day-level arrays
    store_ids = df_subset['store_id'].values
    product_ids = df_subset['product_id'].values
    city_ids = df_subset['city_id'].values
    dows = df_subset['dow'].values
    day_nums = df_subset['day_num'].values
    conts = df_subset[CONT_COLS].values.astype(np.float32)
    sales_arr = np.array(df_subset['hours_sale'].tolist(), dtype=np.float32)  # (N, 24)
    stock_arr = np.array(df_subset['hours_stock_status'].tolist(), dtype=np.float32)

    N_hourly = N * 24

    # Expand to hourly
    sids_h = np.repeat(store_ids, 24)
    pids_h = np.repeat(product_ids, 24)
    city_h = np.repeat(city_ids, 24)
    dow_h = np.repeat(dows, 24)
    hour_h = np.tile(np.arange(24, dtype=np.int32), N)
    conts_h = np.repeat(conts, 24, axis=0)

    y = sales_arr.ravel()
    stock_flat = stock_arr.ravel()

    # Build feature DataFrame
    X = pd.DataFrame({
        'store_id': sids_h,
        'product_id': pids_h,
        'city_id': city_h,
        'dow': dow_h,
        'hour': hour_h,
    })
    for j, col in enumerate(CONT_COLS):
        X[col] = conts_h[:, j]

    # Categorical types
    for col in ['store_id', 'product_id', 'city_id', 'dow', 'hour']:
        X[col] = X[col].astype('category')

    if with_lags:
        lag_names = ['lag_1d', 'lag_7d', 'lag_14d', 'rmean_7d', 'rmean_14d',
                     'rstd_7d', 'lag_dow', 'rmean_dow', 'daily_total_lag1',
                     'daily_total_rmean7', 'momentum_1d_7d']
        lag_arrays = {name: np.full(N_hourly, np.nan, dtype=np.float32)
                      for name in lag_names}

        for i in range(N):
            sid, pid = store_ids[i], product_ids[i]
            d = day_nums[i]
            dow_i = dows[i]
            hs, he = i * 24, (i + 1) * 24

            if anchor_mode == 'rolling':
                anc = d - 1
            else:
                anc = anchor_day

            sc = series_cache.get((sid, pid))
            if sc is None:
                continue

            s_days = sc['days']
            s_sales = sc['sales']

            hist_mask = s_days <= anc
            if not hist_mask.any():
                continue

            avail_days = s_days[hist_mask]
            avail_sales = s_sales[hist_mask]
            K = len(avail_days)

            # lag_1d
            lag_arrays['lag_1d'][hs:he] = avail_sales[-1]

            # lag_7d
            if K >= 7:
                lag_arrays['lag_7d'][hs:he] = avail_sales[-7]

            # lag_14d
            if K >= 14:
                lag_arrays['lag_14d'][hs:he] = avail_sales[-14]

            # rmean_7d
            n7 = min(7, K)
            lag_arrays['rmean_7d'][hs:he] = avail_sales[-n7:].mean(axis=0)
            if K >= 7:
                lag_arrays['rmean_14d'][hs:he] = avail_sales[-min(14, K):].mean(axis=0)

            # rstd_7d
            if K >= 2:
                lag_arrays['rstd_7d'][hs:he] = avail_sales[-n7:].std(axis=0)

            # DoW-specific
            dow_mask = hist_mask & (sc['dows'] == dow_i)
            if dow_mask.any():
                dow_sales = s_sales[dow_mask]
                lag_arrays['lag_dow'][hs:he] = dow_sales[-1]
                lag_arrays['rmean_dow'][hs:he] = dow_sales.mean(axis=0)

            # Daily aggregates
            daily_total_last = avail_sales[-1].sum()
            lag_arrays['daily_total_lag1'][hs:he] = daily_total_last

            if K >= 7:
                daily_totals_7d = avail_sales[-7:].sum(axis=1).mean()
                lag_arrays['daily_total_rmean7'][hs:he] = daily_totals_7d

        # Momentum (vectorized)
        rm7 = lag_arrays['rmean_7d']
        l1d = lag_arrays['lag_1d']
        valid_mom = (~np.isnan(rm7)) & (~np.isnan(l1d)) & (rm7 > 0)
        lag_arrays['momentum_1d_7d'][valid_mom] = l1d[valid_mom] / rm7[valid_mom]

        for name in lag_names:
            X[name] = lag_arrays[name]

    return X, y, stock_flat, sids_h, pids_h


# =========================================================================
# 4. Funzione di evaluation per-serie
# =========================================================================
def evaluate_perseries(preds, y, stock, sids, pids):
    """Compute pooled and per-series metrics."""
    err = preds - y
    results_pooled = {}
    for sub, smask in [('overall', np.ones(len(preds), dtype=bool)),
                       ('instock', stock == 0),
                       ('stockout', stock == 1)]:
        ef = err[smask]
        of = y[smask]
        sao = np.abs(of).sum()
        so = of.sum()
        results_pooled[f'wape_{sub}'] = np.abs(ef).sum() / sao if sao > 0 else np.nan
        results_pooled[f'wpe_{sub}'] = ef.sum() / so if so > 0 else np.nan
        results_pooled[f'n_{sub}'] = int(smask.sum())

    # Per-series
    ps_records = []
    unique_pairs = np.unique(np.column_stack([sids, pids]), axis=0)
    # Faster: use pandas groupby
    df_tmp = pd.DataFrame({
        'sid': sids, 'pid': pids,
        'pred': preds, 'obs': y, 'stock': stock
    })
    for (sid, pid), grp in df_tmp.groupby(['sid', 'pid'], sort=False):
        p = grp['pred'].values
        o = grp['obs'].values
        s = grp['stock'].values
        m = compute_metrics(p, o, s)
        m['store_id'] = sid
        m['product_id'] = pid
        ps_records.append(m)

    return results_pooled, pd.DataFrame(ps_records)


# =========================================================================
# 5. Train & evaluate both variants
# =========================================================================
VARIANTS = ['A', 'F']
variant_results = {}

for variant in VARIANTS:
    with_lags = (variant == 'F')
    print(f"\n{'='*72}")
    print(f"  VARIANTE {variant} {'(base)' if variant == 'A' else '(M5-style lags)'}")
    print(f"{'='*72}")

    # ---- 5a. Build train/val datasets ----
    print(f"\n  5a. Building datasets (variant {variant})...")
    t2 = time.time()

    df_train_sub = df_full[(df_full['day_num'] >= 2) & (df_full['day_num'] <= 83)].copy()
    df_val_sub = df_full[(df_full['day_num'] >= 84) & (df_full['day_num'] <= 90)].copy()

    X_train, y_train, stock_train, sids_train, pids_train = build_hourly_dataset(
        df_train_sub, anchor_mode='rolling', with_lags=with_lags)
    X_val, y_val, stock_val, sids_val, pids_val = build_hourly_dataset(
        df_val_sub, anchor_mode='fixed', anchor_day=83, with_lags=with_lags)

    print(f"    Train: {len(X_train):,} hourly rows, Val: {len(X_val):,} hourly rows")
    print(f"    Features: {X_train.shape[1]}")
    if with_lags:
        nan_pct = X_train[['lag_1d']].isna().mean().values[0] * 100
        print(f"    lag_1d NaN: {nan_pct:.1f}%")
        nan_mom = X_train[['momentum_1d_7d']].isna().mean().values[0] * 100
        print(f"    momentum NaN: {nan_mom:.1f}%")
    print(f"    Build time: {time.time()-t2:.0f}s")

    # ---- 5b. Train with early stopping ----
    print(f"\n  5b. Training LightGBM variant {variant}...")
    t3 = time.time()

    lgb_train_ds = lgb.Dataset(X_train, y_train, free_raw_data=True)
    lgb_val_ds = lgb.Dataset(X_val, y_val, reference=lgb_train_ds, free_raw_data=True)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(50),
    ]

    model = lgb.train(
        LGB_PARAMS, lgb_train_ds,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[lgb_val_ds],
        valid_names=['val'],
        callbacks=callbacks,
    )

    best_iter = model.best_iteration
    best_mae = model.best_score['val']['l1']
    print(f"    Best iteration: {best_iter}, Val MAE: {best_mae:.6f}")
    print(f"    Training time: {time.time()-t3:.0f}s")

    # Save model
    model_path = os.path.join(RESULTS_DIR, f'lgb_variant_{variant}.txt')
    model.save_model(model_path)

    # ---- 5c. Val evaluation ----
    print(f"\n  5c. Val evaluation (variant {variant})...")
    X_val2, y_val2, stock_val2, sids_val2, pids_val2 = build_hourly_dataset(
        df_val_sub, anchor_mode='fixed', anchor_day=83, with_lags=with_lags)

    preds_val = np.clip(model.predict(X_val2), 0, None)
    pooled_val, ps_val = evaluate_perseries(preds_val, y_val2, stock_val2,
                                            sids_val2, pids_val2)

    out_path = os.path.join(RESULTS_DIR, f'lgb_{variant.lower()}_val_per_series.parquet')
    ps_val.to_parquet(out_path, index=False)
    print(f"    WAPE_in pooled: {pooled_val['wape_instock']:.4f}")
    print(f"    WAPE_in median: {ps_val['wape_instock'].median():.4f}")
    print(f"    WPE_in pooled:  {pooled_val['wpe_instock']:.4f}")
    print(f"    Saved: {out_path}")

    # ---- 5d. Retrain on days 2-90 ----
    print(f"\n  5d. Retraining on days 2-90 (variant {variant}, {best_iter} rounds)...")
    t4 = time.time()

    df_retrain = df_full[(df_full['day_num'] >= 2) & (df_full['day_num'] <= 90)].copy()
    X_retrain, y_retrain, _, _, _ = build_hourly_dataset(
        df_retrain, anchor_mode='rolling', with_lags=with_lags)

    lgb_retrain_ds = lgb.Dataset(X_retrain, y_retrain, free_raw_data=True)
    model_retrained = lgb.train(
        LGB_PARAMS, lgb_retrain_ds,
        num_boost_round=best_iter,
    )
    print(f"    Retrain time: {time.time()-t4:.0f}s")

    # Save retrained model
    model_retrained.save_model(os.path.join(RESULTS_DIR,
                                            f'lgb_variant_{variant}_retrained.txt'))

    # ---- 5e. Test evaluation ----
    print(f"\n  5e. Test evaluation (variant {variant})...")

    df_test_sub = df_full[(df_full['day_num'] >= 91) & (df_full['day_num'] <= 97)].copy()
    X_test, y_test, stock_test, sids_test, pids_test = build_hourly_dataset(
        df_test_sub, anchor_mode='fixed', anchor_day=90, with_lags=with_lags)

    preds_test = np.clip(model_retrained.predict(X_test), 0, None)
    pooled_test, ps_test = evaluate_perseries(preds_test, y_test, stock_test,
                                              sids_test, pids_test)

    out_path = os.path.join(RESULTS_DIR, f'lgb_{variant.lower()}_test_per_series.parquet')
    ps_test.to_parquet(out_path, index=False)
    print(f"    WAPE_in pooled: {pooled_test['wape_instock']:.4f}")
    print(f"    WAPE_in median: {ps_test['wape_instock'].median():.4f}")
    print(f"    WPE_in pooled:  {pooled_test['wpe_instock']:.4f}")
    print(f"    WPE_in median:  {ps_test['wpe_instock'].median():.4f}")
    print(f"    Saved: {out_path}")

    # Feature importance (top 10)
    if with_lags:
        fi = model_retrained.feature_importance(importance_type='gain')
        fi_names = model_retrained.feature_name()
        fi_sorted = sorted(zip(fi_names, fi), key=lambda x: -x[1])
        print(f"\n    Feature importance (top 10):")
        for name, imp in fi_sorted[:10]:
            print(f"      {name:<25s} {imp:>12,.0f}")

    variant_results[variant] = {
        'best_iter': best_iter,
        'best_mae': best_mae,
        'pooled_val': pooled_val,
        'pooled_test': pooled_test,
        'ps_val': ps_val,
        'ps_test': ps_test,
    }

    del X_train, X_val, X_val2, X_retrain, X_test
    del y_train, y_val, y_val2, y_retrain, y_test

# =========================================================================
# 6. Variant selection
# =========================================================================
print(f"\n{'='*72}")
print("  6. VARIANT SELECTION")
print(f"{'='*72}")

print(f"\n  {'Var':>4} {'Iter':>6} {'Val_MAE':>10} {'Val_WAPE_in_p':>16} "
      f"{'Val_WAPE_in_m':>16}")
print("  " + "-" * 58)

for v in VARIANTS:
    vr = variant_results[v]
    print(f"  {v:>4} {vr['best_iter']:>6} {vr['best_mae']:>10.6f} "
          f"{vr['pooled_val']['wape_instock']:>16.4f} "
          f"{vr['ps_val']['wape_instock'].median():>16.4f}")

best_var = min(VARIANTS, key=lambda v: variant_results[v]['pooled_val']['wape_instock'])
print(f"\n  Best variant (val WAPE_in pooled): {best_var}")


# =========================================================================
# 7. Confronto finale
# =========================================================================
print(f"\n{'='*72}")
print("  7. CONFRONTO FINALE (TEST)")
print(f"{'='*72}")

print(f"\n  {'Modello':<20} {'WAPE_in_pool':>14} {'WAPE_in_med':>14} "
      f"{'WPE_in_pool':>14} {'WPE_in_med':>14} {'WAPE_all_med':>14}")
print("  " + "-" * 94)

# Load naive baselines from v2/results
naive_models = {
    'Global Mean': 'global_mean',
    'DoW Mean': 'dow_mean',
    'Naive Direct': 'naive_direct',
    'MA Direct': 'ma_direct',
}
for label, prefix in naive_models.items():
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if os.path.exists(path):
        ps = pd.read_parquet(path)
        print(f"  {label:<20} {'N/A':>14} {ps['wape_instock'].median():>14.4f} "
              f"{'N/A':>14} {ps['wpe_instock'].median():>14.4f} "
              f"{ps['wape_overall'].median():>14.4f}")

for v in VARIANTS:
    vr = variant_results[v]
    label = f"LGB {v}"
    pr = vr['pooled_test']
    ps = vr['ps_test']
    print(f"  {label:<20} {pr['wape_instock']:>14.4f} {ps['wape_instock'].median():>14.4f} "
          f"{pr['wpe_instock']:>14.4f} {ps['wpe_instock'].median():>14.4f} "
          f"{ps['wape_overall'].median():>14.4f}")

print(f"\n  Tempo totale: {time.time()-t0:.0f}s")
print("=" * 72)
