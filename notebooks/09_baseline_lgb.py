"""
09_baseline_lgb.py — LightGBM Baseline (Direct Forecast, M5-style features)
=============================================================================
PINN-Retail: Physics-Informed Neural Networks per Demand Forecasting
di Prodotti Deperibili.

Passo 3h: LightGBM baseline su vendite osservate (censurate).
Modello singolo per-ora: ogni riga = (store, product, giorno, ora) -> 1 vendita.
Features categoriche gestite nativamente da LightGBM (no embedding).
Loss: MAE (regression) su tutte le ore (incluse stockout -- ignora censoring).

Feature engineering M5-style (tabular forecasting literature):
  A: nessuno storico (12 features base)
  F: M5-style full features (+11 lag/rolling/momentum features = 23 totali)

Le 11 features lag:
  - Raw lags: lag_1d, lag_7d, lag_14d (stessa ora)
  - Rolling stats: rmean_7d, rmean_14d, rstd_7d (stessa ora)
  - DoW-specific: lag_dow, rmean_dow
  - Aggregati giornalieri: daily_total_lag1, daily_total_rmean7
  - Momentum: momentum_1d_7d (lag_1d / rmean_7d)

Eseguire con: freshnet/bin/python notebooks/09_baseline_lgb.py
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

LAG_VARIANTS = {
    'F': 'M5-style full (+11 lag feat)',
}

# Variant A results from previous run (reuse, don't recompute)
VARIANT_A_CACHED = {
    'wape_pooled': 1.035188,
    'wape_median': 1.262399,
    'best_iter': 298,
    'best_mae': 0.051602,
    'elapsed': 1355.2,
}

# ===========================================================================
print('=' * 72)
print('  LIGHTGBM BASELINE — DIRECT FORECAST')
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

# ---------------------------------------------------------------------------
# 2. Preparazione dati vettorizzata
# ---------------------------------------------------------------------------
print('\n2. Preparazione dati...')

# Pre-parse hourly arrays (once, vectorized)
sales_all = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)    # (N_full, 24)
stock_all = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)  # (N_full, 24)


def build_hourly_dataset(df, sales_arr, stock_arr, split, variant):
    """Build flat per-hour dataset for LightGBM (vectorized).

    Each row = (store, product, day, hour) → 1 target value.
    Uses numpy vectorization instead of Python loops.

    Returns: X (DataFrame), y (array), stock_flat (array),
             store_ids, product_ids
    """
    if split == 'train':
        d_min, d_max = 2, 83
    elif split == 'val':
        d_min, d_max = 84, 90
    elif split == 'test':
        d_min, d_max = 91, 97

    # Filter rows for this split
    mask = (df['day_num'] >= d_min) & (df['day_num'] <= d_max)
    df_split = df[mask]
    idx_split = np.where(mask.values)[0]
    n_days = len(df_split)

    # Base features at day level
    store_ids_day = df_split['store_id'].values
    product_ids_day = df_split['product_id'].values
    city_ids_day = df_split['city_id'].values
    dows_day = df_split['dow'].values
    conts_day = df_split[CONT_FEATURES].values.astype(np.float32)  # (n_days, 7)
    day_nums_day = df_split['day_num'].values

    # Targets and stock at day level (already parsed)
    sales_day = sales_arr[idx_split]   # (n_days, 24)
    stock_day = stock_arr[idx_split]   # (n_days, 24)

    # Expand to hourly: each day → 24 rows
    n_hourly = n_days * 24
    hours = np.tile(np.arange(24, dtype=np.int32), n_days)  # [0,1,...,23, 0,1,...,23, ...]

    # Repeat day-level features 24 times each
    store_ids_h = np.repeat(store_ids_day, 24)
    product_ids_h = np.repeat(product_ids_day, 24)
    city_ids_h = np.repeat(city_ids_day, 24)
    dows_h = np.repeat(dows_day, 24)
    conts_h = np.repeat(conts_day, 24, axis=0)  # (n_hourly, 7)

    # Flatten sales and stock
    y = sales_day.ravel().astype(np.float32)     # (n_hourly,)
    stock_flat = stock_day.ravel().astype(np.float32)

    # Build base feature array
    feat_dict = {
        'store_id': store_ids_h,
        'product_id': product_ids_h,
        'city_id': city_ids_h,
        'dow': dows_h,
        'hour': hours,
    }
    for j, c in enumerate(CONT_FEATURES):
        feat_dict[c] = conts_h[:, j]

    # Compute M5-style lag features for variant F
    if variant == 'F':
        # Pre-allocate 11 lag arrays with NaN (LightGBM handles NaN natively)
        lag_arrays = {name: np.full(n_hourly, np.nan, dtype=np.float32)
                      for name in LAG_FEATURES_F}

        # Build series cache: (store, product) -> {days, dows, sales}
        print('      Building series cache...')
        groups_full = df.groupby(['store_id', 'product_id'], sort=False)
        series_cache = {}
        for (sid, pid), grp in groups_full:
            grp_s = grp.sort_values('day_num')
            series_cache[(sid, pid)] = {
                'days': grp_s['day_num'].values,
                'dows': grp_s['dow'].values,
                'sales': sales_arr[grp_s.index.values],  # (N_series, 24)
            }

        # Compute lag features per day
        print(f'      Computing lag features for {n_days:,} days...')
        day_offset = 0
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
            s_sales = sc['sales']  # (N_series_days, 24)

            # Anchor: train=rolling(d-1), val=fixed(83), test=fixed(90)
            if split == 'train':
                a_day = d - 1
            elif split == 'val':
                a_day = 83
            else:
                a_day = 90

            avail_mask = s_days <= a_day
            K = int(avail_mask.sum())
            hs = day_offset * 24      # hourly_start
            he = hs + 24              # hourly_end

            if K > 0:
                avail_sales = s_sales[avail_mask]  # (K, 24)
                avail_dows = s_dows[avail_mask]    # (K,)

                # --- Raw lags (same hour, vectorized across 24h) ---
                lag_arrays['lag_1d'][hs:he] = avail_sales[-1]       # last day
                if K >= 7:
                    lag_arrays['lag_7d'][hs:he] = avail_sales[-7]   # 7 days ago
                if K >= 14:
                    lag_arrays['lag_14d'][hs:he] = avail_sales[-14] # 14 days ago

                # --- Rolling means (same hour) ---
                if K >= 7:
                    lag_arrays['rmean_7d'][hs:he] = avail_sales[-7:].mean(axis=0)
                if K >= 14:
                    lag_arrays['rmean_14d'][hs:he] = avail_sales[-14:].mean(axis=0)

                # --- Rolling std (same hour, window=min(7,K), need K>=2) ---
                if K >= 2:
                    w = min(7, K)
                    lag_arrays['rstd_7d'][hs:he] = avail_sales[-w:].std(axis=0)

                # --- DoW-specific lags ---
                same_dow = avail_dows == dow_val
                if same_dow.any():
                    dow_sales = avail_sales[same_dow]  # (n_dow, 24)
                    lag_arrays['lag_dow'][hs:he] = dow_sales[-1]
                    lag_arrays['rmean_dow'][hs:he] = dow_sales.mean(axis=0)

                # --- Daily aggregates (same value for all 24 hours) ---
                daily_totals = avail_sales.sum(axis=1)  # (K,)
                lag_arrays['daily_total_lag1'][hs:he] = daily_totals[-1]
                if K >= 7:
                    lag_arrays['daily_total_rmean7'][hs:he] = daily_totals[-7:].mean()

            day_offset += 1

        # --- Momentum: lag_1d / rmean_7d (vectorized after loop) ---
        l1 = lag_arrays['lag_1d']
        rm7 = lag_arrays['rmean_7d']
        valid_mom = (~np.isnan(l1)) & (~np.isnan(rm7)) & (rm7 > 0)
        lag_arrays['momentum_1d_7d'][valid_mom] = l1[valid_mom] / rm7[valid_mom]

        # Insert all lag features into feat_dict
        for name in LAG_FEATURES_F:
            feat_dict[name] = lag_arrays[name]

        # Print NaN stats
        print('      NaN counts per lag feature:')
        for name in LAG_FEATURES_F:
            nan_count = np.isnan(lag_arrays[name]).sum()
            pct = 100.0 * nan_count / n_hourly
            print(f'        {name:<22} {nan_count:>12,} ({pct:.1f}%)')

        del lag_arrays, series_cache
        gc.collect()

    X = pd.DataFrame(feat_dict)
    del feat_dict
    gc.collect()

    for c in CAT_FEATURES:
        X[c] = X[c].astype('category')

    return X, y, stock_flat, store_ids_h, product_ids_h


# ---------------------------------------------------------------------------
# 4. Variant selection loop
# ---------------------------------------------------------------------------
print('\n3. Selezione variante lag su validation...')
print(f'   Varianti: {LAG_VARIANTS}\n')

variant_results = {}

for variant, desc in LAG_VARIANTS.items():
    print(f'  --- Variante {variant}: {desc} ---')
    t0 = time.time()

    # Build train dataset
    print(f'    Costruzione dataset train...')
    X_train, y_train, _, _, _ = build_hourly_dataset(df_full, sales_all, stock_all, 'train', variant)
    print(f'    Train: {len(X_train):,} righe, {X_train.shape[1]} features')

    # Build val dataset
    print(f'    Costruzione dataset val...')
    X_val, y_val, stock_val, sids_val, pids_val = \
        build_hourly_dataset(df_full, sales_all, stock_all, 'val', variant)
    print(f'    Val:   {len(X_val):,} righe')

    # Train LightGBM
    print(f'    Training LightGBM...')
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

    # Predict on val
    preds_val = model.predict(X_val)

    # Compute val WAPE pooled (on all hours)
    sae = np.abs(preds_val - y_val).sum()
    sao = np.abs(y_val).sum()
    val_wape = sae / sao if sao > 0 else float('inf')

    # Compute val WAPE median per-serie (vectorized via pandas)
    df_tmp = pd.DataFrame({
        'sid': sids_val, 'pid': pids_val,
        'abs_err': np.abs(preds_val - y_val),
        'abs_obs': np.abs(y_val),
    })
    grp_sums = df_tmp.groupby(['sid', 'pid'], sort=False)[['abs_err', 'abs_obs']].sum()
    valid = grp_sums['abs_obs'] > 0
    ps_wapes = (grp_sums.loc[valid, 'abs_err'] / grp_sums.loc[valid, 'abs_obs']).values
    med_wape = np.median(ps_wapes) if len(ps_wapes) > 0 else np.nan
    del df_tmp, grp_sums

    elapsed = time.time() - t0
    variant_results[variant] = {
        'wape_pooled': val_wape,
        'wape_median': med_wape,
        'best_iter': best_iter,
        'best_mae': best_score,
        'elapsed': elapsed,
    }

    print(f'    Best iter: {best_iter}, MAE: {best_score:.6f}')
    print(f'    Val WAPE pooled: {val_wape:.6f}, median: {med_wape:.6f}, '
          f'time: {elapsed:.1f}s\n')

    # Save model for this variant
    model.save_model(os.path.join(RESULTS_DIR, f'lgb_variant_{variant}.txt'))

    del lgb_train, lgb_val_ds, model, X_train, y_train, X_val, preds_val
    gc.collect()

# ---------------------------------------------------------------------------
# 5. Variant selection table
# ---------------------------------------------------------------------------
# Inject cached variant A results for comparison
variant_results['A'] = VARIANT_A_CACHED

print('\n' + '=' * 72)
print('  4. SELEZIONE VARIANTE LAG')
print('=' * 72)

all_variants_desc = {'A': 'No history (12 feat)', **LAG_VARIANTS}
print(f'\n  {"Var":<4} {"Description":<32} {"WAPE_pool":>10} {"WAPE_med":>10} '
      f'{"Iter":>6} {"MAE":>10} {"Time":>8}')
print('  ' + '-' * 84)

for v in ['A'] + list(LAG_VARIANTS.keys()):
    r = variant_results[v]
    desc = all_variants_desc.get(v, v)
    print(f'  {v:<4} {desc:<32} {r["wape_pooled"]:>10.6f} '
          f'{r["wape_median"]:>10.6f} {r["best_iter"]:>6d} '
          f'{r["best_mae"]:>10.6f} {r["elapsed"]:>7.1f}s')

best_var_pooled = min(variant_results, key=lambda v: variant_results[v]['wape_pooled'])
best_var_median = min(variant_results, key=lambda v: variant_results[v]['wape_median'])

print(f'\n  Best (WAPE pooled):  variant {best_var_pooled} '
      f'({variant_results[best_var_pooled]["wape_pooled"]:.6f})')
print(f'  Best (WAPE median):  variant {best_var_median} '
      f'({variant_results[best_var_median]["wape_median"]:.6f})')

if best_var_pooled == best_var_median:
    BEST_VAR = best_var_pooled
    print(f'\n  Entrambi i criteri concordano: variante {BEST_VAR}')
else:
    BEST_VAR = best_var_pooled
    print(f'\n  Criteri discordanti. Uso variante {BEST_VAR} (pooled).')

# ---------------------------------------------------------------------------
# 6. Load best model and evaluate on val + test
# ---------------------------------------------------------------------------
print(f'\n5. Valutazione completa con variante {BEST_VAR}...')

best_model = lgb.Booster(model_file=os.path.join(RESULTS_DIR, f'lgb_variant_{BEST_VAR}.txt'))

# Save as the "best" model
best_model.save_model(os.path.join(RESULTS_DIR, 'lgb_best.txt'))

pooled_results = {}
per_series_dfs = {}

for split_name in ['val', 'test']:
    print(f'\n  Valutazione {split_name}...')
    X_split, y_split, stock_split, sids, pids = \
        build_hourly_dataset(df_full, sales_all, stock_all, split_name, BEST_VAR)
    print(f'    {len(X_split):,} righe')

    preds = best_model.predict(X_split)
    preds = np.clip(preds, 0, None)  # ensure non-negative

    # Pooled metrics
    p_flat = preds
    o_flat = y_split
    s_flat = stock_split

    r = {}
    for sub, smask in [('overall', np.ones(len(p_flat), dtype=bool)),
                       ('instock', s_flat == 0),
                       ('stockout', s_flat == 1)]:
        ef = (p_flat - o_flat)[smask]
        of = o_flat[smask]
        sae = np.abs(ef).sum()
        sao = np.abs(of).sum()
        r[f'wape_{sub}'] = sae / sao if sao > 0 else np.nan
        r[f'wpe_{sub}'] = ef.sum() / of.sum() if of.sum() != 0 else np.nan
        r[f'n_{sub}'] = int(smask.sum())
    pooled_results[split_name] = r

    # Per-series metrics (vectorized using pandas groupby)
    print('    Calcolo metriche per-serie...')
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
    out_path = os.path.join(RESULTS_DIR, f'lgb_{split_name}_per_series.parquet')
    ps.to_parquet(out_path, index=False)
    print(f'    Salvato: {out_path} ({len(ps):,} serie)')

    del X_split, preds
    gc.collect()

# ---------------------------------------------------------------------------
# 7. Tabella risultati pooled
# ---------------------------------------------------------------------------
print(format_metrics_table(pooled_results,
                            model_name=f'LightGBM Baseline (variant {BEST_VAR})'))

# ---------------------------------------------------------------------------
# 8. Distribuzione per-serie
# ---------------------------------------------------------------------------
METRIC_COLS = ['wape_overall', 'wape_instock', 'wape_stockout',
               'wpe_overall', 'wpe_instock', 'wpe_stockout']

print('\n' + '=' * 72)
print('  6. DISTRIBUZIONE METRICHE PER-SERIE')
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
        print(f'  {split_name:<8} {col:<16} {vals.mean():>8.4f} {vals.median():>8.4f} '
              f'{vals.std():>8.4f} {q5:>8.4f} {q95:>8.4f} {len(vals):>7,}')

# ---------------------------------------------------------------------------
# 9. Confronto con tutti i baseline
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  7. CONFRONTO CON TUTTI I BASELINE (in-stock, test)')
print('=' * 72)

all_baselines = {
    'Naive (direct)': 'naive_direct',
    'MA K=14 (direct)': 'ma_direct_K14',
    'Global Mean': 'global_mean',
    'DoW Mean': 'dow_mean',
    'MLP (var A)': 'mlp',
    f'LGB (var {BEST_VAR})': 'lgb',
}

print(f'\n  {"Model":<24} {"WAPE_in med":>12} {"WPE_in med":>12} '
      f'{"WAPE_all med":>14}')
print('  ' + '-' * 66)

for label, prefix in all_baselines.items():
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if not os.path.exists(path):
        continue
    ps = pd.read_parquet(path)
    wape_in = ps['wape_instock'].median()
    wpe_in = ps['wpe_instock'].median()
    wape_all = ps['wape_overall'].median()
    print(f'  {label:<24} {wape_in:>12.4f} {wpe_in:>12.4f} {wape_all:>14.4f}')

# ---------------------------------------------------------------------------
# 10. Figure
# ---------------------------------------------------------------------------
print('\n8. Generazione figure...')

# Fig 35: Variant selection bar chart
fig, ax = plt.subplots(figsize=(8, 5))
variants = list(variant_results.keys())
wape_pooled = [variant_results[v]['wape_pooled'] for v in variants]
wape_median = [variant_results[v]['wape_median'] for v in variants]
x = np.arange(len(variants))
w = 0.35

ax.bar(x - w/2, wape_pooled, w, label='WAPE pooled (val)', color='steelblue', alpha=0.8)
ax.bar(x + w/2, wape_median, w, label='WAPE median (val)', color='darkorange', alpha=0.8)
ax.set_xlabel('Lag variant')
ax.set_ylabel('WAPE on validation')
ax.set_title('LightGBM — Lag Variant Selection')
ax.set_xticks(x)
ax.set_xticklabels([f'{v}: {all_variants_desc.get(v, v)}' for v in variants], rotation=30, ha='right')
ax.legend()

best_idx = variants.index(BEST_VAR)
ax.annotate(f'Best: {BEST_VAR}', xy=(best_idx, wape_pooled[best_idx]),
            xytext=(best_idx + 0.5, wape_pooled[best_idx] + 0.02),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red', fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig35_lgb_variant_selection.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig35_lgb_variant_selection.png')

# Fig 36: Boxplot confronto in-stock val+test (tutti i modelli)
colors_all = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#DD8452']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Confronto Tutti i Baseline — Metriche In-Stock (per-serie)', fontsize=15, y=0.98)

for j, split in enumerate(['val', 'test']):
    for row, (metric, ylabel) in enumerate([('wape_instock', 'WAPE in-stock'),
                                              ('wpe_instock', 'WPE in-stock')]):
        ax = axes[row, j]
        box_data = []
        box_labels = []
        box_colors = []
        medians = []

        for k, (label, prefix) in enumerate(all_baselines.items()):
            path = os.path.join(RESULTS_DIR, f'{prefix}_{split}_per_series.parquet')
            if not os.path.exists(path):
                continue
            ps = pd.read_parquet(path)
            vals = ps[metric].dropna()
            if metric.startswith('wape'):
                q99 = vals.quantile(0.99)
                box_data.append(vals.clip(upper=q99).values)
            else:
                q01, q99 = vals.quantile(0.01), vals.quantile(0.99)
                box_data.append(vals.clip(lower=q01, upper=q99).values)
            box_labels.append(label)
            box_colors.append(colors_all[k % len(colors_all)])
            medians.append(vals.median())

        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.6)
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
                ax.text(k + 1, med + 0.01, f'{med:.3f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color='red')
            else:
                offset = 0.005 if med >= 0 else -0.005
                va = 'bottom' if med >= 0 else 'top'
                ax.text(k + 1, med + offset, f'{med:.4f}', ha='center', va=va,
                        fontsize=8, fontweight='bold', color='red')

        ax.set_title(f'{ylabel} — {split}', fontsize=13)
        ax.set_ylabel(ylabel if j == 0 else '')
        ax.tick_params(axis='x', rotation=25)

fig.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(FIG_DIR, 'fig36_compare_instock_all_with_lgb.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Salvata: {out_path}')

# Feature importance
print('\n9. Feature importance (top 15)...')
best_model_reload = lgb.Booster(model_file=os.path.join(RESULTS_DIR, 'lgb_best.txt'))
importance = best_model_reload.feature_importance(importance_type='gain')
feat_names = best_model_reload.feature_name()

fi = sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)
print(f'\n  {"Feature":<20} {"Importance (gain)":>18}')
print('  ' + '-' * 42)
for name, imp in fi[:15]:
    print(f'  {name:<20} {imp:>18,.0f}')

print('\n' + '=' * 72)
print('  DONE')
print('=' * 72)
