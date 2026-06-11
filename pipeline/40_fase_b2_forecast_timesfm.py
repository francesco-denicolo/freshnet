"""
40_fase_b2_forecast_timesfm.py — TimesFM 2.5-200M forecaster per singolo imputer (ore 6-22)
=========================================================================================
Usage: freshnet_timesfm/bin/python pipeline/40_fase_b2_forecast_timesfm.py <imputer_key>

Adattato da 10_fase_b2_forecast_chronos.py. Identica struttura, modello sostituito.
- Context: 1530 punti (90 giorni × 17 ore)
- Pred: 119 punti (7 giorni × 17 ore)
- TimesFM 2.5-200M (Google, decoder-only, PyTorch backend)
"""
import sys, os, time, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
import torch
import timesfm

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

H_START, H_END = 6, 23; N_HOURS = H_END - H_START
CONTEXT_DAYS = 90; PRED_DAYS = 7
CONTEXT_LEN = CONTEXT_DAYS * N_HOURS  # 1530
PRED_LEN = PRED_DAYS * N_HOURS        # 119
BATCH_SIZE = 32  # TimesFM 200M, smaller than Chronos-bolt-small 50M; smaller batch safer

IMP_KEY = sys.argv[1] if len(sys.argv) > 1 else 'no_imp'
IMP_LABELS = {
    'no_imp': 'No imputation',
    'media_cond': 'Media condizionata', 'media_glob': 'Media globale',
    'mediana_cond': 'Mediana condizionata', 'mediana_glob': 'Mediana globale',
    'lgb': 'LGB imputer',
    'dlinear': 'DLinear',
    'forward_fill': 'Forward Fill',
    'seasonal_naive': 'Seasonal Naive',
    'linear_interp': 'Linear Interp',
    'saits': 'SAITS',
    'itransformer': 'iTransformer',
    'timesnet': 'TimesNet',
    'imputeformer': 'ImputeFormer',
}
assert IMP_KEY in IMP_LABELS, f'Unknown imputer: {IMP_KEY}'

cell_key = f'{IMP_KEY}__timesfm'
out_path = os.path.join(RESULTS_DIR, f'{cell_key}_test_per_series.parquet')
if os.path.exists(out_path):
    print(f'SKIP: {out_path} exists'); sys.exit(0)

print(f'=== TimesFM 2.5-200M × {IMP_LABELS[IMP_KEY]} (ore 6-22) ===')
print(f'Context: {CONTEXT_LEN} pts, Pred: {PRED_LEN} pts, Batch: {BATCH_SIZE}')

# ============================================================================
# 1. Load data (identico a Chronos)
# ============================================================================
print('\n1. Loading data...')
t0 = time.time()
df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_train_hf['dt_parsed'] = pd.to_datetime(df_train_hf['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])
df_full = pd.concat([df_train_hf, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i+1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)

sales_orig = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
stock_orig = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)[:, H_START:H_END]
del df_train_hf, df_eval

# ============================================================================
# 2. Load completed_sales (if not no_imp)
# ============================================================================
if IMP_KEY != 'no_imp':
    print(f'\n2. Loading completed_sales: {IMP_KEY}...')
    df_cs = pd.read_parquet(os.path.join(COMPLETED_DIR, f'{IMP_KEY}.parquet'))
    cs_sales = np.array(df_cs['hours_sale'].tolist(), dtype=np.float32)
    if cs_sales.shape[1] == 24:
        cs_sales = cs_sales[:, H_START:H_END]
    df_cs['dt_parsed'] = pd.to_datetime(df_cs['dt'])
    df_cs = df_cs.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
    key_full = df_full[['store_id','product_id','dt_parsed']].apply(tuple, axis=1)
    key_cs = df_cs[['store_id','product_id','dt_parsed']].apply(tuple, axis=1)
    cs_idx = {k: i for i, k in enumerate(key_cs)}
    sales_ctx = sales_orig.copy()
    matched = 0
    for i, k in enumerate(key_full):
        if k in cs_idx:
            sales_ctx[i] = cs_sales[cs_idx[k]]
            matched += 1
    print(f'  Matched: {matched:,}/{len(df_full):,}')
    del df_cs, cs_sales, cs_idx
else:
    sales_ctx = sales_orig.copy()

# ============================================================================
# 3. Build series arrays
# ============================================================================
print('\n3. Building series arrays...')
t1 = time.time()
groups = df_full.groupby(['store_id','product_id'], sort=False)
n_series = len(groups)
print(f'  Series: {n_series:,}')

context_all = np.zeros((n_series, CONTEXT_LEN), dtype=np.float32)
target_sales_all = np.zeros((n_series, PRED_LEN), dtype=np.float32)
target_stock_all = np.zeros((n_series, PRED_LEN), dtype=np.float32)
series_keys = np.zeros((n_series, 2), dtype=np.int64)

for i, ((sid, pid), idx) in enumerate(groups.indices.items()):
    idx_sorted = np.array(sorted(idx, key=lambda k: df_full.iloc[k]['day_num']))
    days = df_full.iloc[idx_sorted]['day_num'].values
    ctx_mask = days <= CONTEXT_DAYS
    tgt_mask = days > CONTEXT_DAYS
    ctx_idx = idx_sorted[ctx_mask]
    tgt_idx = idx_sorted[tgt_mask]
    context_all[i] = sales_ctx[ctx_idx].flatten()
    target_sales_all[i] = sales_orig[tgt_idx].flatten()
    target_stock_all[i] = stock_orig[tgt_idx].flatten()
    series_keys[i] = [sid, pid]
    if (i+1) % 10000 == 0:
        print(f'    {i+1:,}/{n_series:,}')
print(f'  Build time: {time.time()-t1:.1f}s')
del df_full, sales_ctx, sales_orig, stock_orig

# ============================================================================
# 4. Load TimesFM
# ============================================================================
print('\n4. Loading TimesFM 2.5-200M...')
t0 = time.time()
model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
model.compile(timesfm.ForecastConfig(
    max_context=2048,         # > 1530 (our context len)
    max_horizon=128,          # > 119 (our pred len)
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    per_core_batch_size=BATCH_SIZE,
    force_flip_invariance=True,
    infer_is_positive=True,
))
print(f'  Loaded in {time.time()-t0:.1f}s')

# ============================================================================
# 5. Predict
# ============================================================================
print(f'\n5. Predicting {n_series:,} series in batches of {BATCH_SIZE}...')
t0 = time.time()
predictions = np.zeros((n_series, PRED_LEN), dtype=np.float32)

for b_start in range(0, n_series, BATCH_SIZE):
    b_end = min(b_start + BATCH_SIZE, n_series)
    batch = [context_all[i].astype(np.float32) for i in range(b_start, b_end)]
    mean_pred, _quantile = model.forecast(horizon=PRED_LEN, inputs=batch)
    pred_batch = np.asarray(mean_pred)[:, :PRED_LEN]
    pred_batch = np.clip(pred_batch, 0, None)
    predictions[b_start:b_end] = pred_batch
    if (b_start // BATCH_SIZE + 1) % 10 == 0:
        elapsed = time.time() - t0
        progress = b_end / n_series
        eta = elapsed / progress - elapsed
        print(f'    {b_end:,}/{n_series:,}  elapsed={elapsed:.0f}s  eta={eta:.0f}s')

print(f'  Predict time: {time.time()-t0:.1f}s')

# ============================================================================
# 6. Evaluation (identico a Chronos)
# ============================================================================
print('\n6. Evaluation...')
instock_mask = (target_stock_all == 0).astype(np.float32)
abs_err = np.abs(predictions - target_sales_all) * instock_mask
signed_err = (predictions - target_sales_all) * instock_mask
obs_instock = target_sales_all * instock_mask

num_wape = abs_err.sum()
den_wape = obs_instock.sum()
wape_pool = num_wape / max(den_wape, 1e-9)
wpe_pool = signed_err.sum() / max(den_wape, 1e-9)

num_series = abs_err.sum(axis=1)
den_series = obs_instock.sum(axis=1)
signed_series = signed_err.sum(axis=1)
n_hours_instock = instock_mask.sum(axis=1)

wape_series = np.where(den_series > 0, num_series / np.maximum(den_series, 1e-9), np.nan)
wpe_series = np.where(den_series > 0, signed_series / np.maximum(den_series, 1e-9), np.nan)

wape_med = np.nanmedian(wape_series)
wpe_med = np.nanmedian(wpe_series)

print(f'  WAPE pool={wape_pool:.4f}, med={wape_med:.4f}')
print(f'  WPE  pool={wpe_pool:.4f}, med={wpe_med:.4f}')

# Daily aggregation
daily_sales_pred = predictions.reshape(n_series, PRED_DAYS, N_HOURS).sum(axis=2)
daily_sales_true = target_sales_all.reshape(n_series, PRED_DAYS, N_HOURS).sum(axis=2)
daily_instock_count = instock_mask.reshape(n_series, PRED_DAYS, N_HOURS).sum(axis=2)

daily_abs_err = np.abs(daily_sales_pred - daily_sales_true) * (daily_instock_count > 0)
daily_signed_err = (daily_sales_pred - daily_sales_true) * (daily_instock_count > 0)
daily_obs = daily_sales_true * (daily_instock_count > 0)
den_d = daily_obs.sum(axis=1)
daily_wape = np.where(den_d > 0, daily_abs_err.sum(axis=1) / np.maximum(den_d, 1e-9), np.nan)
daily_wpe = np.where(den_d > 0, daily_signed_err.sum(axis=1) / np.maximum(den_d, 1e-9), np.nan)

# ============================================================================
# 7. Save
# ============================================================================
df_out = pd.DataFrame({
    'store_id': series_keys[:, 0],
    'product_id': series_keys[:, 1],
    'hourly_wape': wape_series,
    'hourly_wpe': wpe_series,
    'daily_wape': daily_wape,
    'daily_wpe': daily_wpe,
    'n_hours_instock': n_hours_instock.astype(np.int32),
})
df_out.to_parquet(out_path, index=False)
print(f'\nSaved: {out_path}')
print(f'\n=== DONE — TimesFM × {IMP_LABELS[IMP_KEY]} ===')
