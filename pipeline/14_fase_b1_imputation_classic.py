"""
14_fase_b1_imputation_classic.py — 3 imputer classici di time-series (ore 6-22)
================================================================================
1. Forward Fill — propaga l'ultimo valore in-stock
2. Seasonal Naive — valore stesso (dow, hour) più recente in-stock (1, 2, 3... settimane fa)
3. Linear Interpolation — interpolazione lineare tra in-stock più vicini
"""
import os, time, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
os.makedirs(COMPLETED_DIR, exist_ok=True)

H_START, H_END = 6, 23
N_HOURS = H_END - H_START  # 17
WEEK_STEPS = 7 * N_HOURS   # 119 step = 1 settimana

print('='*72)
print('  FASE B1 EXTRA — 3 IMPUTER CLASSICI (ore 6-22)')
print('='*72)

# ===========================================================================
# 1. Load data
# ===========================================================================
print('\n1. Caricamento dati...')
df = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df['dt_parsed'] = pd.to_datetime(df['dt'])
df = df.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
all_dates = sorted(df['dt_parsed'].unique())
date_to_day = {d: i+1 for i, d in enumerate(all_dates)}
df['day_num'] = df['dt_parsed'].map(date_to_day)
df['dow'] = df['dt_parsed'].dt.dayofweek

sales_all = np.array(df['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
stock_all = np.array(df['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]
n_days = len(all_dates)
print(f'  Rows: {len(df):,}, Days: {n_days}')

# Group into (n_series, n_days, 17)
series_keys = df.groupby(['store_id','product_id'], sort=False).indices
n_series = len(series_keys)

t0 = time.time()
sales_3d = np.zeros((n_series, n_days, N_HOURS), dtype=np.float32)
stock_3d = np.zeros((n_series, n_days, N_HOURS), dtype=np.int8)
series_sid = np.zeros(n_series, dtype=np.int64)
series_pid = np.zeros(n_series, dtype=np.int64)
series_row_idx = []
for i, ((sid, pid), idx) in enumerate(series_keys.items()):
    assert len(idx) == n_days
    sales_3d[i] = sales_all[idx]
    stock_3d[i] = stock_all[idx]
    series_sid[i] = sid
    series_pid[i] = pid
    series_row_idx.append(idx)
print(f'  Tensor build: {time.time()-t0:.1f}s, shape={sales_3d.shape}')

# Flatten chronologically
L_FULL = n_days * N_HOURS
sales_flat = sales_3d.reshape(n_series, L_FULL)
stock_flat = stock_3d.reshape(n_series, L_FULL)

# ===========================================================================
# 2. Load MNAR masks
# ===========================================================================
print('\n2. Loading MNAR masks...')
masks_val = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_val.parquet'))
masks_val = masks_val[(masks_val['hour'] >= H_START) & (masks_val['hour'] < H_END)].reset_index(drop=True)
masks_val['dt_parsed'] = pd.to_datetime(masks_val['dt'])
masks_val['day_num'] = masks_val['dt_parsed'].map(date_to_day)

series_to_i = {(s,p):i for i,(s,p) in enumerate(zip(series_sid, series_pid))}
mask_series_i = np.array([series_to_i.get((s,p), -1) for s,p in zip(masks_val['store_id'], masks_val['product_id'])], dtype=np.int64)
mask_day_i = masks_val['day_num'].values - 1
mask_hour_off = masks_val['hour'].values - H_START
mask_flat_idx = mask_day_i * N_HOURS + mask_hour_off

valid = mask_series_i >= 0
masks_val = masks_val[valid].reset_index(drop=True)
mask_series_i = mask_series_i[valid]
mask_flat_idx = mask_flat_idx[valid]
print(f'  MNAR masks: {len(masks_val):,}')

# Build MNAR mask matrix
mnar_mask_flat = np.zeros((n_series, L_FULL), dtype=np.int8)
mnar_mask_flat[mask_series_i, mask_flat_idx] = 1

# ===========================================================================
# Helper functions
# ===========================================================================
def eval_mnar(imputed, label):
    gt = masks_val['ground_truth'].values.astype(np.float64)
    preds = imputed[mask_series_i, mask_flat_idx].astype(np.float64)
    sao = np.abs(gt).sum(); so = gt.sum()
    sae = np.abs(preds-gt).sum(); se = (preds-gt).sum()
    wape = sae/sao if sao>0 else np.nan
    wpe = se/so if so!=0 else np.nan
    print(f'  {label}: WAPE_recovery={wape:.4f}, WPE_recovery={wpe:.4f}')
    return {'wape_recovery': wape, 'wpe_recovery': wpe}

def save_completed(imputed_flat, label):
    """imputed_flat shape (n_series, L_FULL). Use on stockout positions only."""
    completed = sales_flat.copy()
    so_mask = stock_flat == 1
    completed[so_mask] = np.clip(imputed_flat[so_mask], 0, None)
    # Back to (n_rows, 17) in original df order
    completed_out = np.zeros((len(df), N_HOURS), dtype=np.float32)
    for i, idx in enumerate(series_row_idx):
        completed_out[idx] = completed[i].reshape(n_days, N_HOURS)
    df_out = df[['store_id','product_id','dt','day_num','dow']].copy()
    df_out['hours_sale'] = list(completed_out)
    df_out['hours_stock_status'] = list(stock_all)
    out_path = os.path.join(COMPLETED_DIR, f'{label}.parquet')
    df_out.to_parquet(out_path, index=False)
    imp_mean = completed[so_mask].mean()
    obs_mean = sales_flat[~so_mask].mean()
    print(f'  Salvato: {out_path}  (mean imputed={imp_mean:.4f}, S_obs={obs_mean:.4f})')

# ===========================================================================
# Imputers (vectorized)
# ===========================================================================
def forward_fill_vec(sales, stock_combined):
    """sales: (n, L) with NaN at missing. Returns imputed (n, L)."""
    s_nan = np.where(stock_combined == 1, np.nan, sales)
    filled = pd.DataFrame(s_nan).ffill(axis=1).bfill(axis=1).fillna(0.0).values
    return filled.astype(np.float32)

def linear_interp_vec(sales, stock_combined):
    s_nan = np.where(stock_combined == 1, np.nan, sales)
    # interpolate row-wise: need to transpose for pandas (interpolate axis=1 is supported but has quirks)
    df_nan = pd.DataFrame(s_nan.T)  # shape (L, n_series)
    interp = df_nan.interpolate(method='linear', limit_direction='both').fillna(0.0).values.T
    return interp.astype(np.float32)

def seasonal_naive_vec(sales_obs, stock_obs, stock_combined, L):
    """
    sales_obs: original S_obs (used as candidate source from previous weeks, at positions
               where stock_obs == 0, i.e., truly in-stock)
    stock_obs: original stock status (used to decide if candidate was in-stock)
    stock_combined: stock mask including additional MNAR (these positions need imputation)
    Returns: imputed (n, L)
    """
    s = sales_obs[:, :L].astype(np.float32)
    k_orig = stock_obs[:, :L]
    k_comb = stock_combined[:, :L]
    # Initialize: NaN at positions to impute
    imputed = np.where(k_comb == 1, np.nan, s)
    # Progressive seasonal fallback: 1 week back, 2 weeks back, ..., 12 weeks back
    max_weeks = L // WEEK_STEPS
    for w in range(1, max_weeks + 1):
        shift = w * WEEK_STEPS
        cand_val = np.full_like(s, np.nan, dtype=np.float32)
        cand_stock = np.ones_like(k_orig, dtype=np.int8)  # default: stockout (can't use)
        cand_val[:, shift:] = sales_obs[:, :L-shift]
        cand_stock[:, shift:] = stock_obs[:, :L-shift]  # original status
        # Fill missing (still NaN) where candidate was truly in-stock
        needs = np.isnan(imputed) & (cand_stock == 0)
        imputed[needs] = cand_val[needs]
    # Forward fill fallback for still-NaN
    imputed_ff = pd.DataFrame(imputed).ffill(axis=1).bfill(axis=1).fillna(0.0).values
    still_nan = np.isnan(imputed)
    imputed[still_nan] = imputed_ff[still_nan]
    return imputed.astype(np.float32)

# ===========================================================================
# Run the 3 imputers
# ===========================================================================
traccia_a = []

# Combined stockout mask for eval: original + MNAR
stock_combined_eval = np.maximum(stock_flat, mnar_mask_flat)
# Zero out MNAR positions in sales (hide them from imputer)
sales_hidden = np.where(mnar_mask_flat == 1, 0, sales_flat)

# --- IMPUTER 1: FORWARD FILL ---
print('\n' + '='*72)
print('  IMPUTER 1 — FORWARD FILL')
print('='*72)
t0 = time.time()
imputed_eval = forward_fill_vec(sales_hidden, stock_combined_eval)
print(f'  Eval computed in {time.time()-t0:.1f}s')
ta = eval_mnar(imputed_eval, 'Forward Fill (MNAR eval)')
traccia_a.append({'imputer': 'Forward Fill', **ta})

t0 = time.time()
imputed_full = forward_fill_vec(sales_flat, stock_flat)
print(f'  Full computed in {time.time()-t0:.1f}s')
save_completed(imputed_full, 'forward_fill')

# --- IMPUTER 2: SEASONAL NAIVE ---
print('\n' + '='*72)
print('  IMPUTER 2 — SEASONAL NAIVE')
print('='*72)
t0 = time.time()
imputed_eval = seasonal_naive_vec(sales_hidden, stock_flat, stock_combined_eval, L_FULL)
print(f'  Eval computed in {time.time()-t0:.1f}s')
ta = eval_mnar(imputed_eval, 'Seasonal Naive (MNAR eval)')
traccia_a.append({'imputer': 'Seasonal Naive', **ta})

t0 = time.time()
imputed_full = seasonal_naive_vec(sales_flat, stock_flat, stock_flat, L_FULL)
print(f'  Full computed in {time.time()-t0:.1f}s')
save_completed(imputed_full, 'seasonal_naive')

# --- IMPUTER 3: LINEAR INTERPOLATION ---
print('\n' + '='*72)
print('  IMPUTER 3 — LINEAR INTERPOLATION')
print('='*72)
t0 = time.time()
imputed_eval = linear_interp_vec(sales_hidden, stock_combined_eval)
print(f'  Eval computed in {time.time()-t0:.1f}s')
ta = eval_mnar(imputed_eval, 'Linear Interp (MNAR eval)')
traccia_a.append({'imputer': 'Linear Interp', **ta})

t0 = time.time()
imputed_full = linear_interp_vec(sales_flat, stock_flat)
print(f'  Full computed in {time.time()-t0:.1f}s')
save_completed(imputed_full, 'linear_interp')

# ===========================================================================
# Save Traccia A
# ===========================================================================
print('\n' + '='*72)
print('  RIEPILOGO Traccia A (3 classici)')
print('='*72)
df_ta = pd.DataFrame(traccia_a)
df_ta.to_parquet(os.path.join(RESULTS_DIR, 'traccia_a_classic.parquet'), index=False)
print()
print(df_ta.to_string(index=False))
print(f'\n  Salvato: traccia_a_classic.parquet')

print('\n' + '='*72)
print('  DONE — 14_fase_b1_imputation_classic.py')
print('='*72)
