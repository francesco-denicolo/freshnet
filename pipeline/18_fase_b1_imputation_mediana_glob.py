"""
18_fase_b1_imputation_mediana_glob.py — Mediana globale (ore 6-22)
====================================================================
Imputer naive aggregato: mediana per (store, product, hour) su ore in-stock.
Differenza dalla Mediana condizionata: NON condizionata su day-of-week.
Completa il design 2x2:
  Media   × Globale       (esistente: media_glob)
  Media   × Condizionata  (esistente: media_cond)
  Mediana × Globale       ← NUOVO
  Mediana × Condizionata  (esistente: mediana_cond)
"""
import sys, os, time, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
os.makedirs(COMPLETED_DIR, exist_ok=True)

SEED = 42; np.random.seed(SEED)
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
HOURS_RANGE = np.arange(H_START, H_END, dtype=np.int32)

print('=' * 72)
print('  IMPUTER 11 — MEDIANA GLOBALE (ore 6-22)')
print('=' * 72)

print('\n1. Caricamento dati...')
df = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df['dt_parsed'] = pd.to_datetime(df['dt'])
all_dates = sorted(df['dt_parsed'].unique())
date_to_day = {d: i+1 for i, d in enumerate(all_dates)}
df['day_num'] = df['dt_parsed'].map(date_to_day)
df['dow'] = df['dt_parsed'].dt.dayofweek

sales_all = np.array(df['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
stock_all = np.array(df['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]
n_rows = len(df); n_hourly = n_rows * N_HOURS

store_flat = np.repeat(df['store_id'].values, N_HOURS)
product_flat = np.repeat(df['product_id'].values, N_HOURS)
day_flat = np.repeat(df['day_num'].values, N_HOURS)
hours_flat = np.tile(HOURS_RANGE, n_rows)
sale_flat = sales_all.ravel().astype(np.float32)
stock_flat = stock_all.ravel().astype(np.int8)
print(f'  Rows: {n_rows:,}, Hourly: {n_hourly:,}')

print('  Loading MNAR masks...')
masks_val = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_val.parquet'))
masks_val = masks_val[(masks_val['hour'] >= H_START) & (masks_val['hour'] < H_END)].reset_index(drop=True)
print(f'  MNAR masks: {len(masks_val):,}')

stockout_idx = np.where(stock_flat == 1)[0]
train_mask_83 = (day_flat <= 83) & (stock_flat == 0)
train_mask_90 = (day_flat <= 90) & (stock_flat == 0)

# ===========================================================================
# Compute median for (store, product, hour) — global (no dow)
# ===========================================================================
print('\n2. Computing Mediana globale (gg 1-83)...')
t0 = time.time()
df_83 = pd.DataFrame({
    'store_id': store_flat[train_mask_83],
    'product_id': product_flat[train_mask_83],
    'hour': hours_flat[train_mask_83],
    'sale': sale_flat[train_mask_83]
})
glob_med_83 = df_83.groupby(['store_id','product_id','hour'])['sale'].median()
del df_83
print(f'  Train 83: {len(glob_med_83):,} chiavi, {time.time()-t0:.1f}s')

# ===========================================================================
# MNAR eval (Traccia A)
# ===========================================================================
print('\n3. Valutazione MNAR...')
preds = np.zeros(len(masks_val), dtype=np.float32)
for i in range(len(masks_val)):
    r = masks_val.iloc[i]
    k = (r['store_id'], r['product_id'], r['hour'])
    if k in glob_med_83:
        preds[i] = glob_med_83[k]

gt = masks_val['ground_truth'].values.astype(np.float64)
preds = preds.astype(np.float64)
sao = np.abs(gt).sum(); so = gt.sum()
sae = np.abs(preds-gt).sum(); se = (preds-gt).sum()
wape = sae/sao if sao>0 else np.nan
wpe = se/so if so!=0 else np.nan
print(f'  Mediana globale (83): WAPE_recovery={wape:.4f}, WPE_recovery={wpe:.4f}')

ta_df = pd.DataFrame([{'imputer': 'Mediana globale', 'wape_recovery': wape, 'wpe_recovery': wpe}])
ta_df.to_parquet(os.path.join(RESULTS_DIR, 'traccia_a_mediana_glob.parquet'), index=False)

# ===========================================================================
# Retrain on gg 1-90 + completed_sales
# ===========================================================================
print('\n4. Retrain on gg 1-90 + completed_sales...')
t0 = time.time()
df_90 = pd.DataFrame({
    'store_id': store_flat[train_mask_90],
    'product_id': product_flat[train_mask_90],
    'hour': hours_flat[train_mask_90],
    'sale': sale_flat[train_mask_90]
})
glob_med_90 = df_90.groupby(['store_id','product_id','hour'])['sale'].median()
del df_90
print(f'  Train 90: {len(glob_med_90):,} chiavi, {time.time()-t0:.1f}s')

# Apply to stockout positions
imp = np.zeros(n_hourly, dtype=np.float32)
for idx in stockout_idx:
    k = (store_flat[idx], product_flat[idx], hours_flat[idx])
    if k in glob_med_90:
        imp[idx] = glob_med_90[k]

# Save completed_sales
completed = sale_flat.copy()
so_mask = stock_flat == 1
completed[so_mask] = np.clip(imp[so_mask], 0, None)
completed_17 = completed.reshape(n_rows, N_HOURS)

df_out = df[['store_id','product_id','dt','day_num','dow']].copy()
df_out['hours_sale'] = list(completed_17)
df_out['hours_stock_status'] = list(stock_all)
out_path = os.path.join(COMPLETED_DIR, 'mediana_glob.parquet')
df_out.to_parquet(out_path, index=False)
print(f'  Salvato: {out_path}')

imp_vals = completed[so_mask]
print(f'  Media imputata (stockout): {imp_vals.mean():.4f}')
print(f'  Media S_obs (in-stock):    {sale_flat[~so_mask].mean():.4f}')

print('\n' + '=' * 72)
print('  DONE — Mediana globale')
print('=' * 72)
