"""
16b_saits_predict.py — Continua da SAITS trainato: predict + eval MNAR + completed_sales
==========================================================================================
Carica il modello SAITS salvato e fa predict con batch piccolo per evitare OOM.
"""
import os, gc, time, functools, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

import torch
from pypots.imputation import SAITS
from pypots.optim import Adam

H_START, H_END = 6, 23; N_HOURS = H_END - H_START
WINDOW_DAYS = 30; N_WINDOWS = 3; N_STEPS = WINDOW_DAYS * N_HOURS

D_MODEL = 32; N_HEADS = 4; N_LAYERS = 2; D_FFN = 64
DROPOUT = 0.1; BATCH_SIZE = 16
PREDICT_BATCH = 256  # MUCH smaller for predict to avoid OOM

MODEL_PATH = os.path.join(RESULTS_DIR, 'pypots_saits_val/20260429_T083624/SAITS.pypots')

DEVICE = 'mps' if torch.backends.mps.is_available() else (
    'cuda' if torch.cuda.is_available() else 'cpu')

print('=' * 72)
print('  SAITS predict (continuation)')
print('=' * 72)
print(f'  Loading model: {MODEL_PATH}')

# Reload data
print('\n1. Caricamento dati...')
df = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
sales_17 = np.array(df['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
stock_17 = np.array(df['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]
n_rows = len(df)
n_series = n_rows // 90

sales_win = sales_17.reshape(n_series * N_WINDOWS, WINDOW_DAYS, N_HOURS)
stock_win = stock_17.reshape(n_series * N_WINDOWS, WINDOW_DAYS, N_HOURS)
n_samples = sales_win.shape[0]
sales_nan = np.where(stock_win == 1, np.nan, sales_win)

covs_cols = ['discount', 'holiday_flag', 'precpt', 'avg_temperature']
covs = df[covs_cols].values.astype(np.float32)
covs_win = covs.reshape(n_series * N_WINDOWS, WINDOW_DAYS, len(covs_cols))
covs_max = covs_win.max(axis=1, keepdims=True) + 0.1
covs_norm = covs_win / covs_max
hour_pos = np.arange(N_HOURS, dtype=np.float32)[None, None, :] / (N_HOURS - 1)
hour_pos = np.broadcast_to(hour_pos, (n_samples, WINDOW_DAYS, N_HOURS))

sales_flat = sales_nan.reshape(n_samples, N_STEPS, 1)
covs_broadcast = np.broadcast_to(covs_norm[:, :, None, :],
                                  (n_samples, WINDOW_DAYS, N_HOURS, len(covs_cols)))
covs_flat = covs_broadcast.reshape(n_samples, N_STEPS, len(covs_cols))
hour_flat = hour_pos.reshape(n_samples, N_STEPS, 1)

X_all = np.concatenate([sales_flat, covs_flat, hour_flat], axis=-1)
n_features = X_all.shape[-1]
sales_origin = sales_win.copy()
del covs, covs_win, covs_norm, hour_pos, sales_flat, covs_broadcast, covs_flat, hour_flat
gc.collect()

# Load saved model
print('\n2. Loading saved SAITS model...')
optimizer = Adam(lr=1e-3, weight_decay=1e-5)
model = SAITS(
    n_steps=N_STEPS, n_features=n_features,
    n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS,
    d_k=D_MODEL // N_HEADS, d_v=D_MODEL // N_HEADS, d_ffn=D_FFN,
    dropout=DROPOUT, attn_dropout=DROPOUT,
    diagonal_attention_mask=True,
    ORT_weight=1, MIT_weight=1,
    batch_size=BATCH_SIZE, epochs=10, patience=5,
    optimizer=optimizer, device=DEVICE, verbose=False)
model.load(MODEL_PATH)
print(f'  Model loaded.')

# Predict on all data with small batch
print(f'\n3. Predict {n_samples:,} samples (batch={PREDICT_BATCH})...')
imputed_flat = np.zeros((n_samples, N_STEPS), dtype=np.float32)

t0 = time.time()
for start in range(0, n_samples, PREDICT_BATCH):
    end = min(start + PREDICT_BATCH, n_samples)
    if (start // PREDICT_BATCH) % 50 == 0:
        elapsed = time.time() - t0
        eta = elapsed / max(end, 1) * (n_samples - end)
        print(f'  {start:,}/{n_samples:,}  elapsed={elapsed:.0f}s eta={eta:.0f}s')
    res = model.predict({'X': X_all[start:end]})
    imp = res['imputation']
    if len(imp.shape) == 4: imp = imp.mean(axis=1)
    imputed_flat[start:end] = np.clip(imp[:, :, 0], 0, None)
    del res, imp
    gc.collect()
    if DEVICE == 'mps': torch.mps.empty_cache()
print(f'  Done in {time.time()-t0:.0f}s')

imputed_sales = imputed_flat.reshape(n_samples, WINDOW_DAYS, N_HOURS)
print(f'  Mean imputed: {imputed_sales.mean():.4f}')

# Eval MNAR
print(f'\n4. Eval MNAR...')
masks_val_df = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_val.parquet'))
masks_val_df = masks_val_df[(masks_val_df['hour'] >= H_START) & (masks_val_df['hour'] < H_END)].reset_index(drop=True)
print(f'  MNAR masks: {len(masks_val_df):,}')

df['row_idx'] = np.arange(n_rows)
rl = df.set_index(['store_id', 'product_id', 'dt'])['row_idx']

X_mnar = X_all.copy()
n_added = 0
for _, r in masks_val_df.iterrows():
    key = (r['store_id'], r['product_id'], r['dt'])
    if key in rl.index:
        row_idx = rl[key]
        if isinstance(row_idx, pd.Series): row_idx = row_idx.iloc[0]
        si = row_idx // 90
        di = row_idx % 90
        wi = di // WINDOW_DAYS
        dw = di % WINDOW_DAYS
        samp = si * N_WINDOWS + wi
        hi = int(r['hour']) - H_START
        step = dw * N_HOURS + hi
        X_mnar[samp, step, 0] = np.nan
        n_added += 1
print(f'  MNAR NaN added: {n_added:,}')

t0 = time.time()
imputed_mnar_flat = np.zeros((n_samples, N_STEPS), dtype=np.float32)
for start in range(0, n_samples, PREDICT_BATCH):
    end = min(start + PREDICT_BATCH, n_samples)
    if (start // PREDICT_BATCH) % 50 == 0:
        elapsed = time.time() - t0
        eta = elapsed / max(end, 1) * (n_samples - end)
        print(f'  {start:,}/{n_samples:,}  elapsed={elapsed:.0f}s eta={eta:.0f}s')
    res = model.predict({'X': X_mnar[start:end]})
    imp = res['imputation']
    if len(imp.shape) == 4: imp = imp.mean(axis=1)
    imputed_mnar_flat[start:end] = np.clip(imp[:, :, 0], 0, None)
    del res, imp
    gc.collect()
    if DEVICE == 'mps': torch.mps.empty_cache()
print(f'  Done in {time.time()-t0:.0f}s')

imputed_mnar = imputed_mnar_flat.reshape(n_samples, WINDOW_DAYS, N_HOURS)

preds_mnar = np.zeros(len(masks_val_df), dtype=np.float64)
gt = masks_val_df['ground_truth'].values.astype(np.float64)
for i in range(len(masks_val_df)):
    r = masks_val_df.iloc[i]
    key = (r['store_id'], r['product_id'], r['dt'])
    if key in rl.index:
        row_idx = rl[key]
        if isinstance(row_idx, pd.Series): row_idx = row_idx.iloc[0]
        si = row_idx // 90; di = row_idx % 90
        wi = di // WINDOW_DAYS; dw = di % WINDOW_DAYS
        samp = si * N_WINDOWS + wi
        hi = int(r['hour']) - H_START
        preds_mnar[i] = imputed_mnar[samp, dw, hi]

sao = np.abs(gt).sum()
wape = np.abs(preds_mnar - gt).sum() / sao if sao > 0 else np.nan
wpe = (preds_mnar - gt).sum() / gt.sum() if gt.sum() != 0 else np.nan
print(f'\n  SAITS: WAPE_recovery={wape:.4f}, WPE_recovery={wpe:.4f}')

ta_df = pd.DataFrame([{'imputer': 'SAITS', 'wape_recovery': wape, 'wpe_recovery': wpe}])
ta_df.to_parquet(os.path.join(RESULTS_DIR, 'traccia_a_saits.parquet'), index=False)

# Save completed_sales
print(f'\n5. Salvataggio completed_sales...')
imputed_rows = imputed_sales.reshape(n_rows, N_HOURS)
completed = sales_17.copy()
so_mask = stock_17 == 1
completed[so_mask] = np.clip(imputed_rows[so_mask], 0, None)

df_out = df[['store_id', 'product_id', 'dt']].copy()
df_out['dt_parsed'] = pd.to_datetime(df_out['dt'])
all_dates = sorted(df_out['dt_parsed'].unique())
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}
df_out['day_num'] = df_out['dt_parsed'].map(date_to_day)
df_out['dow'] = df_out['dt_parsed'].dt.dayofweek
df_out['hours_sale'] = list(completed)
df_out['hours_stock_status'] = list(stock_17)
df_out.drop(columns=['dt_parsed', 'row_idx'], inplace=True, errors='ignore')

out_path = os.path.join(COMPLETED_DIR, 'saits.parquet')
df_out.to_parquet(out_path, index=False)
print(f'  Salvato: {out_path}')
print(f'  Media imputata (stockout): {completed[so_mask].mean():.4f}')
print(f'  Media S_obs (in-stock):    {sales_17[~so_mask].mean():.4f}')

print('\n' + '=' * 72)
print('  DONE — SAITS predict (continuation)')
print('=' * 72)
