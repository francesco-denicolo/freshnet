"""
salvage_saits_predict.py — load the already-trained SAITS hi-cap model and run
ONLY the predict + MNAR-eval + save steps that crashed with an MPS OOM during the
overnight run. Predict on CPU (no 18 GB MPS cap) with a small batch. Reuses the
exact data construction of cap_sweep_imputer.py.
"""
import os, gc, time, functools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

import torch
from pypots.imputation import SAITS
from pypots.optim import Adam

SAITS_PATH = os.path.join(os.path.dirname(__file__), 'results',
                          'pypots_saits_hicap_val', '20260629_T224029', 'SAITS.pypots')
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
WINDOW_DAYS = 30; N_WINDOWS = 3; N_STEPS = WINDOW_DAYS * N_HOURS
OUT_NAME = 'saits_hicap'
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
PREDICT_BATCH = 128   # small: SAITS attention memory ~ batch * heads * n_steps^2

# hi-cap SAITS config (must match the trained model for load())
CFG = dict(d_model=128, n_heads=8, n_layers=3, d_ffn=256, dropout=0.1,
           batch_size=16, max_epochs=40, patience=6, lr=1e-3)

print('=' * 72)
print('  SALVAGE: SAITS hi-cap predict-only (CPU)')
print(f'  Loading: {SAITS_PATH}')
print('=' * 72)

# --- data construction (identical to cap_sweep_imputer.py) ---
print('\n1. Caricamento dati...')
df = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
sales_17 = np.array(df['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
stock_17 = np.array(df['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]
n_rows = len(df); n_series = n_rows // 90
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
print(f'  Samples: {n_samples:,}, Steps: {N_STEPS}, Features: {n_features}')

# --- load trained model ---
print('\n2. Costruzione + load modello SAITS (CPU)...')
optimizer = Adam(lr=CFG['lr'], weight_decay=1e-5)
model = SAITS(
    n_steps=N_STEPS, n_features=n_features,
    n_layers=CFG['n_layers'], d_model=CFG['d_model'], n_heads=CFG['n_heads'],
    d_k=CFG['d_model'] // CFG['n_heads'], d_v=CFG['d_model'] // CFG['n_heads'],
    d_ffn=CFG['d_ffn'], dropout=CFG['dropout'], attn_dropout=CFG['dropout'],
    ORT_weight=1, MIT_weight=1,
    batch_size=CFG['batch_size'], epochs=CFG['max_epochs'],
    patience=CFG['patience'], optimizer=optimizer,
    device=DEVICE, saving_path=None, verbose=False)
model.load(SAITS_PATH)
print(f'  Params: {sum(p.numel() for p in model.model.parameters()):,}')

def predict_all(X):
    out = np.zeros((X.shape[0], N_STEPS), dtype=np.float32)
    for start in range(0, X.shape[0], PREDICT_BATCH):
        end = min(start + PREDICT_BATCH, X.shape[0])
        res = model.predict({'X': X[start:end]})
        imp = res['imputation']
        if len(imp.shape) == 4:
            imp = imp.mean(axis=1)
        out[start:end] = np.clip(imp[:, :, 0], 0, None)
        del res, imp; gc.collect()
        if DEVICE == 'mps':
            torch.mps.empty_cache()
        if start % (PREDICT_BATCH * 50) == 0:
            print(f'    {start:,}/{X.shape[0]:,}')
    return out

print('\n3. Predict su tutti i samples (CPU)...')
t0 = time.time()
imputed_flat = predict_all(X_all)
imputed_sales = imputed_flat.reshape(n_samples, WINDOW_DAYS, N_HOURS)
print(f'  Predict time: {time.time()-t0:.0f}s, mean imputed: {imputed_sales.mean():.4f}')

# --- MNAR eval (Track A) ---
print('\n4. Valutazione MNAR...')
masks_val_df = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_val.parquet'))
masks_val_df = masks_val_df[(masks_val_df['hour'] >= H_START) &
                            (masks_val_df['hour'] < H_END)].reset_index(drop=True)
df['row_idx'] = np.arange(n_rows)
rl = df.set_index(['store_id', 'product_id', 'dt'])['row_idx']
X_mnar = X_all.copy()
for _, r in masks_val_df.iterrows():
    key = (r['store_id'], r['product_id'], r['dt'])
    if key in rl.index:
        row_idx = rl[key]
        if isinstance(row_idx, pd.Series): row_idx = row_idx.iloc[0]
        si = row_idx // 90; di = row_idx % 90
        wi = di // WINDOW_DAYS; dw = di % WINDOW_DAYS
        samp = si * N_WINDOWS + wi
        step = dw * N_HOURS + (int(r['hour']) - H_START)
        X_mnar[samp, step, 0] = np.nan
imputed_mnar = predict_all(X_mnar).reshape(n_samples, WINDOW_DAYS, N_HOURS)
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
        preds_mnar[i] = imputed_mnar[samp, dw, int(r['hour']) - H_START]
sao = np.abs(gt).sum()
wape = np.abs(preds_mnar - gt).sum() / sao if sao > 0 else np.nan
wpe = (preds_mnar - gt).sum() / gt.sum() if gt.sum() != 0 else np.nan
print(f'  SAITS (hi-cap): WAPE_recovery={wape:.4f}, WPE_recovery={wpe:.4f}')
pd.DataFrame([{'imputer': OUT_NAME, 'wape_recovery': wape, 'wpe_recovery': wpe}]).to_parquet(
    os.path.join(RESULTS_DIR, f'traccia_a_{OUT_NAME}.parquet'), index=False)

# --- save completed_sales ---
print('\n5. Salvataggio completed_sales...')
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
df_out.to_parquet(os.path.join(COMPLETED_DIR, f'{OUT_NAME}.parquet'), index=False)
print(f'  Salvato: {COMPLETED_DIR}/{OUT_NAME}.parquet')
print('\nDONE — SAITS hi-cap salvaged')
