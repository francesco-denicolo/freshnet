"""
06b_eval_mnar_sota.py — Valutazione MNAR per imputer SOTA
==========================================================
Per ogni modello SOTA già trainato, crea un dataset con NaN aggiuntivi
per le maschere MNAR, fa predict, e confronta con ground truth.

Usage: freshnet/bin/python notebooks_622/06b_eval_mnar_sota.py <model_name>
"""
import sys, os, gc, time, functools, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

import torch
from pypots.imputation import SAITS, TimesNet, iTransformer, DLinear
from pypots.optim import Adam

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
WINDOW_DAYS = 30; N_WINDOWS = 3; N_STEPS = WINDOW_DAYS * N_HOURS

if len(sys.argv) >= 2 and sys.argv[1] in ['SAITS', 'iTransformer']:
    DEVICE = 'cpu'
else:
    DEVICE = 'mps' if torch.backends.mps.is_available() else (
        'cuda' if torch.cuda.is_available() else 'cpu')

HP = {'epochs': 5, 'batch_size': 32, 'patience': 5,
      'n_layers': 2, 'd_model': 64, 'd_ffn': 32,
      'n_heads': 4, 'd_k': 16, 'd_v': 16,
      'dropout': 0., 'attn_dropout': 0.,
      'lr': 0.001, 'weight_decay': 1e-5}

MODEL_NAME = sys.argv[1]
print(f'=== VALUTAZIONE MNAR: {MODEL_NAME} ===')

# ===========================================================================
# 1. Load data and build X with MNAR masks as additional NaN
# ===========================================================================
print('\n1. Caricamento dati...')
df = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df = df.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)

sales_17 = np.array(df['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
stock_17 = np.array(df['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]

n_rows = len(df)
n_series = n_rows // 90

# Reshape to windows
sales_win = sales_17.reshape(n_series * N_WINDOWS, WINDOW_DAYS, N_HOURS)
stock_win = stock_17.reshape(n_series * N_WINDOWS, WINDOW_DAYS, N_HOURS)
n_samples = sales_win.shape[0]

# Set stockout to NaN
sales_nan = np.where(stock_win == 1, np.nan, sales_win)

# Load MNAR masks
masks_val = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_val.parquet'))
masks_val = masks_val[(masks_val['hour'] >= H_START) & (masks_val['hour'] < H_END)].reset_index(drop=True)
print(f'  MNAR masks (6-22): {len(masks_val):,}')

# Build lookup: (store_id, product_id, dt) -> (row_idx in df)
df['row_idx'] = np.arange(n_rows)
row_lookup = df.set_index(['store_id', 'product_id', 'dt'])['row_idx']

# Add MNAR masks as additional NaN in sales_nan
# Only for days 84-90 (validation period)
print('  Adding MNAR masks as NaN...')
n_mnar_added = 0
for _, r in masks_val.iterrows():
    key = (r['store_id'], r['product_id'], r['dt'])
    if key in row_lookup.index:
        row_idx = row_lookup[key]
        if isinstance(row_idx, pd.Series):
            row_idx = row_idx.iloc[0]
        # Convert row_idx to (sample_idx, day_in_window, hour_idx)
        series_idx = row_idx // 90
        day_in_series = row_idx % 90
        window_in_series = day_in_series // WINDOW_DAYS
        day_in_window = day_in_series % WINDOW_DAYS
        sample_idx = series_idx * N_WINDOWS + window_in_series
        hour_idx = int(r['hour']) - H_START

        sales_nan[sample_idx, day_in_window, hour_idx] = np.nan
        n_mnar_added += 1

print(f'  MNAR NaN aggiunti: {n_mnar_added:,}')
print(f'  NaN totali: {np.isnan(sales_nan).sum():,} (stockout + MNAR)')

# Build features (same as training)
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

X_mnar = np.concatenate([sales_flat, covs_flat, hour_flat], axis=-1)
n_features = X_mnar.shape[-1]
print(f'  X_mnar shape: {X_mnar.shape}')

del df, covs, covs_win, covs_norm, hour_pos, sales_flat, covs_broadcast, covs_flat, hour_flat
gc.collect()

# ===========================================================================
# 2. Load trained model and predict
# ===========================================================================
print(f'\n2. Creazione e training modello {MODEL_NAME}...')
# We need to retrain because PyPOTS doesn't easily reload from file
# But training is fast (5 epochs, ~5 min)

# Build training X (with only stockout NaN, no MNAR)
sales_nan_train = np.where(stock_win == 1, np.nan, sales_win)
covs_raw = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
covs_raw = covs_raw.sort_values(['store_id', 'product_id', 'dt'])
covs_train = covs_raw[['discount', 'holiday_flag', 'precpt', 'avg_temperature']].values.astype(np.float32)
covs_train_win = covs_train.reshape(n_series * N_WINDOWS, WINDOW_DAYS, 4)
covs_train_max = covs_train_win.max(axis=1, keepdims=True) + 0.1
covs_train_norm = covs_train_win / covs_train_max
hp_train = np.broadcast_to(np.arange(N_HOURS, dtype=np.float32)[None, None, :] / (N_HOURS - 1),
                            (n_samples, WINDOW_DAYS, N_HOURS))

sf = sales_nan_train.reshape(n_samples, N_STEPS, 1)
cf = np.broadcast_to(covs_train_norm[:, :, None, :],
                      (n_samples, WINDOW_DAYS, N_HOURS, 4)).reshape(n_samples, N_STEPS, 4)
hf = hp_train.reshape(n_samples, N_STEPS, 1)
X_train = np.concatenate([sf, cf, hf], axis=-1)

del covs_raw, covs_train, covs_train_win, covs_train_norm, hp_train, sf, cf, hf, sales_nan_train
gc.collect()

optimizer = Adam(lr=HP['lr'], weight_decay=HP['weight_decay'])
saving_path = os.path.join(RESULTS_DIR, f'pypots_{MODEL_NAME.lower()}_mnar')

if MODEL_NAME == 'SAITS':
    model = SAITS(n_steps=N_STEPS, n_features=n_features,
                  n_layers=HP['n_layers'], d_model=HP['d_model'], d_ffn=HP['d_ffn'],
                  n_heads=HP['n_heads'], d_k=HP['d_k'], d_v=HP['d_v'],
                  dropout=HP['dropout'], attn_dropout=HP['attn_dropout'],
                  diagonal_attention_mask=True, ORT_weight=1, MIT_weight=1,
                  batch_size=HP['batch_size'], epochs=HP['epochs'],
                  patience=HP['patience'], optimizer=optimizer,
                  device=DEVICE, saving_path=saving_path, verbose=True)
elif MODEL_NAME == 'TimesNet':
    model = TimesNet(n_steps=N_STEPS, n_features=n_features,
                     n_layers=HP['n_layers'], top_k=7,
                     d_model=HP['d_model'], d_ffn=HP['d_ffn'], n_kernels=5,
                     dropout=HP['dropout'], apply_nonstationary_norm=True,
                     epochs=HP['epochs'], batch_size=HP['batch_size'],
                     patience=HP['patience'], optimizer=optimizer,
                     device=DEVICE, saving_path=saving_path, verbose=True)
elif MODEL_NAME == 'iTransformer':
    model = iTransformer(n_steps=N_STEPS, n_features=n_features,
                         n_layers=HP['n_layers'], d_model=HP['d_model'], d_ffn=HP['d_ffn'],
                         n_heads=HP['n_heads'], d_k=HP['d_k'], d_v=HP['d_v'],
                         dropout=HP['dropout'], attn_dropout=HP['attn_dropout'],
                         ORT_weight=1, MIT_weight=1,
                         batch_size=HP['batch_size'], epochs=HP['epochs'],
                         patience=HP['patience'], optimizer=optimizer,
                         device=DEVICE, saving_path=saving_path, verbose=True)
elif MODEL_NAME == 'DLinear':
    model = DLinear(n_steps=N_STEPS, n_features=n_features,
                    moving_avg_window_size=N_HOURS // 2 * 2 + 1,
                    individual=False, d_model=HP['d_model'],
                    ORT_weight=1, MIT_weight=1,
                    batch_size=HP['batch_size'], epochs=HP['epochs'],
                    patience=HP['patience'], optimizer=optimizer,
                    device=DEVICE, saving_path=saving_path, verbose=True)

# Train on X_train (only stockout NaN)
print('  Training...')
t0 = time.time()
model.fit({'X': X_train})
print(f'  Training time: {time.time()-t0:.0f}s')

del X_train
gc.collect()

# ===========================================================================
# 3. Predict on X_mnar (stockout NaN + MNAR NaN)
# ===========================================================================
print(f'\n3. Predict su X_mnar (con maschere MNAR come NaN)...')
PREDICT_BATCH = 10000
imputed_flat = np.zeros((n_samples, N_STEPS), dtype=np.float32)

for start in range(0, n_samples, PREDICT_BATCH):
    end = min(start + PREDICT_BATCH, n_samples)
    print(f'  {start:,}-{end:,} / {n_samples:,}...')
    res = model.predict({'X': X_mnar[start:end]})
    imp = res['imputation']
    if len(imp.shape) == 4:
        imp = imp.mean(axis=1)
    imputed_flat[start:end] = np.clip(imp[:, :, 0], 0, None)
    del res, imp
    gc.collect()
    if DEVICE == 'mps':
        torch.mps.empty_cache()

imputed_win = imputed_flat.reshape(n_samples, WINDOW_DAYS, N_HOURS)

# ===========================================================================
# 4. Evaluate on MNAR masks
# ===========================================================================
print(f'\n4. Valutazione MNAR...')

# For each MNAR mask entry, get the imputed value
preds_mnar = np.zeros(len(masks_val), dtype=np.float64)
gt = masks_val['ground_truth'].values.astype(np.float64)

row_lookup2 = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
row_lookup2 = row_lookup2.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
row_lookup2['row_idx'] = np.arange(len(row_lookup2))
rl = row_lookup2.set_index(['store_id', 'product_id', 'dt'])['row_idx']

for i in range(len(masks_val)):
    r = masks_val.iloc[i]
    key = (r['store_id'], r['product_id'], r['dt'])
    if key in rl.index:
        row_idx = rl[key]
        if isinstance(row_idx, pd.Series):
            row_idx = row_idx.iloc[0]
        series_idx = row_idx // 90
        day_in_series = row_idx % 90
        window_in_series = day_in_series // WINDOW_DAYS
        day_in_window = day_in_series % WINDOW_DAYS
        sample_idx = series_idx * N_WINDOWS + window_in_series
        hour_idx = int(r['hour']) - H_START
        preds_mnar[i] = imputed_win[sample_idx, day_in_window, hour_idx]

sao = np.abs(gt).sum()
wape = np.abs(preds_mnar - gt).sum() / sao if sao > 0 else np.nan
wpe = (preds_mnar - gt).sum() / gt.sum() if gt.sum() != 0 else np.nan

print(f'\n  {MODEL_NAME}: WAPE_recovery={wape:.4f}, WPE_recovery={wpe:.4f}')

# Save
ta_df = pd.DataFrame([{'imputer': MODEL_NAME, 'wape_recovery': wape, 'wpe_recovery': wpe}])
ta_df.to_parquet(os.path.join(RESULTS_DIR, f'traccia_a_{MODEL_NAME.lower()}.parquet'), index=False)
print(f'  Salvato: traccia_a_{MODEL_NAME.lower()}.parquet')

print(f'\n=== DONE — {MODEL_NAME} MNAR eval ===')
