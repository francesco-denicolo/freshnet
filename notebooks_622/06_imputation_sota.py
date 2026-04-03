"""
06_imputation_sota.py — Fase B1: Imputation SOTA via PyPOTS (ore 6-22)
=======================================================================
4 modelli SOTA: SAITS, TimesNet, iTransformer, DLinear.
Approccio dal paper baseline (FreshRetailNet-50K):
  - 50K serie × 3 finestre da 30 giorni = 150K samples
  - Per ogni sample: 30 × 17 = 510 time steps, 6 features
  - Features: vendite (NaN se stockout) + 4 covariate + posizione oraria
  - Il modello imputa i NaN

Procedura per ciascuno:
  1. Prepara dati con NaN per stockout
  2. Fit con ORT+MIT
  3. Predict → imputed values
  4. Valutazione su maschere MNAR
  5. Salva completed_sales

Eseguire con: PYTHONUNBUFFERED=1 freshnet/bin/python notebooks_622/06_imputation_sota.py <model_name>
  model_name: SAITS | TimesNet | iTransformer | DLinear
"""
import sys, os, gc, time, functools, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import torch
from pypots.imputation import SAITS, TimesNet, iTransformer, DLinear
from pypots.optim import Adam

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(COMPLETED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

H_START, H_END = 6, 23
N_HOURS = H_END - H_START  # 17
WINDOW_DAYS = 30
N_STEPS = WINDOW_DAYS * N_HOURS  # 510
N_WINDOWS = 3  # 90 days / 30

# Force CPU for attention-based models (SAITS, iTransformer) — OOM on MPS
if len(sys.argv) >= 2 and sys.argv[1] in ['SAITS', 'iTransformer']:
    DEVICE = 'cpu'
else:
    DEVICE = 'mps' if torch.backends.mps.is_available() else (
        'cuda' if torch.cuda.is_available() else 'cpu')

# HP from baseline paper
HP = {
    'epochs': 5, 'batch_size': 32, 'patience': 5,
    'n_layers': 2, 'd_model': 64, 'd_ffn': 32,
    'n_heads': 4, 'd_k': 16, 'd_v': 16,
    'dropout': 0., 'attn_dropout': 0.,
    'lr': 0.001, 'weight_decay': 1e-5,
}
# DLinear can handle larger batch
if len(sys.argv) >= 2 and sys.argv[1] == 'DLinear':
    HP['batch_size'] = 128

# Get model name from command line
if len(sys.argv) < 2:
    print('Usage: python 06_imputation_sota.py <SAITS|TimesNet|iTransformer|DLinear>')
    sys.exit(1)

MODEL_NAME = sys.argv[1]
assert MODEL_NAME in ['SAITS', 'TimesNet', 'iTransformer', 'DLinear'], \
    f'Unknown model: {MODEL_NAME}'

out_cs = os.path.join(COMPLETED_DIR, f'{MODEL_NAME.lower()}.parquet')
if os.path.exists(out_cs):
    print(f'SKIP: {out_cs} exists')
    sys.exit(0)

print(f'=== IMPUTER SOTA: {MODEL_NAME} (ore 6-22) ===')
print(f'Device: {DEVICE}')

# ===========================================================================
# 1. Load and prepare data
# ===========================================================================
print('\n1. Caricamento dati...')
df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_train_hf = df_train_hf.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)

sales_24 = np.array(df_train_hf['hours_sale'].tolist(), dtype=np.float32)
stock_24 = np.array(df_train_hf['hours_stock_status'].tolist(), dtype=np.int8)

# Slice to 6-22
sales_17 = sales_24[:, H_START:H_END]  # (4.5M, 17)
stock_17 = stock_24[:, H_START:H_END]

n_rows = len(df_train_hf)
n_series = n_rows // 90
assert n_rows == n_series * 90, f'Expected 90 days per series, got {n_rows}/{n_series}'
print(f'  {n_series:,} serie × 90 giorni = {n_rows:,} righe')

# Reshape to (n_series × 3, 30, 17)
sales_win = sales_17.reshape(n_series * N_WINDOWS, WINDOW_DAYS, N_HOURS)
stock_win = stock_17.reshape(n_series * N_WINDOWS, WINDOW_DAYS, N_HOURS)
n_samples = sales_win.shape[0]
print(f'  Finestre: {n_samples:,} × {WINDOW_DAYS} giorni × {N_HOURS} ore')

# Set stockout hours to NaN
sales_nan = np.where(stock_win == 1, np.nan, sales_win)

# Prepare covariate features (discount, holiday, precpt, temperature)
covs_cols = ['discount', 'holiday_flag', 'precpt', 'avg_temperature']
covs = df_train_hf[covs_cols].values.astype(np.float32)
covs_win = covs.reshape(n_series * N_WINDOWS, WINDOW_DAYS, len(covs_cols))
# Normalize per-window
covs_max = covs_win.max(axis=1, keepdims=True) + 0.1
covs_norm = covs_win / covs_max

# Hour position feature (0 to 1)
hour_pos = np.arange(N_HOURS, dtype=np.float32)[None, None, :] / (N_HOURS - 1)
hour_pos = np.broadcast_to(hour_pos, (n_samples, WINDOW_DAYS, N_HOURS))

# Build input: (n_samples, 30*17, 6)
# Feature 0: sales (NaN where stockout)
# Features 1-4: covariates (broadcast to hourly)
# Feature 5: hour position
sales_flat = sales_nan.reshape(n_samples, N_STEPS, 1)
covs_broadcast = np.broadcast_to(covs_norm[:, :, None, :],
                                  (n_samples, WINDOW_DAYS, N_HOURS, len(covs_cols)))
covs_flat = covs_broadcast.reshape(n_samples, N_STEPS, len(covs_cols))
hour_flat = hour_pos.reshape(n_samples, N_STEPS, 1)

X = np.concatenate([sales_flat, covs_flat, hour_flat], axis=-1)  # (150K, 510, 6)
n_features = X.shape[-1]
print(f'  Input shape: {X.shape} ({n_features} features)')
print(f'  NaN count: {np.isnan(X[:,:,0]).sum():,} / {n_samples * N_STEPS:,} '
      f'({100*np.isnan(X[:,:,0]).mean():.1f}%)')

# Keep original for later
sales_origin = sales_win.copy()  # (150K, 30, 17) without NaN

del sales_24, stock_24, sales_17, stock_17, covs, df_train_hf
gc.collect()

# ===========================================================================
# 2. Create model
# ===========================================================================
print(f'\n2. Creazione modello {MODEL_NAME}...')

saving_path = os.path.join(RESULTS_DIR, f'pypots_{MODEL_NAME.lower()}')
optimizer = Adam(lr=HP['lr'], weight_decay=HP['weight_decay'])

if MODEL_NAME == 'SAITS':
    model = SAITS(
        n_steps=N_STEPS, n_features=n_features,
        n_layers=HP['n_layers'], d_model=HP['d_model'], d_ffn=HP['d_ffn'],
        n_heads=HP['n_heads'], d_k=HP['d_k'], d_v=HP['d_v'],
        dropout=HP['dropout'], attn_dropout=HP['attn_dropout'],
        diagonal_attention_mask=True, ORT_weight=1, MIT_weight=1,
        batch_size=HP['batch_size'], epochs=HP['epochs'],
        patience=HP['patience'], optimizer=optimizer,
        device=DEVICE, saving_path=saving_path, verbose=True)

elif MODEL_NAME == 'TimesNet':
    model = TimesNet(
        n_steps=N_STEPS, n_features=n_features,
        n_layers=HP['n_layers'], top_k=7,
        d_model=HP['d_model'], d_ffn=HP['d_ffn'], n_kernels=5,
        dropout=HP['dropout'], apply_nonstationary_norm=True,
        epochs=HP['epochs'], batch_size=HP['batch_size'],
        patience=HP['patience'], optimizer=optimizer,
        device=DEVICE, saving_path=saving_path, verbose=True)

elif MODEL_NAME == 'iTransformer':
    model = iTransformer(
        n_steps=N_STEPS, n_features=n_features,
        n_layers=HP['n_layers'], d_model=HP['d_model'], d_ffn=HP['d_ffn'],
        n_heads=HP['n_heads'], d_k=HP['d_k'], d_v=HP['d_v'],
        dropout=HP['dropout'], attn_dropout=HP['attn_dropout'],
        ORT_weight=1, MIT_weight=1,
        batch_size=HP['batch_size'], epochs=HP['epochs'],
        patience=HP['patience'], optimizer=optimizer,
        device=DEVICE, saving_path=saving_path, verbose=True)

elif MODEL_NAME == 'DLinear':
    model = DLinear(
        n_steps=N_STEPS, n_features=n_features,
        moving_avg_window_size=N_HOURS // 2 * 2 + 1,  # 9
        individual=False, d_model=HP['d_model'],
        ORT_weight=1, MIT_weight=1,
        batch_size=HP['batch_size'], epochs=HP['epochs'],
        patience=HP['patience'], optimizer=optimizer,
        device=DEVICE, saving_path=saving_path, verbose=True)

print(f'  Model created: {MODEL_NAME}')

# ===========================================================================
# 3. Fit
# ===========================================================================
print(f'\n3. Training {MODEL_NAME}...')
t0 = time.time()
train_set = {'X': X}
model.fit(train_set)
print(f'  Training time: {time.time()-t0:.0f}s')

# ===========================================================================
# 4. Predict (impute)
# ===========================================================================
print(f'\n4. Predizione (imputation) in batch...')
# Predict in chunks to avoid OOM
PREDICT_BATCH = 10000
imputed_sales_flat = np.zeros((n_samples, N_STEPS), dtype=np.float32)

for start in range(0, n_samples, PREDICT_BATCH):
    end = min(start + PREDICT_BATCH, n_samples)
    print(f'  Predicting {start:,}-{end:,} / {n_samples:,}...')
    chunk = {'X': X[start:end]}
    res = model.predict(chunk)
    imp = res['imputation']
    if len(imp.shape) == 4:
        imp = imp.mean(axis=1)
    imputed_sales_flat[start:end] = np.clip(imp[:, :, 0], 0, None)
    del res, imp, chunk
    gc.collect()
    if DEVICE == 'mps':
        torch.mps.empty_cache()

# Reshape to (150K, 30, 17)
imputed_sales = imputed_sales_flat.reshape(n_samples, WINDOW_DAYS, N_HOURS)

print(f'  Imputed shape: {imputed_sales.shape}')
print(f'  Mean imputed (all): {imputed_sales.mean():.4f}')
print(f'  Mean original (non-NaN): {sales_origin[~np.isnan(sales_nan)].mean():.4f}')

del X, train_set
gc.collect()

# ===========================================================================
# 5. Evaluate on MNAR masks
# ===========================================================================
print(f'\n5. Valutazione su maschere MNAR...')
masks_val = pd.read_parquet(os.path.join(DATA_DIR, 'mnar_masks_val.parquet'))
masks_val = masks_val[(masks_val['hour'] >= H_START) & (masks_val['hour'] < H_END)].reset_index(drop=True)
print(f'  MNAR masks (6-22): {len(masks_val):,}')

# Build lookup: for each mask entry, find the imputed value
# Need to map (store_id, product_id, dt, hour) -> (sample_idx, day_in_window, hour_in_day)
df_ref = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_ref = df_ref.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
df_ref['row_idx'] = np.arange(len(df_ref))

# Map each row to (window_idx, day_in_window)
df_ref['window_idx'] = df_ref['row_idx'] // WINDOW_DAYS
# But windows are per-series, so: series_idx = row_idx // 90, then window within series
df_ref['series_idx'] = df_ref['row_idx'] // 90
df_ref['day_in_series'] = df_ref['row_idx'] % 90
df_ref['window_in_series'] = df_ref['day_in_series'] // WINDOW_DAYS
df_ref['day_in_window'] = df_ref['day_in_series'] % WINDOW_DAYS
df_ref['sample_idx'] = df_ref['series_idx'] * N_WINDOWS + df_ref['window_in_series']

# Build lookup
lookup_key = df_ref.set_index(['store_id', 'product_id', 'dt'])[['sample_idx', 'day_in_window']]

preds_mnar = np.zeros(len(masks_val), dtype=np.float32)
for i in range(len(masks_val)):
    r = masks_val.iloc[i]
    key = (r['store_id'], r['product_id'], r['dt'])
    if key in lookup_key.index:
        row = lookup_key.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        si = int(row['sample_idx'])
        di = int(row['day_in_window'])
        hi = int(r['hour']) - H_START
        preds_mnar[i] = imputed_sales[si, di, hi]

gt = masks_val['ground_truth'].values.astype(np.float64)
preds64 = preds_mnar.astype(np.float64)
sao = np.abs(gt).sum()
wape = np.abs(preds64 - gt).sum() / sao if sao > 0 else np.nan
wpe = (preds64 - gt).sum() / gt.sum() if gt.sum() != 0 else np.nan
print(f'  {MODEL_NAME}: WAPE_recovery={wape:.4f}, WPE_recovery={wpe:.4f}')

# Save traccia A result
ta_df = pd.DataFrame([{'imputer': MODEL_NAME, 'wape_recovery': wape, 'wpe_recovery': wpe}])
ta_df.to_parquet(os.path.join(RESULTS_DIR, f'traccia_a_{MODEL_NAME.lower()}.parquet'), index=False)

# ===========================================================================
# 6. Build completed_sales
# ===========================================================================
print(f'\n6. Costruzione completed_sales...')

# Reshape imputed back to (n_rows, 17): replace stockout values with imputed
# imputed_sales: (150K, 30, 17) -> (n_rows, 17)
imputed_flat = imputed_sales.reshape(n_rows, N_HOURS)
stock_flat = stock_17.reshape(n_rows, N_HOURS) if 'stock_17' in dir() else \
    np.array(pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet')).sort_values(
        ['store_id', 'product_id', 'dt'])['hours_stock_status'].tolist(),
        dtype=np.int8)[:, H_START:H_END]

# Reload stock since we deleted it
df_reload = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_reload = df_reload.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)
stock_reload = np.array(df_reload['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]
sales_reload = np.array(df_reload['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]

completed = sales_reload.copy()
so_mask = stock_reload == 1
completed[so_mask] = np.clip(imputed_flat[so_mask], 0, None)

# Save as parquet
df_out = df_reload[['store_id', 'product_id', 'dt']].copy()
df_out['day_num'] = np.repeat(np.arange(1, 91), n_series)  # approx
# Actually recompute day_num properly
df_out['dt_parsed'] = pd.to_datetime(df_out['dt'])
all_dates = sorted(df_out['dt_parsed'].unique())
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}
df_out['day_num'] = df_out['dt_parsed'].map(date_to_day)
df_out['dow'] = df_out['dt_parsed'].dt.dayofweek
df_out['hours_sale'] = list(completed)
df_out['hours_stock_status'] = list(stock_reload)
df_out.drop(columns=['dt_parsed'], inplace=True)

df_out.to_parquet(out_cs, index=False)
print(f'  Salvato: {out_cs}')
print(f'  Media imputata (stockout): {completed[so_mask].mean():.4f}')
print(f'  Media S_obs (in-stock):    {sales_reload[~so_mask].mean():.4f}')

print(f'\n=== DONE — {MODEL_NAME} ===')
