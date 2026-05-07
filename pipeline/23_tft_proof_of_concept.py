"""
23_tft_proof_of_concept.py — TFT proof of concept (no imputation)
===================================================================
Setup minimale per verificare:
- Data loading TimeSeriesDataSet funziona
- Training converge
- Predict produce output sensato
- WAPE calcolabile

Configurazione conservativa:
- 5.000 serie campionate stratificate per quartile di volume
- Encoder length: 119 step (1 settimana × 17 ore)
- Prediction length: 119 step (7 giorni × 17 ore = test horizon)
- Hidden size: 32, attention heads: 4
- Max epochs: 5 (per testare velocemente)
- Batch size: 64

Tempo stimato: 1-3 ore.
"""
import os, time, functools, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data import GroupNormalizer

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

H_START, H_END = 6, 23
N_HOURS = H_END - H_START  # 17
ENCODER_LENGTH = 7 * N_HOURS    # 119 step (1 settimana)
PRED_LENGTH = 7 * N_HOURS       # 119 step (test horizon)
HIDDEN_SIZE = 32
ATTENTION_HEADS = 4
BATCH_SIZE = 64
MAX_EPOCHS = 3                  # ridotto per POC
SUBSET_SIZE = 1000              # ridotto per POC (era 5000)
MAX_TRAIN_SAMPLES = 30000       # subset di sample per epoch
LR = 0.03

# NOTE: TFT su MPS è instabile (buffer error). Forziamo CPU per stabilità.
DEVICE = 'cpu'  # 'mps' if torch.backends.mps.is_available() else 'cpu'

print('=' * 72)
print('  TFT PROOF OF CONCEPT — No imputation, 5K serie')
print('=' * 72)
print(f'  Encoder length:   {ENCODER_LENGTH} step ({ENCODER_LENGTH // N_HOURS} giorni × {N_HOURS} ore)')
print(f'  Prediction len:   {PRED_LENGTH} step ({PRED_LENGTH // N_HOURS} giorni × {N_HOURS} ore)')
print(f'  Hidden size:      {HIDDEN_SIZE}')
print(f'  Subset:           {SUBSET_SIZE} serie')
print(f'  Device:           {DEVICE}')
print(f'  Max epochs:       {MAX_EPOCHS}')

# ===========================================================================
# 1. Caricamento dati
# ===========================================================================
print('\n1. Caricamento dati...')
df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))
df_train_hf['dt_parsed'] = pd.to_datetime(df_train_hf['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])
df_full = pd.concat([df_train_hf, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i+1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek
print(f'  Full: {len(df_full):,} righe, {len(all_dates)} giorni')

# Parse hourly arrays (slice 6-22)
sales_arr = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
stock_arr = np.array(df_full['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]

# ===========================================================================
# 2. Subset campionato stratificato per quartile di volume
# ===========================================================================
print(f'\n2. Subset stratificato per quartile di volume ({SUBSET_SIZE} serie)...')
strat = pd.read_parquet(os.path.join(RESULTS_DIR, 'stratification.parquet'))
# Sample equally from each quartile
samples_per_q = SUBSET_SIZE // 4
sampled = strat.groupby('vol_bin', group_keys=False).apply(
    lambda x: x.sample(min(len(x), samples_per_q), random_state=SEED)
).reset_index(drop=True)
sampled_keys = set(zip(sampled['store_id'], sampled['product_id']))
print(f'  Sampled {len(sampled)} serie:')
print(f'    {sampled["vol_bin"].value_counts().to_dict()}')

# Filter df_full to sampled series
df_sub = df_full[df_full.set_index(['store_id','product_id']).index.isin(sampled_keys)].copy()
sales_sub = sales_arr[df_sub.index.values]
stock_sub = stock_arr[df_sub.index.values]
df_sub = df_sub.reset_index(drop=True)
print(f'  Filtered: {len(df_sub):,} righe ({len(df_sub) // 97} serie × 97 giorni)')

# ===========================================================================
# 3. Espansione in formato lungo (long format orario)
# ===========================================================================
print(f'\n3. Espansione formato lungo orario...')
t0 = time.time()
n_rows = len(df_sub)
n_hourly = n_rows * N_HOURS

# Costruzione DataFrame long: una riga per (serie, day, hour)
long_data = pd.DataFrame({
    'store_id':    np.repeat(df_sub['store_id'].values, N_HOURS),
    'product_id':  np.repeat(df_sub['product_id'].values, N_HOURS),
    'city_id':     np.repeat(df_sub['city_id'].values, N_HOURS),
    'day_num':     np.repeat(df_sub['day_num'].values, N_HOURS),
    'dow':         np.repeat(df_sub['dow'].values, N_HOURS),
    'discount':    np.repeat(df_sub['discount'].values, N_HOURS),
    'avg_temperature':  np.repeat(df_sub['avg_temperature'].values, N_HOURS),
    'avg_humidity':     np.repeat(df_sub['avg_humidity'].values, N_HOURS),
    'precpt':           np.repeat(df_sub['precpt'].values, N_HOURS),
    'avg_wind_level':   np.repeat(df_sub['avg_wind_level'].values, N_HOURS),
    'holiday_flag': np.repeat(df_sub['holiday_flag'].values, N_HOURS),
    'activity_flag': np.repeat(df_sub['activity_flag'].values, N_HOURS),
    'hour':        np.tile(np.arange(H_START, H_END), n_rows),
    'sales':       sales_sub.ravel().astype(np.float32),
    'stock':       stock_sub.ravel().astype(np.int8),
})

# time_idx: progressivo orario per ogni serie
# Per ogni serie: 0, 1, ..., (97 × 17 - 1) = 1648
long_data['time_idx'] = (long_data['day_num'] - 1) * N_HOURS + (long_data['hour'] - H_START)

# Cast types
long_data['store_id'] = long_data['store_id'].astype(str)
long_data['product_id'] = long_data['product_id'].astype(str)
long_data['city_id'] = long_data['city_id'].astype(str)
long_data['dow'] = long_data['dow'].astype(str)
long_data['hour'] = long_data['hour'].astype(str)
long_data['holiday_flag'] = long_data['holiday_flag'].astype(str)
long_data['activity_flag'] = long_data['activity_flag'].astype(str)

print(f'  Long format: {len(long_data):,} righe in {time.time()-t0:.1f}s')

# Free memory
del df_sub, df_full, sales_arr, stock_arr, sales_sub, stock_sub
import gc; gc.collect()

# ===========================================================================
# 4. TimeSeriesDataSet
# ===========================================================================
print(f'\n4. Costruzione TimeSeriesDataSet...')
t0 = time.time()

# Training cutoff: gg 1-83 → time_idx 0..(83 × 17 - 1) = 0..1410
# Validation: gg 84-90 → 1411..1529
# Test: gg 91-97 → 1530..1648
TRAINING_CUTOFF = 83 * N_HOURS - 1  # 1410
VAL_CUTOFF = 90 * N_HOURS - 1       # 1529 (last train_HF time_idx)
TEST_END = 97 * N_HOURS - 1         # 1648

# Training set: time_idx <= TRAINING_CUTOFF
training_data = long_data[long_data['time_idx'] <= TRAINING_CUTOFF].copy()

training = TimeSeriesDataSet(
    training_data,
    time_idx='time_idx',
    target='sales',
    group_ids=['store_id', 'product_id'],
    min_encoder_length=ENCODER_LENGTH,
    max_encoder_length=ENCODER_LENGTH,
    min_prediction_length=PRED_LENGTH,
    max_prediction_length=PRED_LENGTH,
    static_categoricals=['store_id', 'product_id', 'city_id'],
    time_varying_known_categoricals=['dow', 'hour', 'holiday_flag', 'activity_flag'],
    time_varying_known_reals=['discount', 'avg_temperature', 'avg_humidity',
                                'precpt', 'avg_wind_level'],
    time_varying_unknown_reals=['sales'],
    target_normalizer=GroupNormalizer(groups=['store_id', 'product_id']),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)
print(f'  Training set built in {time.time()-t0:.1f}s')
print(f'    n_samples_train: {len(training)}')

# Validation set
validation = TimeSeriesDataSet.from_dataset(
    training, long_data, predict=True, stop_randomization=True,
    min_prediction_idx=TRAINING_CUTOFF + 1
)
print(f'    n_samples_val:   {len(validation)}')

# DataLoaders — subset training to MAX_TRAIN_SAMPLES per epoch
n_train_full = len(training)
print(f'  Training samples (full): {n_train_full:,}')
if n_train_full > MAX_TRAIN_SAMPLES:
    print(f'  Subsampling to {MAX_TRAIN_SAMPLES:,} for POC')
    rng = np.random.default_rng(SEED)
    idx_subset = rng.choice(n_train_full, MAX_TRAIN_SAMPLES, replace=False)
    sub_training = torch.utils.data.Subset(training, idx_subset.tolist())
    train_dataloader = torch.utils.data.DataLoader(
        sub_training, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=training._collate_fn,
    )
else:
    train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=0)

# ===========================================================================
# 5. Modello TFT
# ===========================================================================
print(f'\n5. Creazione modello TFT...')
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LR,
    hidden_size=HIDDEN_SIZE,
    attention_head_size=ATTENTION_HEADS,
    dropout=0.1,
    hidden_continuous_size=HIDDEN_SIZE // 2,
    output_size=1,
    loss=MAE(),
    log_interval=10,
    reduce_on_plateau_patience=2,
)
print(f'  Parametri: {sum(p.numel() for p in tft.parameters() if p.requires_grad):,}')

# ===========================================================================
# 6. Training
# ===========================================================================
print(f'\n6. Training TFT (max {MAX_EPOCHS} epoche)...')
t0 = time.time()

early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=2, verbose=False, mode='min')

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator='cpu',
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[early_stop],
    enable_progress_bar=True,
    logger=False,
    log_every_n_steps=50,
)

try:
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print(f'  Training time: {time.time()-t0:.0f}s')
except RuntimeError as e:
    if 'out of memory' in str(e).lower() or 'mps' in str(e).lower():
        print(f'\n  OOM su MPS! Errore: {e}')
        print(f'  Suggerimento: ridurre BATCH_SIZE o HIDDEN_SIZE')
        raise
    raise

# ===========================================================================
# 7. Predict (val period — gg 84-90)
# ===========================================================================
print(f'\n7. Predict su validation (gg 84-90)...')
t0 = time.time()
val_predictions = tft.predict(val_dataloader, return_y=True, return_index=True, mode='prediction',
                                trainer_kwargs={'accelerator': 'cpu', 'devices': 1, 'logger': False})
print(f'  Predict time: {time.time()-t0:.0f}s')
print(f'  Output shape: {val_predictions.output.shape}')
print(f'  Output range: [{val_predictions.output.min():.4f}, {val_predictions.output.max():.4f}]')
print(f'  Sample mean: {val_predictions.output.mean():.4f}')
print(f'  Sample std: {val_predictions.output.std():.4f}')

# ===========================================================================
# 8. Valutazione WAPE/WPE in-stock (val)
# ===========================================================================
print(f'\n8. Valutazione val (stockout filter)...')
preds = val_predictions.output.cpu().numpy().clip(0, None)  # (n_samples, pred_len)
truths = val_predictions.y[0].cpu().numpy()

# Reconstruction stock_status by joining with long_data
idx_df = val_predictions.index
print(f'  Predictions for {len(idx_df)} sequences (a sequence = 1 (serie, start_time))')

# Save predictions
np.save(os.path.join(RESULTS_DIR, 'tft_poc_val_preds.npy'), preds)
np.save(os.path.join(RESULTS_DIR, 'tft_poc_val_truths.npy'), truths)
idx_df.to_parquet(os.path.join(RESULTS_DIR, 'tft_poc_val_idx.parquet'), index=False)

# Quick eval over all (in-stock OR not) - just to verify training works
abs_err = np.abs(preds - truths)
sao = np.abs(truths).sum()
wape = abs_err.sum() / sao if sao > 0 else np.nan
wpe = (preds - truths).sum() / truths.sum() if truths.sum() != 0 else np.nan
print(f'  WAPE val (all): {wape:.4f}')
print(f'  WPE val (all):  {wpe:.4f}')

# ===========================================================================
# 9. Sintesi
# ===========================================================================
print('\n' + '=' * 72)
print('  PROOF OF CONCEPT — Risultati')
print('=' * 72)
print(f'  Training time:     {time.time()-t0:.0f}s totali')
print(f'  Numero parametri:  {sum(p.numel() for p in tft.parameters() if p.requires_grad):,}')
print(f'  Best val_loss:     {trainer.callback_metrics.get("val_loss", "N/A")}')
print(f'  WAPE val (raw):    {wape:.4f}')
print(f'  WPE val (raw):     {wpe:.4f}')
print(f'\n  → Verifica che WAPE sia ragionevole (~1.0-1.5)')
print(f'  → Se OK, procedere con training completo (Fase 2)')

print('\n' + '=' * 72)
print('  DONE — TFT proof of concept')
print('=' * 72)
