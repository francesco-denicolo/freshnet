"""
24_tft_poc_extended.py — TFT POC esteso con valutazione corretta su TEST
==========================================================================
Step 1 della procedura: verificare se TFT è competitivo con Chronos-bolt.

Setup:
- 3.000 serie campionate stratificate per quartile di volume
- Encoder length: 119 step (1 settimana × 17 ore)
- Prediction length: 119 step (test horizon: 7 giorni × 17 ore)
- Hidden size: 32, attention heads: 4
- Max epochs: 10 con early stopping (patience=3)
- Predict su TEST (gg 91-97) e WAPE filtrato per ore in-stock
- Confronto diretto con Chronos-bolt sulle stesse 3K serie
"""
import os, time, functools, warnings, gc
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

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
ENCODER_LENGTH = 7 * N_HOURS    # 119 step
PRED_LENGTH = 7 * N_HOURS       # 119 step (test horizon)
HIDDEN_SIZE = 32
ATTENTION_HEADS = 4
BATCH_SIZE = 64
MAX_EPOCHS = 10
PATIENCE = 3
SUBSET_SIZE = 3000
MAX_TRAIN_SAMPLES = 80000
LR = 0.03

DEVICE = 'cpu'  # forziamo CPU per stabilità (MPS instabile con TFT)

print('=' * 72)
print('  TFT POC ESTESO — Valutazione su TEST con stock filter')
print('=' * 72)
print(f'  Subset:           {SUBSET_SIZE} serie')
print(f'  Encoder length:   {ENCODER_LENGTH} step')
print(f'  Prediction len:   {PRED_LENGTH} step')
print(f'  Max epochs:       {MAX_EPOCHS} (patience={PATIENCE})')
print(f'  Max train samples: {MAX_TRAIN_SAMPLES:,}/epoch')
print(f'  Device:           {DEVICE}')

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

sales_arr = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
stock_arr = np.array(df_full['hours_stock_status'].tolist(), dtype=np.int8)[:, H_START:H_END]
print(f'  Full: {len(df_full):,} righe, {len(all_dates)} giorni')

# ===========================================================================
# 2. Subset stratificato
# ===========================================================================
print(f'\n2. Subset stratificato per quartile di volume ({SUBSET_SIZE})...')
strat = pd.read_parquet(os.path.join(RESULTS_DIR, 'stratification.parquet'))
samples_per_q = SUBSET_SIZE // 4
sampled = strat.groupby('vol_bin', group_keys=False).apply(
    lambda x: x.sample(min(len(x), samples_per_q), random_state=SEED)
).reset_index(drop=True)
sampled_keys = set(zip(sampled['store_id'], sampled['product_id']))
print(f'  Sampled {len(sampled)} serie ({sampled["vol_bin"].value_counts().to_dict()})')

key_full = list(zip(df_full['store_id'], df_full['product_id']))
mask_sub = np.array([k in sampled_keys for k in key_full])
df_sub = df_full[mask_sub].reset_index(drop=True)
sales_sub = sales_arr[mask_sub]
stock_sub = stock_arr[mask_sub]
print(f'  Filtered: {len(df_sub):,} righe ({len(df_sub) // 97} serie × 97 giorni)')

# ===========================================================================
# 3. Long format
# ===========================================================================
print(f'\n3. Long format orario...')
t0 = time.time()
long_data = pd.DataFrame({
    'store_id':         np.repeat(df_sub['store_id'].values, N_HOURS),
    'product_id':       np.repeat(df_sub['product_id'].values, N_HOURS),
    'city_id':          np.repeat(df_sub['city_id'].values, N_HOURS),
    'day_num':          np.repeat(df_sub['day_num'].values, N_HOURS),
    'dow':              np.repeat(df_sub['dow'].values, N_HOURS),
    'discount':         np.repeat(df_sub['discount'].values, N_HOURS),
    'avg_temperature':  np.repeat(df_sub['avg_temperature'].values, N_HOURS),
    'avg_humidity':     np.repeat(df_sub['avg_humidity'].values, N_HOURS),
    'precpt':           np.repeat(df_sub['precpt'].values, N_HOURS),
    'avg_wind_level':   np.repeat(df_sub['avg_wind_level'].values, N_HOURS),
    'holiday_flag':     np.repeat(df_sub['holiday_flag'].values, N_HOURS),
    'activity_flag':    np.repeat(df_sub['activity_flag'].values, N_HOURS),
    'hour':             np.tile(np.arange(H_START, H_END), len(df_sub)),
    'sales':            sales_sub.ravel().astype(np.float32),
    'stock':            stock_sub.ravel().astype(np.int8),
})
long_data['time_idx'] = (long_data['day_num'] - 1) * N_HOURS + (long_data['hour'] - H_START)

# Cast categoricals
for c in ['store_id','product_id','city_id','dow','hour','holiday_flag','activity_flag']:
    long_data[c] = long_data[c].astype(str)

print(f'  {len(long_data):,} righe in {time.time()-t0:.1f}s')
del df_sub, df_full, sales_arr, stock_arr, sales_sub, stock_sub
gc.collect()

# ===========================================================================
# 4. TimeSeriesDataSet — Train, Val, Test
# ===========================================================================
print(f'\n4. TimeSeriesDataSet...')
t0 = time.time()

# Splits temporali
TRAINING_CUTOFF = 83 * N_HOURS - 1  # 1410 (last time_idx of gg 1-83)
VAL_CUTOFF = 90 * N_HOURS - 1       # 1529 (last time_idx of gg 84-90)
TEST_END = 97 * N_HOURS - 1         # 1648

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
print(f'  Training: {len(training):,} samples')

# Validation set: predict gg 84-90 (used for early stopping)
validation = TimeSeriesDataSet.from_dataset(
    training, long_data, predict=True, stop_randomization=True,
    min_prediction_idx=TRAINING_CUTOFF + 1
)
print(f'  Validation: {len(validation):,} samples (gg 84-90)')

# Test set: predict gg 91-97 (final eval)
test_set = TimeSeriesDataSet.from_dataset(
    training, long_data, predict=True, stop_randomization=True,
    min_prediction_idx=VAL_CUTOFF + 1
)
print(f'  Test: {len(test_set):,} samples (gg 91-97)')
print(f'  Build time: {time.time()-t0:.1f}s')

# DataLoaders
n_train = len(training)
print(f'\n  Subsampling training to {MAX_TRAIN_SAMPLES:,} for tractability')
rng = np.random.default_rng(SEED)
idx_subset = rng.choice(n_train, MAX_TRAIN_SAMPLES, replace=False)
sub_training = torch.utils.data.Subset(training, idx_subset.tolist())
train_dataloader = torch.utils.data.DataLoader(
    sub_training, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    collate_fn=training._collate_fn,
)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=0)
test_dataloader = test_set.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=0)

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
    log_interval=50,
    reduce_on_plateau_patience=2,
)
print(f'  Parametri: {sum(p.numel() for p in tft.parameters() if p.requires_grad):,}')

# ===========================================================================
# 6. Training
# ===========================================================================
print(f'\n6. Training...')
t0 = time.time()
early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=PATIENCE, mode='min')

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator='cpu',
    devices=1,
    enable_model_summary=False,
    gradient_clip_val=0.1,
    callbacks=[early_stop],
    enable_progress_bar=True,
    logger=False,
    log_every_n_steps=50,
)

trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
training_time = time.time() - t0
best_val = trainer.callback_metrics.get("val_loss", float('nan'))
print(f'  Training time: {training_time:.0f}s ({training_time/60:.1f} min)')
print(f'  Best val_loss: {best_val:.4f}')

# ===========================================================================
# 7. Predict su TEST (gg 91-97)
# ===========================================================================
print(f'\n7. Predict TEST (gg 91-97)...')
t0 = time.time()
test_predictions = tft.predict(
    test_dataloader, return_y=True, return_index=True, mode='prediction',
    trainer_kwargs={'accelerator': 'cpu', 'devices': 1, 'logger': False, 'enable_progress_bar': False}
)
print(f'  Predict time: {time.time()-t0:.0f}s')
print(f'  Output shape: {test_predictions.output.shape}')

# ===========================================================================
# 8. Valutazione WAPE/WPE filtrata per ore in-stock
# ===========================================================================
print(f'\n8. Valutazione su test (filtro ore in-stock)...')
preds = np.clip(test_predictions.output.cpu().numpy(), 0, None)  # (n_serie, 119)
truths = test_predictions.y[0].cpu().numpy()                       # (n_serie, 119)
idx_df = test_predictions.index                                     # (n_serie, ...)

# Ricostruzione stock mask per il test (gg 91-97)
test_long = long_data[long_data['time_idx'] >= VAL_CUTOFF + 1].copy()
test_long['store_id'] = test_long['store_id'].astype(str)
test_long['product_id'] = test_long['product_id'].astype(str)

# idx_df ha 'store_id', 'product_id' (string) e 'time_idx' del primo step di prediction
# Per ogni serie nel test, ricostruisco stock_status dei 119 step
print(f'  Ricostruzione stock mask per {len(idx_df):,} serie...')
stock_mask = np.zeros((len(idx_df), PRED_LENGTH), dtype=bool)
for i in range(len(idx_df)):
    sid = idx_df.iloc[i]['store_id']
    pid = idx_df.iloc[i]['product_id']
    sub = test_long[(test_long['store_id'] == sid) & (test_long['product_id'] == pid)]
    sub = sub.sort_values('time_idx')
    if len(sub) >= PRED_LENGTH:
        stock_mask[i] = sub['stock'].values[:PRED_LENGTH] == 0  # in-stock = 0

print(f'  Total in-stock test cells: {stock_mask.sum():,} / {stock_mask.size:,}')

# Per-serie metrics (WAPE in-stock)
results_per_serie = []
for i in range(len(idx_df)):
    mask = stock_mask[i]
    if mask.sum() == 0:
        wape = np.nan; wpe = np.nan
    else:
        p_in = preds[i][mask]; t_in = truths[i][mask]
        sao = np.abs(t_in).sum()
        if sao > 0:
            wape = np.abs(p_in - t_in).sum() / sao
        else:
            wape = np.nan
        if t_in.sum() != 0:
            wpe = (p_in - t_in).sum() / t_in.sum()
        else:
            wpe = np.nan
    results_per_serie.append({
        'store_id': int(idx_df.iloc[i]['store_id']),
        'product_id': int(idx_df.iloc[i]['product_id']),
        'hourly_wape': wape,
        'hourly_wpe': wpe,
        'n_in_stock': int(mask.sum())
    })
df_tft = pd.DataFrame(results_per_serie)

# Pooled metrics
all_in = stock_mask.flatten()
all_p = preds.flatten()[all_in]
all_t = truths.flatten()[all_in]
sao_p = np.abs(all_t).sum()
wape_pool = np.abs(all_p - all_t).sum() / sao_p if sao_p > 0 else np.nan
wpe_pool = (all_p - all_t).sum() / all_t.sum() if all_t.sum() != 0 else np.nan
wape_med = df_tft['hourly_wape'].dropna().median()
wpe_med = df_tft['hourly_wpe'].dropna().median()

print(f'\n  TFT (No imputation, {SUBSET_SIZE} serie subset, {MAX_EPOCHS} max epochs):')
print(f'    WAPE pooled:    {wape_pool:.4f}')
print(f'    WAPE mediana:   {wape_med:.4f}')
print(f'    WPE pooled:     {wpe_pool:+.4f}')
print(f'    WPE mediana:    {wpe_med:+.4f}')

# Save TFT results
df_tft.to_parquet(os.path.join(RESULTS_DIR, 'tft_poc_test_per_series.parquet'), index=False)

# ===========================================================================
# 9. Confronto con Chronos-bolt sulle STESSE 3K serie
# ===========================================================================
print(f'\n9. Confronto con Chronos-bolt sulle stesse {SUBSET_SIZE} serie...')
chr_df = pd.read_parquet(os.path.join(RESULTS_DIR, 'no_imp__chronos_bolt_test_per_series.parquet'))
df_tft_keys = df_tft[['store_id','product_id']]
chr_sub = chr_df.merge(df_tft_keys, on=['store_id','product_id'], how='inner')
print(f'  Match: {len(chr_sub)}/{len(df_tft)} serie')

# Paired comparison
merged = df_tft.merge(chr_sub, on=['store_id','product_id'], suffixes=('_tft','_chr'))
merged = merged.dropna(subset=['hourly_wape_tft','hourly_wape_chr'])
print(f'  Paired serie: {len(merged):,}')

print(f'\n  Stesse {len(merged):,} serie — confronto WAPE mediana:')
print(f'    TFT:           {merged["hourly_wape_tft"].median():.4f}')
print(f'    Chronos-bolt:  {merged["hourly_wape_chr"].median():.4f}')
print(f'    Δ (TFT-Chr):   {merged["hourly_wape_tft"].median() - merged["hourly_wape_chr"].median():+.4f}')

print(f'\n  Confronto WPE mediana:')
print(f'    TFT:           {merged["hourly_wpe_tft"].median():+.4f}')
print(f'    Chronos-bolt:  {merged["hourly_wpe_chr"].median():+.4f}')

# Cliff's delta paired
diff = merged['hourly_wape_tft'].values - merged['hourly_wape_chr'].values
diff = diff[~np.isnan(diff)]
n_pos = (diff > 0).sum(); n_neg = (diff < 0).sum()
cliff_d = (n_pos - n_neg) / len(diff) if len(diff) > 0 else 0.0
print(f'\n  Cliff\'s δ paired (TFT vs Chronos): {cliff_d:+.3f}')
if abs(cliff_d) < 0.147: eff = 'negligible'
elif abs(cliff_d) < 0.33: eff = 'small'
elif abs(cliff_d) < 0.474: eff = 'medium'
else: eff = 'large'
winner = 'Chronos-bolt vince' if cliff_d > 0 else 'TFT vince'
print(f'  Effect: {eff} ({winner})')
print(f'  TFT batte Chronos su: {n_neg}/{len(diff)} serie ({100*n_neg/len(diff):.1f}%)')

# Wilcoxon
from scipy import stats as sstats
try:
    stat, p = sstats.wilcoxon(merged['hourly_wape_tft'], merged['hourly_wape_chr'])
    print(f'  Wilcoxon paired p-value: {p:.2e}')
except Exception as e:
    print(f'  Wilcoxon failed: {e}')

# ===========================================================================
# 10. Decisione
# ===========================================================================
print('\n' + '=' * 72)
print('  DECISIONE')
print('=' * 72)

tft_med = merged["hourly_wape_tft"].median()
chr_med = merged["hourly_wape_chr"].median()
delta = tft_med - chr_med

if delta < -0.005 and cliff_d < -0.147:
    print(f'  ✅ TFT BATTE Chronos-bolt (Δ={delta:+.4f}, Cliff δ={cliff_d:+.3f})')
    print(f'  → Procedere con training completo (Fase 2)')
elif abs(delta) < 0.01 and abs(cliff_d) < 0.147:
    print(f'  ⚠️  TFT ≈ Chronos-bolt (Δ={delta:+.4f}, Cliff δ={cliff_d:+.3f})')
    print(f'  → Equivalenza statistica. Decidere se vale la pena training completo.')
else:
    print(f'  ❌ TFT PEGGIO di Chronos-bolt (Δ={delta:+.4f}, Cliff δ={cliff_d:+.3f})')
    print(f'  → Considerare iperparametri diversi o citare nelle limitations')

print(f'\n  Tempo training POC esteso: {training_time/60:.1f} min')
print(f'  Stima full training (50K serie, 5 imputer, 10 epoche): ~{training_time/60 * 16 * 5 / 60:.1f} h')

print('\n' + '=' * 72)
print('  DONE')
print('=' * 72)
