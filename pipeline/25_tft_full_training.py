"""
25_tft_full_training.py — TFT su 50K serie, 1 imputer alla volta
==================================================================
Usage: freshnet/bin/python pipeline/25_tft_full_training.py <imputer_key>
  imputer_key: no_imp | mediana_cond | mediana_glob | dlinear | saits

Setup Fase 2:
- 50.000 serie (tutte)
- Encoder: 119 step (1 settimana)
- Pred: 119 step (test horizon)
- Hidden 32, batch 64
- Max 10 epoche, patience 3
- 200K samples/epoch (subset)

Output:
- pipeline/results/{imp}__tft_test_per_series.parquet
- pipeline/results/tft_{imp}_log.txt (training log)
"""
import os, sys, time, functools, warnings, gc
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'tft_checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data import GroupNormalizer


class StepLogger(Callback):
    """Stampa progress ogni N step + ETA per epoca."""
    def __init__(self, every_n_steps=100):
        self.every_n_steps = every_n_steps
        self.epoch_start_time = None
        self.epoch_steps = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        self.epoch_steps = 0
        print(f'\n[EPOCH {trainer.current_epoch}] Start training')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.epoch_steps += 1
        if batch_idx > 0 and batch_idx % self.every_n_steps == 0:
            elapsed = time.time() - self.epoch_start_time
            it_per_s = batch_idx / elapsed
            total_batches = trainer.num_training_batches
            eta = (total_batches - batch_idx) / it_per_s if it_per_s > 0 else 0
            loss_val = outputs['loss'].item() if isinstance(outputs, dict) else float(outputs)
            print(f'[EPOCH {trainer.current_epoch}] step {batch_idx}/{total_batches} '
                  f'({100*batch_idx/total_batches:.0f}%) | '
                  f'loss={loss_val:.4f} | '
                  f'elapsed={elapsed:.0f}s eta={eta:.0f}s '
                  f'({it_per_s:.2f} it/s)')

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.epoch_start_time
        train_loss = trainer.callback_metrics.get('train_loss_epoch', float('nan'))
        val_loss = trainer.callback_metrics.get('val_loss', float('nan'))
        print(f'[EPOCH {trainer.current_epoch}] DONE in {elapsed:.0f}s '
              f'({elapsed/60:.1f} min) | '
              f'train_loss={train_loss:.4f} val_loss={val_loss:.4f}')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Config
H_START, H_END = 6, 23
N_HOURS = H_END - H_START
ENCODER_LENGTH = 7 * N_HOURS
PRED_LENGTH = 7 * N_HOURS
HIDDEN_SIZE = 32
ATTENTION_HEADS = 4
BATCH_SIZE = 64
MAX_EPOCHS = 10
PATIENCE = 3
MAX_TRAIN_SAMPLES = 200000
LR = 0.03
DROPOUT = 0.1
WEIGHT_DECAY = 0.0

if os.getenv('HPO_VARIANT') == '1':
    import json
    with open(os.path.join(RESULTS_DIR, 'hpo_tft_best.json')) as f:
        hpo = json.load(f)['best_params']
    head_dim = int(hpo['head_dim'])
    ATTENTION_HEADS = int(hpo['attention_heads'])
    HIDDEN_SIZE = head_dim * ATTENTION_HEADS
    DROPOUT = float(hpo['dropout'])
    LR = float(hpo['lr'])
    BATCH_SIZE = int(hpo['batch_size'])
    WEIGHT_DECAY = float(hpo['weight_decay'])
    print(f'[HPO] head_dim={head_dim} heads={ATTENTION_HEADS} hidden={HIDDEN_SIZE} '
          f'dropout={DROPOUT} lr={LR:.3e} bs={BATCH_SIZE} wd={WEIGHT_DECAY:.2e}')

DEVICE = 'cpu'  # MPS instabile con TFT

# Subset: con dtype compression provo 50K.
SUBSET_SIZE = 50000

# Argomento imputer
IMP_KEY = sys.argv[1] if len(sys.argv) > 1 else 'no_imp'
IMP_LABELS = {
    'no_imp': 'No imputation',
    'mediana_cond': 'Mediana condizionata',
    'mediana_glob': 'Mediana globale',
    'dlinear': 'DLinear',
    'saits': 'SAITS',
    'media_cond': 'Media condizionata',
    'media_glob': 'Media globale',
    'forward_fill': 'Forward fill',
    'seasonal_naive': 'Seasonal naive',
    'linear_interp': 'Linear interpolation',
    'lgb': 'LGB imputer',
    'itransformer': 'iTransformer',
    'timesnet': 'TimesNet',
    'csdi': 'CSDI',
    'imputeformer': 'ImputeFormer',
}
assert IMP_KEY in IMP_LABELS, f'Unknown imputer: {IMP_KEY}'

_suffix = '_hpo' if os.getenv('HPO_VARIANT') == '1' else ''
OUT_PATH = os.path.join(RESULTS_DIR, f'{IMP_KEY}__tft{_suffix}_test_per_series.parquet')
if os.path.exists(OUT_PATH):
    print(f'SKIP: {OUT_PATH} already exists')
    sys.exit(0)

print('=' * 72)
print(f'  TFT × {IMP_LABELS[IMP_KEY]} ({SUBSET_SIZE} serie subset)')
print('=' * 72)
print(f'  Encoder: {ENCODER_LENGTH} step | Pred: {PRED_LENGTH} step')
print(f'  Hidden: {HIDDEN_SIZE} | Batch: {BATCH_SIZE} | Max epochs: {MAX_EPOCHS}')
print(f'  Max train samples/epoch: {MAX_TRAIN_SAMPLES:,}')
print(f'  Device: {DEVICE}')

T_START = time.time()

# ===========================================================================
# 1. Caricamento dati
# ===========================================================================
print(f'\n[{time.time()-T_START:.0f}s] 1. Caricamento dati...')
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

# Subset stratificato per quartile di volume
print(f'\n[{time.time()-T_START:.0f}s]    Subset stratificato {SUBSET_SIZE} serie...')
strat = pd.read_parquet(os.path.join(RESULTS_DIR, 'stratification.parquet'))
samples_per_q = SUBSET_SIZE // 4
sampled = strat.groupby('vol_bin', group_keys=False).apply(
    lambda x: x.sample(min(len(x), samples_per_q), random_state=SEED)
).reset_index(drop=True)
sampled_keys = set(zip(sampled['store_id'], sampled['product_id']))
key_full = list(zip(df_full['store_id'], df_full['product_id']))
mask_sub = np.array([k in sampled_keys for k in key_full])
df_full = df_full[mask_sub].reset_index(drop=True)
sales_arr = sales_arr[mask_sub]
stock_arr = stock_arr[mask_sub]
print(f'    Sampled {len(sampled)} serie, {len(df_full):,} righe')
del sampled, key_full, mask_sub; gc.collect()

# Apply imputer (replace stockout values with completed_sales for training)
if IMP_KEY != 'no_imp':
    print(f'[{time.time()-T_START:.0f}s]    Loading completed_sales: {IMP_KEY}...')
    df_cs = pd.read_parquet(os.path.join(COMPLETED_DIR, f'{IMP_KEY}.parquet'))
    cs_sales = np.array(df_cs['hours_sale'].tolist(), dtype=np.float32)
    if cs_sales.shape[1] == 24:
        cs_sales = cs_sales[:, H_START:H_END]
    df_cs['dt_parsed'] = pd.to_datetime(df_cs['dt'])
    df_cs = df_cs.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
    # Match by (store_id, product_id, dt)
    key_full = list(zip(df_full['store_id'], df_full['product_id'], df_full['dt_parsed']))
    key_cs = list(zip(df_cs['store_id'], df_cs['product_id'], df_cs['dt_parsed']))
    cs_idx_map = {k: i for i, k in enumerate(key_cs)}
    matched = 0
    sales_imputed = sales_arr.copy()
    for i, k in enumerate(key_full):
        if k in cs_idx_map:
            sales_imputed[i] = cs_sales[cs_idx_map[k]]
            matched += 1
    print(f'    Matched: {matched:,}/{len(df_full):,}')
    sales_arr = sales_imputed
    del df_cs, cs_sales, cs_idx_map, sales_imputed; gc.collect()

print(f'[{time.time()-T_START:.0f}s]   Full data shape: {sales_arr.shape}')

# ===========================================================================
# 2. Long format
# ===========================================================================
print(f'\n[{time.time()-T_START:.0f}s] 2. Long format orario...')
n_rows = len(df_full)
long_data = pd.DataFrame({
    'store_id':         np.repeat(df_full['store_id'].values, N_HOURS),
    'product_id':       np.repeat(df_full['product_id'].values, N_HOURS),
    'city_id':          np.repeat(df_full['city_id'].values, N_HOURS),
    'day_num':          np.repeat(df_full['day_num'].values, N_HOURS),
    'dow':              np.repeat(df_full['dow'].values, N_HOURS),
    'discount':         np.repeat(df_full['discount'].values, N_HOURS),
    'avg_temperature':  np.repeat(df_full['avg_temperature'].values, N_HOURS),
    'avg_humidity':     np.repeat(df_full['avg_humidity'].values, N_HOURS),
    'precpt':           np.repeat(df_full['precpt'].values, N_HOURS),
    'avg_wind_level':   np.repeat(df_full['avg_wind_level'].values, N_HOURS),
    'holiday_flag':     np.repeat(df_full['holiday_flag'].values, N_HOURS),
    'activity_flag':    np.repeat(df_full['activity_flag'].values, N_HOURS),
    'hour':             np.tile(np.arange(H_START, H_END), n_rows),
    'sales':            sales_arr.ravel().astype(np.float32),
    'stock':            stock_arr.ravel().astype(np.int8),
})
long_data['time_idx'] = (long_data['day_num'] - 1) * N_HOURS + (long_data['hour'] - H_START)

# String cast per categoriche (richiesto da pytorch-forecasting)
for c in ['store_id','product_id','city_id','dow','hour','holiday_flag','activity_flag']:
    long_data[c] = long_data[c].astype(str)

# Dtype compression — riduce RAM ~40%
print(f'[{time.time()-T_START:.0f}s]   Compressing dtypes...')
mem_before = long_data.memory_usage(deep=True).sum() / 1e9
# Categoricals (riducono memoria per stringhe ripetute)
for c in ['store_id','product_id','city_id','dow','hour','holiday_flag','activity_flag']:
    long_data[c] = long_data[c].astype(str).astype('category')
# Numeric continui — float32
for c in ['discount','avg_temperature','avg_humidity','precpt','avg_wind_level','sales']:
    long_data[c] = long_data[c].astype('float32')
# Integer — int compatti
long_data['stock'] = long_data['stock'].astype('int8')
long_data['day_num'] = long_data['day_num'].astype('int16')
long_data['time_idx'] = long_data['time_idx'].astype('int32')
mem_after = long_data.memory_usage(deep=True).sum() / 1e9
print(f'[{time.time()-T_START:.0f}s]   RAM long_data: {mem_before:.2f} GB → {mem_after:.2f} GB (-{100*(1-mem_after/mem_before):.0f}%)')

print(f'[{time.time()-T_START:.0f}s]   Long format: {len(long_data):,} righe')
del df_full, sales_arr, stock_arr; gc.collect()

# ===========================================================================
# 3. TimeSeriesDataSet
# ===========================================================================
print(f'\n[{time.time()-T_START:.0f}s] 3. TimeSeriesDataSet...')
TRAINING_CUTOFF = 83 * N_HOURS - 1
VAL_CUTOFF = 90 * N_HOURS - 1
TEST_END = 97 * N_HOURS - 1

training_data = long_data[long_data['time_idx'] <= TRAINING_CUTOFF].copy()

t0 = time.time()
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
print(f'[{time.time()-T_START:.0f}s]   Training: {len(training):,} samples (build {time.time()-t0:.0f}s)')

t0 = time.time()
validation = TimeSeriesDataSet.from_dataset(
    training, long_data, predict=True, stop_randomization=True,
    min_prediction_idx=TRAINING_CUTOFF + 1
)
print(f'[{time.time()-T_START:.0f}s]   Validation: {len(validation):,} samples (build {time.time()-t0:.0f}s)')

t0 = time.time()
test_set = TimeSeriesDataSet.from_dataset(
    training, long_data, predict=True, stop_randomization=True,
    min_prediction_idx=VAL_CUTOFF + 1
)
print(f'[{time.time()-T_START:.0f}s]   Test: {len(test_set):,} samples (build {time.time()-t0:.0f}s)')

# Subsample training
print(f'[{time.time()-T_START:.0f}s]   Subsampling training to {MAX_TRAIN_SAMPLES:,}...')
n_train = len(training)
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
# 4. Modello TFT
# ===========================================================================
print(f'\n[{time.time()-T_START:.0f}s] 4. Creazione modello TFT...')
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LR,
    hidden_size=HIDDEN_SIZE,
    attention_head_size=ATTENTION_HEADS,
    dropout=DROPOUT,
    hidden_continuous_size=HIDDEN_SIZE // 2,
    output_size=1,
    loss=MAE(),
    log_interval=100,
    reduce_on_plateau_patience=2,
)
n_params = sum(p.numel() for p in tft.parameters() if p.requires_grad)
print(f'[{time.time()-T_START:.0f}s]   Parametri: {n_params:,}')

# ===========================================================================
# 5. Training
# ===========================================================================
print(f'\n[{time.time()-T_START:.0f}s] 5. Training (max {MAX_EPOCHS} epochs)...')
t_train = time.time()
early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=PATIENCE, mode='min')
checkpoint_cb = ModelCheckpoint(
    dirpath=os.path.join(CHECKPOINT_DIR, IMP_KEY),
    filename='best',
    monitor='val_loss',
    save_top_k=1,
    mode='min',
)

step_logger = StepLogger(every_n_steps=200)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator='cpu',
    devices=1,
    enable_model_summary=False,
    gradient_clip_val=0.1,
    callbacks=[early_stop, checkpoint_cb, step_logger],
    enable_progress_bar=False,  # disabled to use custom StepLogger instead
    logger=False,
    log_every_n_steps=100,
)

trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
training_time = time.time() - t_train
best_val = trainer.callback_metrics.get("val_loss", float('nan'))
print(f'\n[{time.time()-T_START:.0f}s]   Training completed in {training_time:.0f}s ({training_time/60:.1f} min)')
print(f'[{time.time()-T_START:.0f}s]   Best val_loss: {best_val:.4f}')
print(f'[{time.time()-T_START:.0f}s]   Stopped at epoch: {trainer.current_epoch}')

# ===========================================================================
# 6. Predict TEST
# ===========================================================================
print(f'\n[{time.time()-T_START:.0f}s] 6. Predict TEST (gg 91-97)...')
t0 = time.time()
test_predictions = tft.predict(
    test_dataloader, return_y=True, return_index=True, mode='prediction',
    trainer_kwargs={'accelerator': 'cpu', 'devices': 1, 'logger': False, 'enable_progress_bar': False}
)
print(f'[{time.time()-T_START:.0f}s]   Predict time: {time.time()-t0:.0f}s')
print(f'[{time.time()-T_START:.0f}s]   Output shape: {test_predictions.output.shape}')

# ===========================================================================
# 7. Valutazione filtrata in-stock
# ===========================================================================
print(f'\n[{time.time()-T_START:.0f}s] 7. Valutazione in-stock...')
preds = np.clip(test_predictions.output.cpu().numpy(), 0, None)
truths = test_predictions.y[0].cpu().numpy()
idx_df = test_predictions.index

# Stock mask per il test (gg 91-97)
test_long = long_data[long_data['time_idx'] >= VAL_CUTOFF + 1].copy()

# Build stock matrix indexed by (store_id, product_id) → (PRED_LENGTH,)
test_long_sorted = test_long.sort_values(['store_id','product_id','time_idx'])
# Group by serie e prendi sequence di stock
stock_by_serie = {}
for (sid, pid), grp in test_long_sorted.groupby(['store_id','product_id'], sort=False):
    stock_by_serie[(sid, pid)] = grp['stock'].values[:PRED_LENGTH]

# Build stock_mask in same order as idx_df
stock_mask = np.zeros((len(idx_df), PRED_LENGTH), dtype=bool)
for i in range(len(idx_df)):
    sid = idx_df.iloc[i]['store_id']
    pid = idx_df.iloc[i]['product_id']
    if (sid, pid) in stock_by_serie:
        stk = stock_by_serie[(sid, pid)]
        if len(stk) >= PRED_LENGTH:
            stock_mask[i] = stk == 0  # in-stock = 0

print(f'[{time.time()-T_START:.0f}s]   In-stock cells: {stock_mask.sum():,}/{stock_mask.size:,} ({100*stock_mask.mean():.1f}%)')

# Per-serie metrics
results_per_serie = []
for i in range(len(idx_df)):
    mask = stock_mask[i]
    if mask.sum() == 0:
        wape = np.nan; wpe = np.nan
    else:
        p_in = preds[i][mask]; t_in = truths[i][mask]
        sao = np.abs(t_in).sum()
        wape = np.abs(p_in - t_in).sum() / sao if sao > 0 else np.nan
        wpe = (p_in - t_in).sum() / t_in.sum() if t_in.sum() != 0 else np.nan
    results_per_serie.append({
        'store_id': int(idx_df.iloc[i]['store_id']),
        'product_id': int(idx_df.iloc[i]['product_id']),
        'hourly_wape': wape,
        'hourly_wpe': wpe,
        'n_hours_instock': int(mask.sum())
    })
df_tft = pd.DataFrame(results_per_serie)

# Pooled
all_in = stock_mask.flatten()
all_p = preds.flatten()[all_in]
all_t = truths.flatten()[all_in]
sao_p = np.abs(all_t).sum()
wape_pool = np.abs(all_p - all_t).sum() / sao_p if sao_p > 0 else np.nan
wpe_pool = (all_p - all_t).sum() / all_t.sum() if all_t.sum() != 0 else np.nan
wape_med = df_tft['hourly_wape'].dropna().median()
wpe_med = df_tft['hourly_wpe'].dropna().median()

print(f'\n[{time.time()-T_START:.0f}s]   TFT × {IMP_LABELS[IMP_KEY]}:')
print(f'    WAPE pool:   {wape_pool:.4f}')
print(f'    WAPE med:    {wape_med:.4f}')
print(f'    WPE pool:    {wpe_pool:+.4f}')
print(f'    WPE med:     {wpe_med:+.4f}')

# Save
df_tft.to_parquet(OUT_PATH, index=False)
print(f'[{time.time()-T_START:.0f}s]   Salvato: {OUT_PATH}')

print('\n' + '=' * 72)
print(f'  DONE — TFT × {IMP_LABELS[IMP_KEY]} ({(time.time()-T_START)/60:.1f} min totali)')
print('=' * 72)
