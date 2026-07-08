"""
25_tft_full_training_gpu.py — TFT su 50K serie su GPU cloud, 1 imputer alla volta
=================================================================================
Versione GPU di 25_tft_full_training.py. Usa la config vincente da
hpo_tft_gpu_best.json (spazio HP AMPIO, senza il cap a hidden=32).

Usage: python pipeline/25_tft_full_training_gpu.py <imputer_key>
  imputer_key: no_imp | media_glob | media_cond | mediana_glob | mediana_cond |
               forward_fill | seasonal_naive | linear_interp | lgb |
               dlinear | saits | itransformer | timesnet | imputeformer

Output: pipeline/results/{imp}__tft_gpu_test_per_series.parquet  (non-distruttivo)
"""
import os, sys, time, functools, warnings, gc, json
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'tft_gpu_checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data import GroupNormalizer

# --- Device auto-detect ---
if torch.cuda.is_available():
    ACCEL, DEVICES = 'gpu', 1
    PRECISION = os.getenv('TFT_PRECISION', '32-true')
    print(f'*** CUDA: {torch.cuda.get_device_name(0)} | precision={PRECISION} ***')
else:
    ACCEL, DEVICES, PRECISION = 'cpu', 1, '32-true'
    print('*** No CUDA -> CPU fallback (lento). ***')


class StepLogger(Callback):
    def __init__(self, every_n_steps=200):
        self.every_n_steps = every_n_steps; self.epoch_start_time = None
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        print(f'\n[EPOCH {trainer.current_epoch}] Start')
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx > 0 and batch_idx % self.every_n_steps == 0:
            el = time.time() - self.epoch_start_time; ips = batch_idx / el
            tot = trainer.num_training_batches
            loss_val = outputs['loss'].item() if isinstance(outputs, dict) else float(outputs)
            print(f'[E{trainer.current_epoch}] {batch_idx}/{tot} ({100*batch_idx/tot:.0f}%) '
                  f'loss={loss_val:.4f} {el:.0f}s eta={(tot-batch_idx)/ips if ips>0 else 0:.0f}s ({ips:.1f} it/s)')
    def on_train_epoch_end(self, trainer, pl_module):
        el = time.time() - self.epoch_start_time
        tl = trainer.callback_metrics.get('train_loss_epoch', float('nan'))
        vl = trainer.callback_metrics.get('val_loss', float('nan'))
        print(f'[EPOCH {trainer.current_epoch}] DONE {el:.0f}s | train_loss={tl:.4f} val_loss={vl:.4f}')

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# Config
H_START, H_END = 6, 23; N_HOURS = H_END - H_START
ENCODER_LENGTH = 7 * N_HOURS; PRED_LENGTH = 7 * N_HOURS
MAX_EPOCHS = int(os.getenv('TFT_MAX_EPOCHS', 30))
PATIENCE = int(os.getenv('TFT_PATIENCE', 5))
MAX_TRAIN_SAMPLES = int(os.getenv('TFT_MAX_TRAIN', 400_000))
SUBSET_SIZE = int(os.getenv('SERIES_SUBSAMPLE', 0)) or 50000  # su Colab free usa es. 15000

# HP dalla config GPU vincente (obbligatoria: questa è la 'resourcing vera')
HPO_JSON = os.path.join(RESULTS_DIR, 'hpo_tft_gpu_best.json')
if not os.path.exists(HPO_JSON):
    sys.exit(f'ERRORE: manca {HPO_JSON}. Esegui prima 31_hpo_tft_gpu.py.')
with open(HPO_JSON) as f:
    hpo = json.load(f)['best_params']
head_dim = int(hpo['head_dim']); ATTENTION_HEADS = int(hpo['attention_heads'])
HIDDEN_SIZE = head_dim * ATTENTION_HEADS
DROPOUT = float(hpo['dropout']); LR = float(hpo['lr'])
BATCH_SIZE = int(hpo['batch_size']); WEIGHT_DECAY = float(hpo['weight_decay'])
print(f'[HPO-GPU] head_dim={head_dim} heads={ATTENTION_HEADS} hidden={HIDDEN_SIZE} '
      f'dropout={DROPOUT} lr={LR:.3e} bs={BATCH_SIZE} wd={WEIGHT_DECAY:.2e}')

IMP_KEY = sys.argv[1] if len(sys.argv) > 1 else 'no_imp'
IMP_LABELS = {
    'no_imp': 'No imputation', 'mediana_cond': 'Mediana condizionata',
    'mediana_glob': 'Mediana globale', 'dlinear': 'DLinear', 'saits': 'SAITS',
    'media_cond': 'Media condizionata', 'media_glob': 'Media globale',
    'forward_fill': 'Forward fill', 'seasonal_naive': 'Seasonal naive',
    'linear_interp': 'Linear interpolation', 'lgb': 'LGB imputer',
    'itransformer': 'iTransformer', 'timesnet': 'TimesNet', 'csdi': 'CSDI',
    'imputeformer': 'ImputeFormer',
}
IMP_LABEL = IMP_LABELS.get(IMP_KEY, IMP_KEY)

OUT_PATH = os.path.join(RESULTS_DIR, f'{IMP_KEY}__tft_gpu_test_per_series.parquet')
if os.path.exists(OUT_PATH):
    print(f'SKIP: {OUT_PATH} already exists'); sys.exit(0)

print('=' * 72)
print(f'  TFT-GPU × {IMP_LABEL} | hidden={HIDDEN_SIZE} bs={BATCH_SIZE} '
      f'epochs<= {MAX_EPOCHS} max_train={MAX_TRAIN_SAMPLES:,} | dev={ACCEL}')
print('=' * 72)
T_START = time.time()

# 1. Dati
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
strat = pd.read_parquet(os.path.join(RESULTS_DIR, 'stratification.parquet'))
samples_per_q = SUBSET_SIZE // 4
sampled = strat.groupby('vol_bin', group_keys=False).apply(
    lambda x: x.sample(min(len(x), samples_per_q), random_state=SEED)).reset_index(drop=True)
sampled_keys = set(zip(sampled['store_id'], sampled['product_id']))
key_full = list(zip(df_full['store_id'], df_full['product_id']))
mask_sub = np.array([k in sampled_keys for k in key_full])
df_full = df_full[mask_sub].reset_index(drop=True)
sales_arr = sales_arr[mask_sub]; stock_arr = stock_arr[mask_sub]
print(f'    Sampled {len(sampled)} serie, {len(df_full):,} righe')
del sampled, key_full, mask_sub; gc.collect()

# Imputer (sostituisce le ore di stockout con completed_sales per il training)
if IMP_KEY != 'no_imp':
    print(f'[{time.time()-T_START:.0f}s]    Loading completed_sales: {IMP_KEY}...')
    df_cs = pd.read_parquet(os.path.join(COMPLETED_DIR, f'{IMP_KEY}.parquet'))
    cs_sales = np.array(df_cs['hours_sale'].tolist(), dtype=np.float32)
    if cs_sales.shape[1] == 24:
        cs_sales = cs_sales[:, H_START:H_END]
    df_cs['dt_parsed'] = pd.to_datetime(df_cs['dt'])
    df_cs = df_cs.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
    key_full = list(zip(df_full['store_id'], df_full['product_id'], df_full['dt_parsed']))
    key_cs = list(zip(df_cs['store_id'], df_cs['product_id'], df_cs['dt_parsed']))
    cs_idx_map = {k: i for i, k in enumerate(key_cs)}
    matched = 0; sales_imputed = sales_arr.copy()
    for i, k in enumerate(key_full):
        if k in cs_idx_map:
            sales_imputed[i] = cs_sales[cs_idx_map[k]]; matched += 1
    print(f'    Matched: {matched:,}/{len(df_full):,}')
    sales_arr = sales_imputed
    del df_cs, cs_sales, cs_idx_map, sales_imputed; gc.collect()

# 2. Long format
print(f'\n[{time.time()-T_START:.0f}s] 2. Long format...')
n_rows = len(df_full)
# Memory-efficient long build: categoricals via from_codes (evita ~82M stringhe Python
# in un colpo -> è ciò che manda in OOM Colab a 12.7 GB di RAM di sistema).
NH = N_HOURS
day_rep = np.repeat(df_full['day_num'].values.astype(np.int32), NH)
hour_code = np.tile(np.arange(NH, dtype=np.int16), n_rows)
cols = {}
for c in ['store_id','product_id','city_id','dow','holiday_flag','activity_flag']:
    cc = df_full[c].astype(str).astype('category')
    cols[c] = pd.Categorical.from_codes(np.repeat(cc.cat.codes.values, NH), cc.cat.categories)
    del cc
cols['hour'] = pd.Categorical.from_codes(hour_code, [str(h) for h in range(H_START, H_END)])
for c in ['discount','avg_temperature','avg_humidity','precpt','avg_wind_level']:
    cols[c] = np.repeat(df_full[c].values.astype(np.float32), NH)
cols['sales'] = sales_arr.ravel().astype(np.float32)
cols['stock'] = stock_arr.ravel().astype(np.int8)
cols['day_num'] = day_rep.astype(np.int16)
cols['time_idx'] = ((day_rep - 1) * NH + hour_code.astype(np.int32)).astype(np.int32)
long_data = pd.DataFrame(cols)
print(f'[{time.time()-T_START:.0f}s]   Long: {len(long_data):,} righe')
del cols, day_rep, hour_code, df_full, sales_arr, stock_arr; gc.collect()

# 3. TimeSeriesDataSet
print(f'\n[{time.time()-T_START:.0f}s] 3. TimeSeriesDataSet...')
TRAINING_CUTOFF = 83 * N_HOURS - 1
VAL_CUTOFF = 90 * N_HOURS - 1
training_data = long_data[long_data['time_idx'] <= TRAINING_CUTOFF].copy()
training = TimeSeriesDataSet(
    training_data, time_idx='time_idx', target='sales', group_ids=['store_id', 'product_id'],
    min_encoder_length=ENCODER_LENGTH, max_encoder_length=ENCODER_LENGTH,
    min_prediction_length=PRED_LENGTH, max_prediction_length=PRED_LENGTH,
    static_categoricals=['store_id', 'product_id', 'city_id'],
    time_varying_known_categoricals=['dow', 'hour', 'holiday_flag', 'activity_flag'],
    time_varying_known_reals=['discount', 'avg_temperature', 'avg_humidity', 'precpt', 'avg_wind_level'],
    time_varying_unknown_reals=['sales'],
    target_normalizer=GroupNormalizer(groups=['store_id', 'product_id']),
    add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    allow_missing_timesteps=True)
validation = TimeSeriesDataSet.from_dataset(
    training, long_data, predict=True, stop_randomization=True, min_prediction_idx=TRAINING_CUTOFF + 1)
test_set = TimeSeriesDataSet.from_dataset(
    training, long_data, predict=True, stop_randomization=True, min_prediction_idx=VAL_CUTOFF + 1)
print(f'[{time.time()-T_START:.0f}s]   Train {len(training):,} | Val {len(validation):,} | Test {len(test_set):,}')

n_train = len(training)
if n_train > MAX_TRAIN_SAMPLES:
    rng = np.random.default_rng(SEED)
    idx_subset = rng.choice(n_train, MAX_TRAIN_SAMPLES, replace=False)
    sub_training = torch.utils.data.Subset(training, idx_subset.tolist())
else:
    sub_training = training
train_dataloader = torch.utils.data.DataLoader(
    sub_training, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=training._collate_fn)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=0)
test_dataloader = test_set.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=0)

# 4. Modello
print(f'\n[{time.time()-T_START:.0f}s] 4. TFT...')
tft = TemporalFusionTransformer.from_dataset(
    training, learning_rate=LR, hidden_size=HIDDEN_SIZE, attention_head_size=ATTENTION_HEADS,
    dropout=DROPOUT, hidden_continuous_size=min(HIDDEN_SIZE, 16), output_size=1,
    loss=MAE(), log_interval=0, reduce_on_plateau_patience=2, weight_decay=WEIGHT_DECAY)
print(f'[{time.time()-T_START:.0f}s]   Parametri: {sum(p.numel() for p in tft.parameters() if p.requires_grad):,}')

# 5. Training
print(f'\n[{time.time()-T_START:.0f}s] 5. Training (max {MAX_EPOCHS} epochs)...')
t_train = time.time()
early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=PATIENCE, mode='min')
checkpoint_cb = ModelCheckpoint(dirpath=os.path.join(CHECKPOINT_DIR, IMP_KEY),
                                filename='best', monitor='val_loss', save_top_k=1, mode='min')
trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS, accelerator=ACCEL, devices=DEVICES, precision=PRECISION,
    enable_model_summary=False, gradient_clip_val=0.1,
    callbacks=[early_stop, checkpoint_cb, StepLogger(200)],
    enable_progress_bar=False, logger=False, log_every_n_steps=100)
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
best_score = getattr(early_stop, 'best_score', None)
best_val = float(best_score) if best_score is not None else float('nan')
print(f'\n[{time.time()-T_START:.0f}s]   Training done in {(time.time()-t_train)/60:.1f} min | '
      f'best val_loss={best_val:.4f} | stopped epoch {trainer.current_epoch}')

# 6. Predict TEST (dal best checkpoint)
print(f'\n[{time.time()-T_START:.0f}s] 6. Predict TEST (gg 91-97)...')
best_ckpt = checkpoint_cb.best_model_path
if best_ckpt and os.path.exists(best_ckpt):
    print(f'   Loading best checkpoint: {os.path.basename(best_ckpt)}')
    tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
test_predictions = tft.predict(
    test_dataloader, return_y=True, return_index=True, mode='prediction',
    trainer_kwargs={'accelerator': ACCEL, 'devices': DEVICES, 'logger': False, 'enable_progress_bar': False})
print(f'[{time.time()-T_START:.0f}s]   Output shape: {test_predictions.output.shape}')

# 7. Valutazione in-stock
preds = np.clip(test_predictions.output.cpu().numpy(), 0, None)
truths = test_predictions.y[0].cpu().numpy()
idx_df = test_predictions.index
test_long = long_data[long_data['time_idx'] >= VAL_CUTOFF + 1].copy()
test_long_sorted = test_long.sort_values(['store_id','product_id','time_idx'])
stock_by_serie = {}
for (sid, pid), grp in test_long_sorted.groupby(['store_id','product_id'], sort=False):
    stock_by_serie[(sid, pid)] = grp['stock'].values[:PRED_LENGTH]
stock_mask = np.zeros((len(idx_df), PRED_LENGTH), dtype=bool)
for i in range(len(idx_df)):
    sid, pid = idx_df.iloc[i]['store_id'], idx_df.iloc[i]['product_id']
    if (sid, pid) in stock_by_serie:
        stk = stock_by_serie[(sid, pid)]
        if len(stk) >= PRED_LENGTH:
            stock_mask[i] = stk == 0

results_per_serie = []
for i in range(len(idx_df)):
    mask = stock_mask[i]
    if mask.sum() == 0:
        wape = wpe = np.nan
    else:
        p_in, t_in = preds[i][mask], truths[i][mask]
        sao = np.abs(t_in).sum()
        wape = np.abs(p_in - t_in).sum() / sao if sao > 0 else np.nan
        wpe = (p_in - t_in).sum() / t_in.sum() if t_in.sum() != 0 else np.nan
    results_per_serie.append({'store_id': int(idx_df.iloc[i]['store_id']),
                              'product_id': int(idx_df.iloc[i]['product_id']),
                              'hourly_wape': wape, 'hourly_wpe': wpe,
                              'n_hours_instock': int(mask.sum())})
df_tft = pd.DataFrame(results_per_serie)
all_in = stock_mask.flatten()
all_p, all_t = preds.flatten()[all_in], truths.flatten()[all_in]
wape_pool = np.abs(all_p - all_t).sum() / max(np.abs(all_t).sum(), 1e-9)
wpe_pool = (all_p - all_t).sum() / all_t.sum() if all_t.sum() != 0 else np.nan
print(f'\n[{time.time()-T_START:.0f}s]   TFT-GPU × {IMP_LABEL}:')
print(f'    WAPE pool={wape_pool:.4f} | WAPE med={df_tft["hourly_wape"].dropna().median():.4f} | '
      f'WPE pool={wpe_pool:+.4f} | WPE med={df_tft["hourly_wpe"].dropna().median():+.4f}')
df_tft.to_parquet(OUT_PATH, index=False)
print(f'[{time.time()-T_START:.0f}s]   Salvato: {OUT_PATH}')
print('=' * 72)
print(f'  DONE — TFT-GPU × {IMP_LABEL} ({(time.time()-T_START)/60:.1f} min)')
print('=' * 72)
