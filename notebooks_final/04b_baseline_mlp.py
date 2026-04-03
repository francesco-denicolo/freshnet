"""
04b_baseline_mlp.py — Fase A2: Baseline MLP (senza imputation)
===============================================================
MLP su dati sporchi (S_obs con zeri da stockout):
  1. MLP (no lags): embeddings + feature continue
  2. MLP (M5 lags): embeddings + feature continue + M5-style lag features

Valutazione solo ore in-stock, WAPE/WPE orario + giornaliero.
Direct forecast: lag da gg 1-83 (val) o gg 1-90 (test).

Eseguire con: freshnet/bin/python notebooks_final/04b_baseline_mlp.py
"""

import sys
import os
import gc
import time
import functools
import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = 'mps' if torch.backends.mps.is_available() else (
    'cuda' if torch.cuda.is_available() else 'cpu')

CONT_FEATURES = ['discount', 'avg_temperature', 'avg_humidity',
                  'precpt', 'avg_wind_level', 'holiday_flag', 'activity_flag']
LAG_FEATURE_NAMES = [
    'lag_1d', 'lag_7d', 'lag_14d',
    'rmean_7d', 'rmean_14d', 'rstd_7d',
    'lag_dow', 'rmean_dow',
    'daily_total_lag1', 'daily_total_rmean7',
    'momentum_1d_7d',
]

BATCH_SIZE = 4096
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
HIDDEN = [128, 64]
EMB_DIMS = {'store_id': 32, 'product_id': 32, 'city_id': 8, 'dow': 4}
CARDINALITIES = {'store_id': 898, 'product_id': 865, 'city_id': 18, 'dow': 7}

# ===========================================================================
print('=' * 72)
print('  FASE A2 — MLP BASELINE (senza imputation)')
print('=' * 72)

# ---------------------------------------------------------------------------
# 1. Caricamento dati
# ---------------------------------------------------------------------------
print('\n1. Caricamento dati...')
df_train = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))

df_train['dt_parsed'] = pd.to_datetime(df_train['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])

df_full = pd.concat([df_train, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)

all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek

print(f'  Full: {len(df_full):,} righe, {len(all_dates)} giorni')
print(f'  Device: {DEVICE}')

del df_train, df_eval

print('  Parsing hourly arrays...')
sales_all = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)
stock_all = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)
print('  Done.')


# ---------------------------------------------------------------------------
# 2. Build series cache
# ---------------------------------------------------------------------------
print('\n2. Building series cache...')
series_cache = {}
for (sid, pid), grp in df_full.groupby(['store_id', 'product_id'], sort=False):
    grp_s = grp.sort_values('day_num')
    series_cache[(sid, pid)] = {
        'days': grp_s['day_num'].values,
        'dows': grp_s['dow'].values,
        'sales': sales_all[grp_s.index.values],
        'stock': stock_all[grp_s.index.values],
        'city_id': grp_s['city_id'].values[0],
        'conts': grp_s[CONT_FEATURES].values.astype(np.float32),
    }
print(f'  {len(series_cache):,} serie')

# Free df_full after building cache
del df_full, sales_all, stock_all
gc.collect()


# ---------------------------------------------------------------------------
# 3. Lag computation
# ---------------------------------------------------------------------------
def compute_lags_for_day(avail_sales, avail_dows, target_dow, K):
    """Compute 11 M5-style lag features for one day."""
    z = np.float32
    lags = {n: np.full(24, np.nan, dtype=z) for n in LAG_FEATURE_NAMES}
    if K == 0:
        return lags

    lags['lag_1d'] = avail_sales[-1]
    if K >= 7:  lags['lag_7d'] = avail_sales[-7]
    if K >= 14: lags['lag_14d'] = avail_sales[-14]
    if K >= 7:  lags['rmean_7d'] = avail_sales[-7:].mean(axis=0)
    if K >= 14: lags['rmean_14d'] = avail_sales[-14:].mean(axis=0)
    if K >= 2:
        w = min(7, K)
        lags['rstd_7d'] = avail_sales[-w:].std(axis=0)

    same_dow = avail_dows == target_dow
    if same_dow.any():
        dow_sales = avail_sales[same_dow]
        lags['lag_dow'] = dow_sales[-1]
        lags['rmean_dow'] = dow_sales.mean(axis=0)

    daily_totals = avail_sales.sum(axis=1)
    lags['daily_total_lag1'] = np.full(24, daily_totals[-1], dtype=z)
    if K >= 7:
        lags['daily_total_rmean7'] = np.full(24, daily_totals[-7:].mean(), dtype=z)

    rm7, l1 = lags['rmean_7d'], lags['lag_1d']
    if not np.isnan(rm7).all():
        valid_h = (~np.isnan(l1)) & (~np.isnan(rm7)) & (rm7 > 0)
        if valid_h.any():
            mom = np.full(24, np.nan, dtype=z)
            mom[valid_h] = l1[valid_h] / rm7[valid_h]
            lags['momentum_1d_7d'] = mom

    return lags


# ---------------------------------------------------------------------------
# 4. Dataset builder
# ---------------------------------------------------------------------------
def build_dataset(split, use_lags, cont_mean=None, cont_std=None,
                  lag_mean=None, lag_std=None):
    """Build MLP dataset (day-level, 24h output)."""
    if split == 'train':
        d_min, d_max = 2, 83
    elif split == 'val':
        d_min, d_max = 84, 90
    else:
        d_min, d_max = 91, 97

    cat_list, cont_list, lag_list = [], [], []
    target_list, stock_list, sid_list, pid_list = [], [], [], []

    n_done = 0
    for (sid, pid), sd in series_cache.items():
        n_done += 1
        if n_done % 10000 == 0:
            print(f'      ... {n_done:,}/{len(series_cache):,} serie')

        days, dows, sales, stock = sd['days'], sd['dows'], sd['sales'], sd['stock']
        city, conts = sd['city_id'], sd['conts']

        for idx in range(len(days)):
            d = days[idx]
            if d < d_min or d > d_max:
                continue

            a_day = d - 1 if split == 'train' else (83 if split == 'val' else 90)

            cat_list.append([sid, pid, city, dows[idx]])
            cont_list.append(conts[idx])
            target_list.append(sales[idx])
            stock_list.append(stock[idx])
            sid_list.append(sid)
            pid_list.append(pid)

            if use_lags:
                avail_mask = days <= a_day
                K = int(avail_mask.sum())
                if K > 0:
                    lag_dict = compute_lags_for_day(
                        sales[avail_mask], dows[avail_mask], dows[idx], K)
                else:
                    lag_dict = {n: np.full(24, np.nan, dtype=np.float32)
                                for n in LAG_FEATURE_NAMES}

                feat_arrays, masks = [], np.zeros(11, dtype=np.float32)
                for fi, name in enumerate(LAG_FEATURE_NAMES):
                    arr = lag_dict[name]
                    if not np.isnan(arr).all():
                        masks[fi] = 1.0
                        feat_arrays.append(np.where(np.isnan(arr), 0.0, arr).astype(np.float32))
                    else:
                        feat_arrays.append(np.zeros(24, dtype=np.float32))
                feat_arrays.append(masks)
                lag_list.append(np.concatenate(feat_arrays))
            else:
                lag_list.append(np.array([], dtype=np.float32))

    cat_arr = np.array(cat_list, dtype=np.int64)
    cont_arr = np.array(cont_list, dtype=np.float32)
    target_arr = np.array(target_list, dtype=np.float32)
    stock_arr = np.array(stock_list, dtype=np.float32)

    if len(lag_list) > 0 and len(lag_list[0]) > 0:
        lag_arr = np.array(lag_list, dtype=np.float32)
    else:
        lag_arr = np.zeros((len(cat_list), 0), dtype=np.float32)

    if cont_mean is None:
        cont_mean = cont_arr.mean(axis=0)
        cont_std = cont_arr.std(axis=0)
        cont_std[cont_std < 1e-8] = 1.0
    cont_arr = (cont_arr - cont_mean) / cont_std

    if lag_arr.shape[1] > 0:
        if lag_mean is None:
            lag_mean = lag_arr.mean(axis=0)
            lag_std = lag_arr.std(axis=0)
            lag_std[lag_std < 1e-8] = 1.0
        lag_arr = (lag_arr - lag_mean) / lag_std

    return {
        'cat': cat_arr, 'cont': cont_arr, 'lags': lag_arr,
        'targets': target_arr, 'stock': stock_arr,
        'store_ids': np.array(sid_list, dtype=np.int64),
        'product_ids': np.array(pid_list, dtype=np.int64),
        'cont_mean': cont_mean, 'cont_std': cont_std,
        'lag_mean': lag_mean, 'lag_std': lag_std,
    }


# ---------------------------------------------------------------------------
# 5. Model
# ---------------------------------------------------------------------------
class RetailDataset(Dataset):
    def __init__(self, cat, cont, lags, targets):
        self.cat = torch.from_numpy(cat)
        self.cont = torch.from_numpy(cont)
        self.lags = torch.from_numpy(lags)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.cat[idx], self.cont[idx], self.lags[idx], self.targets[idx]


class RetailMLP(nn.Module):
    def __init__(self, n_cont, n_lags):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(CARDINALITIES[name], EMB_DIMS[name])
            for name in EMB_DIMS
        })
        self.emb_names = ['store_id', 'product_id', 'city_id', 'dow']
        total_emb = sum(EMB_DIMS.values())
        input_dim = total_emb + n_cont + n_lags

        layers = []
        prev_dim = input_dim
        for h in HIDDEN:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 24))
        layers.append(nn.Softplus())
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat, cont, lags):
        emb_list = [self.embeddings[name](cat[:, i])
                    for i, name in enumerate(self.emb_names)]
        x = torch.cat(emb_list + [cont], dim=1)
        if lags.shape[1] > 0:
            x = torch.cat([x, lags], dim=1)
        return self.mlp(x)


def predict(model, data):
    model.eval()
    cat_t = torch.from_numpy(data['cat']).to(DEVICE)
    cont_t = torch.from_numpy(data['cont']).to(DEVICE)
    lags_t = torch.from_numpy(data['lags']).to(DEVICE)

    all_preds = []
    with torch.no_grad():
        for start in range(0, len(cat_t), 10000):
            end = min(start + 10000, len(cat_t))
            p = model(cat_t[start:end], cont_t[start:end], lags_t[start:end])
            all_preds.append(p.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


# ---------------------------------------------------------------------------
# 6. Evaluation (in-stock only)
# ---------------------------------------------------------------------------
def eval_instock(preds_24, obs_24, stock_24, store_ids, product_ids):
    """Compute in-stock metrics from day-level arrays. Returns (pooled, ps_df)."""
    instock = stock_24 == 0

    # Hourly pooled
    p_h, o_h = preds_24[instock], obs_24[instock]
    sae_h, sao_h = np.abs(p_h - o_h).sum(), np.abs(o_h).sum()
    se_h, so_h = (p_h - o_h).sum(), o_h.sum()

    # Daily pooled
    n_s = preds_24.shape[0]
    sae_d, sao_d, se_d, so_d = 0., 0., 0., 0.
    for d in range(n_s):
        m_d = instock[d]
        if m_d.any():
            pd_v, od_v = preds_24[d, m_d].sum(), obs_24[d, m_d].sum()
            sae_d += abs(pd_v - od_v)
            sao_d += abs(od_v)
            se_d += pd_v - od_v
            so_d += od_v

    pooled = {
        'hourly_wape': sae_h / sao_h if sao_h > 0 else np.nan,
        'hourly_wpe': se_h / so_h if so_h != 0 else np.nan,
        'daily_wape': sae_d / sao_d if sao_d > 0 else np.nan,
        'daily_wpe': se_d / so_d if so_d != 0 else np.nan,
    }

    # Per-series
    series_map = {}
    for i in range(n_s):
        key = (store_ids[i], product_ids[i])
        if key not in series_map:
            series_map[key] = []
        series_map[key].append(i)

    records = []
    for (sid, pid), indices in series_map.items():
        sae_sh, sao_sh, se_sh, so_sh = 0., 0., 0., 0.
        sae_sd, sao_sd, se_sd, so_sd = 0., 0., 0., 0.
        n_in, n_vd = 0, 0

        for i in indices:
            m_d = instock[i]
            n_in += int(m_d.sum())
            sae_sh += np.abs(preds_24[i, m_d] - obs_24[i, m_d]).sum()
            sao_sh += np.abs(obs_24[i, m_d]).sum()
            se_sh += (preds_24[i, m_d] - obs_24[i, m_d]).sum()
            so_sh += obs_24[i, m_d].sum()
            if m_d.any():
                pd_v, od_v = preds_24[i, m_d].sum(), obs_24[i, m_d].sum()
                sae_sd += abs(pd_v - od_v)
                sao_sd += abs(od_v)
                se_sd += pd_v - od_v
                so_sd += od_v
                n_vd += 1

        records.append({
            'store_id': sid, 'product_id': pid,
            'hourly_wape': sae_sh / sao_sh if sao_sh > 0 else np.nan,
            'hourly_wpe': se_sh / so_sh if so_sh != 0 else np.nan,
            'daily_wape': sae_sd / sao_sd if sao_sd > 0 else np.nan,
            'daily_wpe': se_sd / so_sd if so_sd != 0 else np.nan,
            'n_hours_instock': n_in, 'n_days_valid': n_vd,
        })

    return pooled, pd.DataFrame(records)


# ===========================================================================
# 7. Training loop
# ===========================================================================
all_results = {}

for use_lags, variant_label in [(False, 'MLP (no lags)'), (True, 'MLP (M5 lags)')]:
    print(f'\n{"="*72}')
    print(f'  === {variant_label} ===')
    print(f'{"="*72}')
    t0 = time.time()

    # Build datasets
    print('  Building train dataset...')
    train_data = build_dataset('train', use_lags)
    print(f'  Train: {len(train_data["targets"]):,} samples, '
          f'cont={train_data["cont"].shape[1]}, lags={train_data["lags"].shape[1]}')

    print('  Building val dataset...')
    val_data = build_dataset('val', use_lags,
                              cont_mean=train_data['cont_mean'],
                              cont_std=train_data['cont_std'],
                              lag_mean=train_data['lag_mean'],
                              lag_std=train_data['lag_std'])
    print(f'  Val:   {len(val_data["targets"]):,} samples')

    n_lags = train_data['lags'].shape[1]
    n_cont = train_data['cont'].shape[1]
    model = RetailMLP(n_cont, n_lags).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model: {n_params:,} parameters')

    # Prepare loaders
    train_ds = RetailDataset(train_data['cat'], train_data['cont'],
                              train_data['lags'], train_data['targets'])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=False)

    # Pre-compute val tensors + in-stock mask
    val_instock = val_data['stock'] == 0
    val_cat_t = torch.from_numpy(val_data['cat']).to(DEVICE)
    val_cont_t = torch.from_numpy(val_data['cont']).to(DEVICE)
    val_lags_t = torch.from_numpy(val_data['lags']).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_wape, best_epoch, best_state = float('inf'), 0, None
    epochs_no_improve = 0

    print(f'  Training ({MAX_EPOCHS} max epochs, patience={PATIENCE})...')

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss, n_batches = 0.0, 0

        for cat, cont, lags, targets in train_loader:
            cat, cont, lags, targets = (
                cat.to(DEVICE), cont.to(DEVICE), lags.to(DEVICE), targets.to(DEVICE))
            preds = model(cat, cont, lags)
            loss = nn.functional.mse_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        # Validate: WAPE on in-stock hours (pooled)
        model.eval()
        with torch.no_grad():
            all_preds = []
            for start in range(0, len(val_cat_t), 10000):
                end = min(start + 10000, len(val_cat_t))
                p = model(val_cat_t[start:end], val_cont_t[start:end], val_lags_t[start:end])
                all_preds.append(p.cpu().numpy())
            val_preds = np.concatenate(all_preds, axis=0)

        p_h = val_preds[val_instock]
        o_h = val_data['targets'][val_instock]
        val_wape = np.abs(p_h - o_h).sum() / np.abs(o_h).sum()

        print(f'    Epoch {epoch:3d}: loss={train_loss/n_batches:.6f}, '
              f'val_WAPE_instock={val_wape:.6f}')

        if val_wape < best_val_wape:
            best_val_wape = val_wape
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f'    Early stopping (best={best_epoch}, WAPE={best_val_wape:.6f})')
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)

    # Evaluate val
    val_preds_final = predict(model, val_data)
    pooled_val, ps_val = eval_instock(
        val_preds_final, val_data['targets'], val_data['stock'],
        val_data['store_ids'], val_data['product_ids'])
    med_val = {c: ps_val[c].dropna().median() for c in
               ['hourly_wape', 'hourly_wpe', 'daily_wape', 'daily_wpe']}
    print(f'  Val final: WAPE_h pool={pooled_val["hourly_wape"]:.4f}, '
          f'med={med_val["hourly_wape"]:.4f}')

    # Build + evaluate test
    print('  Building test dataset...')
    test_data = build_dataset('test', use_lags,
                               cont_mean=train_data['cont_mean'],
                               cont_std=train_data['cont_std'],
                               lag_mean=train_data['lag_mean'],
                               lag_std=train_data['lag_std'])
    print(f'  Test:  {len(test_data["targets"]):,} samples')

    test_preds = predict(model, test_data)
    pooled_test, ps_test = eval_instock(
        test_preds, test_data['targets'], test_data['stock'],
        test_data['store_ids'], test_data['product_ids'])
    med_test = {c: ps_test[c].dropna().median() for c in
                ['hourly_wape', 'hourly_wpe', 'daily_wape', 'daily_wpe']}

    elapsed = time.time() - t0

    # Save
    safe_name = 'mlp_nolags' if not use_lags else 'mlp_m5lags'
    ps_val.to_parquet(os.path.join(RESULTS_DIR, f'{safe_name}_val_per_series.parquet'), index=False)
    ps_test.to_parquet(os.path.join(RESULTS_DIR, f'{safe_name}_test_per_series.parquet'), index=False)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f'{safe_name}.pt'))

    all_results[variant_label] = {
        'val': {'pooled': pooled_val, 'median': med_val},
        'test': {'pooled': pooled_test, 'median': med_test},
        'best_epoch': best_epoch, 'n_params': n_params, 'elapsed': elapsed,
    }

    print(f'  Test: WAPE_h pool={pooled_test["hourly_wape"]:.4f}, '
          f'med={med_test["hourly_wape"]:.4f}, time={elapsed:.0f}s')

    del train_data, val_data, test_data, model, train_ds, train_loader
    del val_cat_t, val_cont_t, val_lags_t
    gc.collect()
    if DEVICE == 'mps':
        torch.mps.empty_cache()


# ===========================================================================
# SUMMARY
# ===========================================================================
print('\n' + '=' * 72)
print('  RIEPILOGO MLP (test, in-stock only)')
print('=' * 72)

print(f'\n  {"Model":<20} '
      f'{"WAPE_h pool":>12} {"WPE_h pool":>11} '
      f'{"WAPE_h med":>11} {"WPE_h med":>10} '
      f'{"WAPE_d pool":>12}')
print('  ' + '-' * 80)

for label, res in all_results.items():
    p = res['test']['pooled']
    m = res['test']['median']
    print(f'  {label:<20} '
          f'{p["hourly_wape"]:>12.4f} {p["hourly_wpe"]:>11.4f} '
          f'{m["hourly_wape"]:>11.4f} {m["hourly_wpe"]:>10.4f} '
          f'{p["daily_wape"]:>12.4f}')

print('\n' + '=' * 72)
print('  DONE — 04b_baseline_mlp.py')
print('=' * 72)
