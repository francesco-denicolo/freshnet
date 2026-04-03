"""
04_fase_a_mlp.py — Fase A: MLP Forecasting su dati sporchi
============================================================
Piano: CLAUDE_SEQUENTIAL-2.md, Fase A, punto (2)

MLP su dati sporchi (S_obs con stockout), due varianti:
  A: solo embeddings + 7 covariate continue (83 dim input)
  F: + 11 M5-style lag features (264 valori + 11 mask = 275 dim extra, tot 358)

Workflow:
  1. TUNING:     Train gg 2-83, val gg 84-90 → best epoch per variante
  2. SELECT:     Scelta miglior variante (val WAPE pooled)
  3. RETRAINING: Retrain su gg 2-90 per best_epoch epoche (no early stopping)
  4. TEST:       Eval su gg 91-97 (eval HF)

Architettura:
  Embeddings: store(32) + product(32) + city(8) + dow(4) = 76
  MLP: [input_dim, 128, 64, 24] + Softplus
  Loss: MSE su tutte le 24 ore

Output:
  notebooks/v2/results/mlp_{a|f}_{val|test}_per_series.parquet
  notebooks/v2/results/mlp_variant_{A|F}.pt
  notebooks/v2/results/mlp_variant_{A|F}_retrained.pt

Eseguire con: freshnet/bin/python notebooks/v2/04_fase_a_mlp.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import functools

print = functools.partial(print, flush=True)

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.evaluation.metrics import compute_metrics

# ---- Config ----
DEVICE = 'mps' if torch.backends.mps.is_available() else (
    'cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE = 4096
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
HIDDEN_SIZES = [128, 64]

EMB_DIMS = {'store_id': 32, 'product_id': 32, 'city_id': 8, 'dow': 4}
CARDINALITIES = {'store_id': 898, 'product_id': 865, 'city_id': 18, 'dow': 7}

CONT_COLS = ['discount', 'avg_temperature', 'avg_humidity', 'precpt',
             'avg_wind_level', 'holiday_flag', 'activity_flag']

print("=" * 72)
print("  FASE A — MLP FORECASTING (dati sporchi)")
print("=" * 72)
print(f"  Device: {DEVICE}")

# =========================================================================
# 1. Caricamento dati
# =========================================================================
t0 = time.time()
print("\n1. Caricamento dati...")
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

n_series = df_full.groupby(['store_id', 'product_id']).ngroups
print(f"  Train: {len(df_train):,}, Eval: {len(df_eval):,}")
print(f"  Full: {len(df_full):,}, {len(all_dates)} giorni, {n_series:,} serie")
print(f"  Tempo loading: {time.time()-t0:.1f}s")

del df_train, df_eval

# =========================================================================
# 2. Build series cache
# =========================================================================
print("\n2. Building series cache...")
t1 = time.time()

series_data = {}
for (sid, pid), grp in df_full.groupby(['store_id', 'product_id'], sort=False):
    grp_s = grp.sort_values('day_num')
    series_data[(sid, pid)] = {
        'days': grp_s['day_num'].values,
        'dows': grp_s['dow'].values,
        'sales': np.array(grp_s['hours_sale'].tolist(), dtype=np.float32),
        'stock': np.array(grp_s['hours_stock_status'].tolist(), dtype=np.float32),
        'city_id': grp_s['city_id'].values[0],
        'conts': grp_s[CONT_COLS].values.astype(np.float32),
    }

print(f"  {len(series_data):,} serie in {time.time()-t1:.0f}s")


# =========================================================================
# 3. Lag features computation
# =========================================================================
def _compute_lag_features_f(sales, days, dows, target_day_idx, anchor_day):
    """Compute M5-style lag features: 11 × 24 values + 11 binary masks = 275."""
    z = np.float32
    feats = [np.zeros(24, dtype=z) for _ in range(11)]
    masks = np.zeros(11, dtype=z)

    avail_mask = days <= anchor_day
    K = int(avail_mask.sum())

    if K > 0:
        avail_sales = sales[avail_mask]
        avail_dows = dows[avail_mask]
        target_dow = dows[target_day_idx]

        # lag_1d
        feats[0][:] = avail_sales[-1]
        masks[0] = 1.0

        # lag_7d
        if K >= 7:
            feats[1][:] = avail_sales[-7]
            masks[1] = 1.0

        # lag_14d
        if K >= 14:
            feats[2][:] = avail_sales[-14]
            masks[2] = 1.0

        # rmean_7d
        if K >= 7:
            feats[3][:] = avail_sales[-7:].mean(axis=0)
            masks[3] = 1.0

        # rmean_14d
        if K >= 14:
            feats[4][:] = avail_sales[-14:].mean(axis=0)
            masks[4] = 1.0

        # rstd_7d
        if K >= 2:
            w = min(7, K)
            feats[5][:] = avail_sales[-w:].std(axis=0)
            masks[5] = 1.0

        # DoW-specific
        same_dow = avail_dows == target_dow
        if same_dow.any():
            dow_sales = avail_sales[same_dow]
            feats[6][:] = dow_sales[-1]
            feats[7][:] = dow_sales.mean(axis=0)
            masks[6] = 1.0
            masks[7] = 1.0

        # Daily aggregates
        daily_totals = avail_sales.sum(axis=1)
        feats[8][:] = daily_totals[-1]
        masks[8] = 1.0

        if K >= 7:
            feats[9][:] = daily_totals[-7:].mean()
            masks[9] = 1.0

        # Momentum
        if masks[3] == 1.0:
            rm7 = feats[3]
            valid_h = rm7 > 0
            if valid_h.any():
                feats[10][valid_h] = feats[0][valid_h] / rm7[valid_h]
                masks[10] = 1.0

    return np.concatenate(feats + [masks])


# =========================================================================
# 4. Build dataset arrays
# =========================================================================
def build_dataset_arrays(series_data, d_min, d_max, anchor_mode='rolling',
                         anchor_day=None, with_lags=False,
                         cont_mean=None, cont_std=None,
                         lag_mean=None, lag_std=None):
    """Build arrays for MLP training/evaluation.

    anchor_mode='rolling': anchor = d-1 (for training)
    anchor_mode='fixed': anchor = anchor_day (for val/test)
    """
    cat_list, cont_list, lag_list = [], [], []
    target_list, stock_list = [], []
    sid_list, pid_list = [], []

    for (sid, pid), sd in series_data.items():
        days = sd['days']
        dows = sd['dows']
        sales = sd['sales']
        stock = sd['stock']
        city = sd['city_id']
        conts = sd['conts']

        for idx in range(len(days)):
            d = days[idx]
            if d < d_min or d > d_max:
                continue

            a_day = d - 1 if anchor_mode == 'rolling' else anchor_day

            cat_list.append([sid, pid, city, dows[idx]])
            cont_list.append(conts[idx])

            if with_lags:
                lag_list.append(_compute_lag_features_f(
                    sales, days, dows, idx, a_day))
            else:
                lag_list.append(np.zeros(0, dtype=np.float32))

            target_list.append(sales[idx])
            stock_list.append(stock[idx])
            sid_list.append(sid)
            pid_list.append(pid)

    cat_arr = np.array(cat_list, dtype=np.int64)
    cont_arr = np.array(cont_list, dtype=np.float32)
    target_arr = np.array(target_list, dtype=np.float32)
    stock_arr = np.array(stock_list, dtype=np.float32)

    if with_lags and len(lag_list) > 0:
        lag_arr = np.array(lag_list, dtype=np.float32)
    else:
        lag_arr = np.zeros((len(cat_list), 0), dtype=np.float32)

    # Normalize continuous features
    if cont_mean is None:
        cont_mean = cont_arr.mean(axis=0)
        cont_std = cont_arr.std(axis=0)
        cont_std[cont_std < 1e-8] = 1.0
    cont_arr = (cont_arr - cont_mean) / cont_std

    # Normalize lag features
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


# =========================================================================
# 5. Model definition
# =========================================================================
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
    def __init__(self, n_cont, n_lags, emb_dims, cardinalities, hidden_sizes):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(cardinalities[name], emb_dims[name])
            for name in emb_dims
        })
        self.emb_names = ['store_id', 'product_id', 'city_id', 'dow']

        total_emb = sum(emb_dims.values())
        input_dim = total_emb + n_cont + n_lags

        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
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


def predict(model, data, device, chunk_size=10000):
    """Generate predictions in chunks."""
    model.eval()
    cat_t = torch.from_numpy(data['cat'])
    cont_t = torch.from_numpy(data['cont'])
    lags_t = torch.from_numpy(data['lags'])

    all_preds = []
    with torch.no_grad():
        for s in range(0, len(cat_t), chunk_size):
            e = min(s + chunk_size, len(cat_t))
            p = model(cat_t[s:e].to(device), cont_t[s:e].to(device),
                      lags_t[s:e].to(device))
            all_preds.append(p.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def train_model(model, train_loader, val_data, device, lr, max_epochs,
                patience, verbose_every=5):
    """Train with early stopping on val WAPE pooled."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_wape = float('inf')
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for cat, cont, lags, targets in train_loader:
            preds = model(cat.to(device), cont.to(device), lags.to(device))
            loss = nn.functional.mse_loss(preds, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        avg_loss = train_loss / n_batches

        # Val WAPE
        val_preds = predict(model, val_data, device)
        sae = np.abs(val_preds - val_data['targets']).sum()
        sao = np.abs(val_data['targets']).sum()
        val_wape = sae / sao if sao > 0 else float('inf')

        if epoch % verbose_every == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}: loss={avg_loss:.6f}, val_WAPE={val_wape:.6f}")

        if val_wape < best_val_wape:
            best_val_wape = val_wape
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch} "
                  f"(best={best_epoch}, WAPE={best_val_wape:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return best_val_wape, best_epoch


def train_fixed_epochs(model, train_loader, device, lr, n_epochs):
    """Train for a fixed number of epochs (no validation, for retraining)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for cat, cont, lags, targets in train_loader:
            preds = model(cat.to(device), cont.to(device), lags.to(device))
            loss = nn.functional.mse_loss(preds, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{n_epochs}: loss={train_loss/n_batches:.6f}")

    return model


def evaluate_model(model, data, device):
    """Compute pooled + per-series metrics."""
    preds = predict(model, data, device)
    obs = data['targets']
    stock = data['stock']
    sids = data['store_ids']
    pids = data['product_ids']

    p_flat = preds.ravel()
    o_flat = obs.ravel()
    s_flat = stock.ravel()
    err = p_flat - o_flat

    pooled = {}
    for sub, smask in [('overall', np.ones(len(p_flat), dtype=bool)),
                       ('instock', s_flat == 0),
                       ('stockout', s_flat == 1)]:
        ef = err[smask]
        of = o_flat[smask]
        sao = np.abs(of).sum()
        so = of.sum()
        pooled[f'wape_{sub}'] = np.abs(ef).sum() / sao if sao > 0 else np.nan
        pooled[f'wpe_{sub}'] = ef.sum() / so if so != 0 else np.nan
        pooled[f'n_{sub}'] = int(smask.sum())

    # Per-series
    df_idx = pd.DataFrame({'sid': sids, 'pid': pids, 'row': np.arange(len(sids))})
    records = []
    for (sid, pid), grp in df_idx.groupby(['sid', 'pid']):
        idx = grp['row'].values
        m = compute_metrics(preds[idx], obs[idx], stock[idx])
        m['store_id'] = sid
        m['product_id'] = pid
        records.append(m)

    return pooled, pd.DataFrame(records)


# =========================================================================
# 6. Train both variants
# =========================================================================
VARIANTS = ['A', 'F']
variant_results = {}

for variant in VARIANTS:
    with_lags = (variant == 'F')
    print(f"\n{'='*72}")
    print(f"  VARIANTE {variant} {'(base 83 dim)' if variant == 'A' else '(M5 lags 358 dim)'}")
    print(f"{'='*72}")

    # ---- Build train/val ----
    print(f"\n  6a. Building datasets (variant {variant})...")
    t2 = time.time()

    train_data = build_dataset_arrays(
        series_data, 2, 83, anchor_mode='rolling', with_lags=with_lags)
    val_data = build_dataset_arrays(
        series_data, 84, 90, anchor_mode='fixed', anchor_day=83,
        with_lags=with_lags,
        cont_mean=train_data['cont_mean'], cont_std=train_data['cont_std'],
        lag_mean=train_data['lag_mean'], lag_std=train_data['lag_std'])

    n_cont = train_data['cont'].shape[1]
    n_lags = train_data['lags'].shape[1]
    print(f"    Train: {len(train_data['targets']):,}, Val: {len(val_data['targets']):,}")
    print(f"    Input dim: {sum(EMB_DIMS.values()) + n_cont + n_lags} "
          f"(emb={sum(EMB_DIMS.values())}, cont={n_cont}, lags={n_lags})")
    print(f"    Build time: {time.time()-t2:.0f}s")

    # ---- Train with early stopping ----
    print(f"\n  6b. Training variant {variant}...")
    t3 = time.time()

    torch.manual_seed(SEED)
    model = RetailMLP(n_cont, n_lags, EMB_DIMS, CARDINALITIES, HIDDEN_SIZES)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model params: {n_params:,}")

    train_ds = RetailDataset(train_data['cat'], train_data['cont'],
                              train_data['lags'], train_data['targets'])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=False)

    best_wape, best_epoch = train_model(
        model, train_loader, val_data, DEVICE, LR, MAX_EPOCHS, PATIENCE)

    print(f"    Best epoch: {best_epoch}, val WAPE: {best_wape:.6f}")
    print(f"    Training time: {time.time()-t3:.0f}s")

    # Save tuned model
    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, f'mlp_variant_{variant}.pt'))

    # ---- Val evaluation ----
    print(f"\n  6c. Val evaluation (variant {variant})...")
    pooled_val, ps_val = evaluate_model(model, val_data, DEVICE)
    out_path = os.path.join(RESULTS_DIR, f'mlp_{variant.lower()}_val_per_series.parquet')
    ps_val.to_parquet(out_path, index=False)
    print(f"    WAPE_in pooled: {pooled_val['wape_instock']:.4f}")
    print(f"    WAPE_in median: {ps_val['wape_instock'].median():.4f}")
    print(f"    WPE_in pooled:  {pooled_val['wpe_instock']:.4f}")

    # ---- Retrain on days 2-90 ----
    print(f"\n  6d. Retraining on days 2-90 ({best_epoch} epochs, variant {variant})...")
    t4 = time.time()

    retrain_data = build_dataset_arrays(
        series_data, 2, 90, anchor_mode='rolling', with_lags=with_lags)

    torch.manual_seed(SEED)
    model_rt = RetailMLP(n_cont, n_lags, EMB_DIMS, CARDINALITIES, HIDDEN_SIZES)
    retrain_ds = RetailDataset(retrain_data['cat'], retrain_data['cont'],
                                retrain_data['lags'], retrain_data['targets'])
    retrain_loader = DataLoader(retrain_ds, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=0, pin_memory=False)

    model_rt = train_fixed_epochs(model_rt, retrain_loader, DEVICE, LR, best_epoch)
    print(f"    Retrain time: {time.time()-t4:.0f}s")

    torch.save(model_rt.state_dict(),
               os.path.join(RESULTS_DIR, f'mlp_variant_{variant}_retrained.pt'))

    # ---- Test evaluation ----
    print(f"\n  6e. Test evaluation (variant {variant})...")
    test_data = build_dataset_arrays(
        series_data, 91, 97, anchor_mode='fixed', anchor_day=90,
        with_lags=with_lags,
        cont_mean=retrain_data['cont_mean'], cont_std=retrain_data['cont_std'],
        lag_mean=retrain_data['lag_mean'], lag_std=retrain_data['lag_std'])

    pooled_test, ps_test = evaluate_model(model_rt, test_data, DEVICE)
    out_path = os.path.join(RESULTS_DIR, f'mlp_{variant.lower()}_test_per_series.parquet')
    ps_test.to_parquet(out_path, index=False)
    print(f"    WAPE_in pooled: {pooled_test['wape_instock']:.4f}")
    print(f"    WAPE_in median: {ps_test['wape_instock'].median():.4f}")
    print(f"    WPE_in pooled:  {pooled_test['wpe_instock']:.4f}")
    print(f"    WPE_in median:  {ps_test['wpe_instock'].median():.4f}")

    variant_results[variant] = {
        'best_epoch': best_epoch,
        'best_wape': best_wape,
        'n_params': n_params,
        'pooled_val': pooled_val,
        'pooled_test': pooled_test,
        'ps_val': ps_val,
        'ps_test': ps_test,
    }

    del model, model_rt, train_data, val_data, retrain_data, test_data
    del train_ds, retrain_ds, train_loader, retrain_loader
    if DEVICE == 'mps':
        torch.mps.empty_cache()


# =========================================================================
# 7. Variant selection
# =========================================================================
print(f"\n{'='*72}")
print("  7. VARIANT SELECTION")
print(f"{'='*72}")

print(f"\n  {'Var':>4} {'Epochs':>8} {'Params':>10} {'Val_WAPE_pool':>16} "
      f"{'Val_WAPE_in_m':>16}")
print("  " + "-" * 60)

for v in VARIANTS:
    vr = variant_results[v]
    print(f"  {v:>4} {vr['best_epoch']:>8} {vr['n_params']:>10,} "
          f"{vr['best_wape']:>16.6f} "
          f"{vr['ps_val']['wape_instock'].median():>16.4f}")

best_var = min(VARIANTS, key=lambda v: variant_results[v]['pooled_val']['wape_instock'])
print(f"\n  Best variant (val WAPE_in pooled): {best_var}")


# =========================================================================
# 8. Confronto finale
# =========================================================================
print(f"\n{'='*72}")
print("  8. CONFRONTO FINALE (TEST)")
print(f"{'='*72}")

print(f"\n  {'Modello':<20} {'WAPE_in_pool':>14} {'WAPE_in_med':>14} "
      f"{'WPE_in_pool':>14} {'WPE_in_med':>14} {'WAPE_all_med':>14}")
print("  " + "-" * 94)

# Load naive baselines
naive_models = {
    'Global Mean': 'global_mean',
    'DoW Mean': 'dow_mean',
    'Naive Direct': 'naive_direct',
    'MA Direct': 'ma_direct',
}
for label, prefix in naive_models.items():
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if os.path.exists(path):
        ps = pd.read_parquet(path)
        print(f"  {label:<20} {'N/A':>14} {ps['wape_instock'].median():>14.4f} "
              f"{'N/A':>14} {ps['wpe_instock'].median():>14.4f} "
              f"{ps['wape_overall'].median():>14.4f}")

# LGB results
for v in ['A', 'F']:
    path = os.path.join(RESULTS_DIR, f'lgb_{v.lower()}_test_per_series.parquet')
    if os.path.exists(path):
        ps = pd.read_parquet(path)
        print(f"  {'LGB ' + v:<20} {'N/A':>14} {ps['wape_instock'].median():>14.4f} "
              f"{'N/A':>14} {ps['wpe_instock'].median():>14.4f} "
              f"{ps['wape_overall'].median():>14.4f}")

# MLP results
for v in VARIANTS:
    vr = variant_results[v]
    pr = vr['pooled_test']
    ps = vr['ps_test']
    label = f"MLP {v}"
    print(f"  {label:<20} {pr['wape_instock']:>14.4f} {ps['wape_instock'].median():>14.4f} "
          f"{pr['wpe_instock']:>14.4f} {ps['wpe_instock'].median():>14.4f} "
          f"{ps['wape_overall'].median():>14.4f}")

print(f"\n  Tempo totale: {time.time()-t0:.0f}s")
print("=" * 72)
