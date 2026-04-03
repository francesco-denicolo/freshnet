"""
08c_mlp_single.py — Train MLP M5-lags per un singolo imputer
==============================================================
Eseguito come processo separato per ogni imputer (evita OOM).

Usage: freshnet/bin/python notebooks_final/08c_mlp_single.py <imputer_key>
  imputer_key: media_cond | media_glob | mediana_cond | lgb
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# ---------------------------------------------------------------------------
# Get imputer from command line
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    print('Usage: python 08c_mlp_single.py <imputer_key>')
    sys.exit(1)

IMP_KEY = sys.argv[1]
IMP_LABELS = {'media_cond': 'Media condizionata', 'media_glob': 'Media globale',
              'mediana_cond': 'Mediana condizionata', 'lgb': 'LGB imputer'}
IMP_LABEL = IMP_LABELS[IMP_KEY]

cell_key = f'{IMP_KEY}__mlp_m5lags'
out_parquet = os.path.join(RESULTS_DIR, f'{cell_key}_test_per_series.parquet')

if os.path.exists(out_parquet):
    print(f'SKIP: {out_parquet} already exists')
    sys.exit(0)

print(f'=== MLP (M5 lags) × {IMP_LABEL} ({IMP_KEY}) ===')
print(f'Device: {DEVICE}')

# ---------------------------------------------------------------------------
# 1. Load data and build series cache
# ---------------------------------------------------------------------------
print('\n1. Loading data...')
df_train_hf = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_eval = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_eval.parquet'))

df_train_hf['dt_parsed'] = pd.to_datetime(df_train_hf['dt'])
df_eval['dt_parsed'] = pd.to_datetime(df_eval['dt'])

df_full = pd.concat([df_train_hf, df_eval], ignore_index=True)
df_full = df_full.sort_values(['store_id', 'product_id', 'dt_parsed']).reset_index(drop=True)

all_dates = sorted(df_full['dt_parsed'].unique())
date_to_day = {d: i + 1 for i, d in enumerate(all_dates)}
df_full['day_num'] = df_full['dt_parsed'].map(date_to_day)
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek

sales_orig = np.array(df_full['hours_sale'].tolist(), dtype=np.float32)
stock_orig = np.array(df_full['hours_stock_status'].tolist(), dtype=np.float32)

del df_train_hf, df_eval

# Load completed_sales and align
print('  Loading completed_sales...')
df_cs = pd.read_parquet(os.path.join(COMPLETED_DIR, f'{IMP_KEY}.parquet'))
cs_sales = np.array(df_cs['hours_sale'].tolist(), dtype=np.float32)

completed_full = sales_orig.copy()
cs_keys = (df_cs['store_id'].astype(str) + '_' + df_cs['product_id'].astype(str) + '_' + df_cs['dt']).values
full_keys = (df_full['store_id'].astype(str) + '_' + df_full['product_id'].astype(str) + '_' + df_full['dt']).values
key_to_idx = dict(zip(cs_keys, range(len(df_cs))))

for i in range(len(df_full)):
    k = full_keys[i]
    if k in key_to_idx:
        completed_full[i] = cs_sales[key_to_idx[k]]

del df_cs, cs_sales, cs_keys, full_keys, key_to_idx
gc.collect()

# Build series cache
print('  Building series cache...')
series_cache = {}
for (sid, pid), grp in df_full.groupby(['store_id', 'product_id'], sort=False):
    grp_s = grp.sort_values('day_num')
    idx = grp_s.index.values
    series_cache[(sid, pid)] = {
        'days': grp_s['day_num'].values,
        'dows': grp_s['dow'].values,
        'sales_completed': completed_full[idx],
        'sales_orig': sales_orig[idx],
        'stock': stock_orig[idx],
        'city_id': grp_s['city_id'].values[0],
        'conts': grp_s[CONT_FEATURES].values.astype(np.float32),
    }
print(f'  {len(series_cache):,} series')

# Free large arrays - we have everything in series_cache now
del df_full, sales_orig, stock_orig, completed_full
gc.collect()


# ---------------------------------------------------------------------------
# 2. Lag computation
# ---------------------------------------------------------------------------
def compute_lags_for_day(avail_sales, avail_dows, target_dow, K):
    z = np.float32
    lags = {n: np.full(24, np.nan, dtype=z) for n in LAG_FEATURE_NAMES}
    if K == 0:
        return lags
    lags['lag_1d'] = avail_sales[-1]
    if K >= 7:  lags['lag_7d'] = avail_sales[-7]
    if K >= 14: lags['lag_14d'] = avail_sales[-14]
    if K >= 7:  lags['rmean_7d'] = avail_sales[-7:].mean(axis=0)
    if K >= 14: lags['rmean_14d'] = avail_sales[-14:].mean(axis=0)
    if K >= 2:  lags['rstd_7d'] = avail_sales[-min(7,K):].std(axis=0)
    same_dow = avail_dows == target_dow
    if same_dow.any():
        ds = avail_sales[same_dow]
        lags['lag_dow'] = ds[-1]
        lags['rmean_dow'] = ds.mean(axis=0)
    dt = avail_sales.sum(axis=1)
    lags['daily_total_lag1'] = np.full(24, dt[-1], dtype=z)
    if K >= 7: lags['daily_total_rmean7'] = np.full(24, dt[-7:].mean(), dtype=z)
    rm7, l1 = lags['rmean_7d'], lags['lag_1d']
    if not np.isnan(rm7).all():
        v = (~np.isnan(l1)) & (~np.isnan(rm7)) & (rm7 > 0)
        if v.any():
            mom = np.full(24, np.nan, dtype=z)
            mom[v] = l1[v] / rm7[v]
            lags['momentum_1d_7d'] = mom
    return lags


# ---------------------------------------------------------------------------
# 3. Build dataset for one split
# ---------------------------------------------------------------------------
def build_dataset(split, cont_mean=None, cont_std=None, lag_mean=None, lag_std=None):
    if split == 'train':
        d_min, d_max = 2, 83
    elif split == 'val':
        d_min, d_max = 84, 90
    else:
        d_min, d_max = 91, 97

    cat_l, cont_l, lag_l, tgt_l, stk_l, sid_l, pid_l = [], [], [], [], [], [], []

    n_done = 0
    for (sid, pid), sd in series_cache.items():
        n_done += 1
        if n_done % 10000 == 0:
            print(f'    ... {n_done:,}/{len(series_cache):,}')

        days, dows = sd['days'], sd['dows']
        sc, so, stk = sd['sales_completed'], sd['sales_orig'], sd['stock']
        city, conts = sd['city_id'], sd['conts']

        for idx in range(len(days)):
            d = days[idx]
            if d < d_min or d > d_max:
                continue

            a_day = d - 1 if split == 'train' else (83 if split == 'val' else 90)
            cat_l.append([sid, pid, city, dows[idx]])
            cont_l.append(conts[idx])
            tgt_l.append(so[idx])
            stk_l.append(stk[idx])
            sid_l.append(sid)
            pid_l.append(pid)

            avail = days <= a_day
            K = int(avail.sum())
            ld = compute_lags_for_day(sc[avail], dows[avail], dows[idx], K) if K > 0 \
                else {n: np.full(24, np.nan, dtype=np.float32) for n in LAG_FEATURE_NAMES}

            fa, masks = [], np.zeros(11, dtype=np.float32)
            for fi, name in enumerate(LAG_FEATURE_NAMES):
                arr = ld[name]
                if not np.isnan(arr).all():
                    masks[fi] = 1.0
                    fa.append(np.where(np.isnan(arr), 0.0, arr).astype(np.float32))
                else:
                    fa.append(np.zeros(24, dtype=np.float32))
            fa.append(masks)
            lag_l.append(np.concatenate(fa))

    cat_arr = np.array(cat_l, dtype=np.int64)
    cont_arr = np.array(cont_l, dtype=np.float32)
    tgt_arr = np.array(tgt_l, dtype=np.float32)
    stk_arr = np.array(stk_l, dtype=np.float32)
    lag_arr = np.array(lag_l, dtype=np.float32)

    if cont_mean is None:
        cont_mean = cont_arr.mean(axis=0)
        cont_std = cont_arr.std(axis=0)
        cont_std[cont_std < 1e-8] = 1.0
    cont_arr = (cont_arr - cont_mean) / cont_std

    if lag_mean is None:
        lag_mean = lag_arr.mean(axis=0)
        lag_std = lag_arr.std(axis=0)
        lag_std[lag_std < 1e-8] = 1.0
    lag_arr = (lag_arr - lag_mean) / lag_std

    return {
        'cat': cat_arr, 'cont': cont_arr, 'lags': lag_arr,
        'targets': tgt_arr, 'stock': stk_arr,
        'store_ids': np.array(sid_l, dtype=np.int64),
        'product_ids': np.array(pid_l, dtype=np.int64),
        'cont_mean': cont_mean, 'cont_std': cont_std,
        'lag_mean': lag_mean, 'lag_std': lag_std,
    }


# ---------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------
class DS(Dataset):
    def __init__(self, c, co, l, t):
        self.c, self.co, self.l, self.t = (
            torch.from_numpy(c), torch.from_numpy(co),
            torch.from_numpy(l), torch.from_numpy(t))
    def __len__(self): return len(self.t)
    def __getitem__(self, i): return self.c[i], self.co[i], self.l[i], self.t[i]


class MLP(nn.Module):
    def __init__(self, n_cont, n_lags):
        super().__init__()
        self.embs = nn.ModuleDict({
            n: nn.Embedding(CARDINALITIES[n], EMB_DIMS[n]) for n in EMB_DIMS})
        self.names = ['store_id', 'product_id', 'city_id', 'dow']
        inp = sum(EMB_DIMS.values()) + n_cont + n_lags
        layers = []
        for h in HIDDEN:
            layers += [nn.Linear(inp, h), nn.ReLU()]
            inp = h
        layers += [nn.Linear(inp, 24), nn.Softplus()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat, cont, lags):
        e = [self.embs[n](cat[:, i]) for i, n in enumerate(self.names)]
        x = torch.cat(e + [cont, lags], dim=1)
        return self.mlp(x)


# ---------------------------------------------------------------------------
# 5. Build datasets, then free series_cache
# ---------------------------------------------------------------------------
print('\n2. Building datasets...')
t0 = time.time()

print('  Train:')
tr = build_dataset('train')
print(f'  Train: {len(tr["targets"]):,} samples, lags={tr["lags"].shape[1]}')

print('  Val:')
va = build_dataset('val', tr['cont_mean'], tr['cont_std'], tr['lag_mean'], tr['lag_std'])
print(f'  Val: {len(va["targets"]):,} samples')

print('  Test:')
te = build_dataset('test', tr['cont_mean'], tr['cont_std'], tr['lag_mean'], tr['lag_std'])
print(f'  Test: {len(te["targets"]):,} samples')

# FREE series_cache — no longer needed
del series_cache
gc.collect()
print(f'  Datasets built in {time.time()-t0:.0f}s, series_cache freed')

# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
print('\n3. Training...')
model = MLP(tr['cont'].shape[1], tr['lags'].shape[1]).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f'  Model: {n_params:,} params')

ds = DS(tr['cat'], tr['cont'], tr['lags'], tr['targets'])
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Free train raw arrays (DataLoader holds them via DS)
del tr
gc.collect()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
val_instock = va['stock'] == 0
vc = torch.from_numpy(va['cat']).to(DEVICE)
vco = torch.from_numpy(va['cont']).to(DEVICE)
vl = torch.from_numpy(va['lags']).to(DEVICE)

best_wape, best_ep, best_state = float('inf'), 0, None
no_imp = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    tl, nb = 0., 0
    for c, co, l, t in loader:
        c, co, l, t = c.to(DEVICE), co.to(DEVICE), l.to(DEVICE), t.to(DEVICE)
        p = model(c, co, l)
        loss = nn.functional.mse_loss(p, t)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tl += loss.item(); nb += 1

    model.eval()
    with torch.no_grad():
        ap = []
        for s in range(0, len(vc), 10000):
            e = min(s+10000, len(vc))
            ap.append(model(vc[s:e], vco[s:e], vl[s:e]).cpu().numpy())
        vp = np.concatenate(ap)

    w = np.abs(vp[val_instock] - va['targets'][val_instock]).sum() / \
        np.abs(va['targets'][val_instock]).sum()

    print(f'  Epoch {epoch:3d}: loss={tl/nb:.6f}, val_WAPE={w:.6f}')

    if w < best_wape:
        best_wape, best_ep = w, epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_imp = 0
    else:
        no_imp += 1
    if no_imp >= PATIENCE:
        print(f'  Early stop (best={best_ep}, WAPE={best_wape:.6f})')
        break

if best_state:
    model.load_state_dict(best_state)
model.to(DEVICE)

# ---------------------------------------------------------------------------
# 7. Evaluate test
# ---------------------------------------------------------------------------
print('\n4. Evaluating test...')
model.eval()
tc = torch.from_numpy(te['cat']).to(DEVICE)
tco = torch.from_numpy(te['cont']).to(DEVICE)
tlg = torch.from_numpy(te['lags']).to(DEVICE)

ap = []
with torch.no_grad():
    for s in range(0, len(tc), 10000):
        e = min(s+10000, len(tc))
        ap.append(model(tc[s:e], tco[s:e], tlg[s:e]).cpu().numpy())
preds = np.concatenate(ap)

instock = te['stock'] == 0
p_h, o_h = preds[instock], te['targets'][instock]
sae_h, sao_h = np.abs(p_h - o_h).sum(), np.abs(o_h).sum()
se_h, so_h = (p_h - o_h).sum(), o_h.sum()

n_s = preds.shape[0]
sae_d, sao_d, se_d, so_d = 0., 0., 0., 0.
for d in range(n_s):
    m = instock[d]
    if m.any():
        pv, ov = preds[d, m].sum(), te['targets'][d, m].sum()
        sae_d += abs(pv-ov); sao_d += abs(ov); se_d += pv-ov; so_d += ov

pooled = {
    'hourly_wape': sae_h/sao_h, 'hourly_wpe': se_h/so_h,
    'daily_wape': sae_d/sao_d, 'daily_wpe': se_d/so_d,
}

# Per-series
sm = {}
for i in range(n_s):
    k = (te['store_ids'][i], te['product_ids'][i])
    if k not in sm: sm[k] = []
    sm[k].append(i)

records = []
for (sid, pid), idxs in sm.items():
    sh, aoh, eh, oh = 0., 0., 0., 0.
    sd2, aod, ed, od, nvd, ni = 0., 0., 0., 0., 0, 0
    for i in idxs:
        m = instock[i]; ni += int(m.sum())
        sh += np.abs(preds[i,m]-te['targets'][i,m]).sum()
        aoh += np.abs(te['targets'][i,m]).sum()
        eh += (preds[i,m]-te['targets'][i,m]).sum()
        oh += te['targets'][i,m].sum()
        if m.any():
            pv, ov = preds[i,m].sum(), te['targets'][i,m].sum()
            sd2 += abs(pv-ov); aod += abs(ov); ed += pv-ov; od += ov; nvd += 1
    records.append({'store_id': sid, 'product_id': pid,
                    'hourly_wape': sh/aoh if aoh>0 else np.nan,
                    'hourly_wpe': eh/oh if oh!=0 else np.nan,
                    'daily_wape': sd2/aod if aod>0 else np.nan,
                    'daily_wpe': ed/od if od!=0 else np.nan,
                    'n_hours_instock': ni, 'n_days_valid': nvd})

ps_df = pd.DataFrame(records)
ps_df.to_parquet(out_parquet, index=False)

med = {c: ps_df[c].dropna().median() for c in ['hourly_wape', 'hourly_wpe']}

print(f'\n  WAPE_h pool={pooled["hourly_wape"]:.4f}, med={med["hourly_wape"]:.4f}')
print(f'  WPE_h pool={pooled["hourly_wpe"]:.4f}, med={med["hourly_wpe"]:.4f}')
print(f'  Salvato: {out_parquet}')
print('DONE')
