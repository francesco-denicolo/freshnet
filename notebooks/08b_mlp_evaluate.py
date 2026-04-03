"""
08b_mlp_evaluate.py — Evaluate MLP Variant A + Compare All Baselines
=====================================================================
Carica il modello MLP (variante A, no history) salvato, valuta su val/test,
salva parquet per-serie, genera tabelle e figure di confronto.

Eseguire con: freshnet/bin/python notebooks/08b_mlp_evaluate.py
"""

import sys
import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from src.evaluation.metrics import compute_metrics, format_metrics_table

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = 'mps' if torch.backends.mps.is_available() else (
    'cuda' if torch.cuda.is_available() else 'cpu')

CONT_FEATURES = ['discount', 'avg_temperature', 'avg_humidity',
                  'precpt', 'avg_wind_level', 'holiday_flag', 'activity_flag']

EMB_DIMS = {'store_id': 32, 'product_id': 32, 'city_id': 8, 'dow': 4}
CARDINALITIES = {'store_id': 898, 'product_id': 865, 'city_id': 18, 'dow': 7}
HIDDEN_SIZES = [128, 64]

# ---------------------------------------------------------------------------
# Model definition (same as 08)
# ---------------------------------------------------------------------------
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
        emb_list = [self.embeddings[name](cat[:, i]) for i, name in enumerate(self.emb_names)]
        x = torch.cat(emb_list + [cont], dim=1)
        if lags.shape[1] > 0:
            x = torch.cat([x, lags], dim=1)
        return self.mlp(x)

# ===========================================================================
print('=' * 72)
print('  MLP VARIANT A — EVALUATION & COMPARISON')
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

# ---------------------------------------------------------------------------
# 2. Build arrays per split (variant A = no lag features)
# ---------------------------------------------------------------------------
print('\n2. Preparazione dati per split...')


def build_split_arrays(df_full, d_min, d_max, cont_mean=None, cont_std=None):
    """Build feature arrays for a split. Variant A = no lags."""
    mask = (df_full['day_num'] >= d_min) & (df_full['day_num'] <= d_max)
    df_split = df_full[mask].copy()

    cat = df_split[['store_id', 'product_id', 'city_id', 'dow']].values.astype(np.int64)
    cont = df_split[CONT_FEATURES].values.astype(np.float32)
    targets = np.array(df_split['hours_sale'].tolist(), dtype=np.float32)
    stock = np.array(df_split['hours_stock_status'].tolist(), dtype=np.float32)
    store_ids = df_split['store_id'].values.astype(np.int64)
    product_ids = df_split['product_id'].values.astype(np.int64)

    if cont_mean is None:
        cont_mean = cont.mean(axis=0)
        cont_std = cont.std(axis=0)
        cont_std[cont_std < 1e-8] = 1.0

    cont = (cont - cont_mean) / cont_std

    return {
        'cat': cat, 'cont': cont,
        'lags': np.zeros((len(cat), 0), dtype=np.float32),
        'targets': targets, 'stock': stock,
        'store_ids': store_ids, 'product_ids': product_ids,
        'cont_mean': cont_mean, 'cont_std': cont_std,
    }


train_data = build_split_arrays(df_full, 2, 83)
val_data = build_split_arrays(df_full, 84, 90,
                               cont_mean=train_data['cont_mean'],
                               cont_std=train_data['cont_std'])
test_data = build_split_arrays(df_full, 91, 97,
                                cont_mean=train_data['cont_mean'],
                                cont_std=train_data['cont_std'])

for name, d in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
    print(f'  {name}: {len(d["targets"]):,} samples')

# ---------------------------------------------------------------------------
# 3. Load model
# ---------------------------------------------------------------------------
print('\n3. Caricamento modello...')
n_cont = train_data['cont'].shape[1]
n_lags = 0  # variant A

model = RetailMLP(n_cont, n_lags, EMB_DIMS, CARDINALITIES, HIDDEN_SIZES)
state_path = os.path.join(RESULTS_DIR, 'mlp_variant_A.pt')
model.load_state_dict(torch.load(state_path, map_location='cpu', weights_only=True))
model.to(DEVICE)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f'  Parametri: {n_params:,}')
print(f'  Caricato da: {state_path}')


def predict(model, data, device):
    """Generate predictions."""
    cat_t = torch.from_numpy(data['cat']).to(device)
    cont_t = torch.from_numpy(data['cont']).to(device)
    lags_t = torch.from_numpy(data['lags']).to(device)
    all_preds = []
    chunk = 10000
    with torch.no_grad():
        for s in range(0, len(cat_t), chunk):
            e = min(s + chunk, len(cat_t))
            p = model(cat_t[s:e], cont_t[s:e], lags_t[s:e])
            all_preds.append(p.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


# ---------------------------------------------------------------------------
# 4. Evaluation on all splits
# ---------------------------------------------------------------------------
print('\n4. Valutazione su tutti gli split...')

pooled_results = {}
per_series_dfs = {}

for split_name, data in [('val', val_data), ('test', test_data)]:
    preds = predict(model, data, DEVICE)
    obs = data['targets']
    stock = data['stock']
    sids = data['store_ids']
    pids = data['product_ids']

    # Pooled metrics
    p_flat = preds.ravel()
    o_flat = obs.ravel()
    s_flat = stock.ravel()

    r = {}
    for sub, smask in [('overall', np.ones(len(p_flat), dtype=bool)),
                       ('instock', s_flat == 0),
                       ('stockout', s_flat == 1)]:
        ef = (p_flat - o_flat)[smask]
        of = o_flat[smask]
        sae = np.abs(ef).sum()
        sao = np.abs(of).sum()
        r[f'wape_{sub}'] = sae / sao if sao > 0 else np.nan
        r[f'wpe_{sub}'] = ef.sum() / of.sum() if of.sum() != 0 else np.nan
        r[f'n_{sub}'] = int(smask.sum())
    pooled_results[split_name] = r

    # Per-series metrics
    records = []
    unique_pairs = sorted(set(zip(sids.tolist(), pids.tolist())))
    for (sid, pid) in unique_pairs:
        mask_s = (sids == sid) & (pids == pid)
        m = compute_metrics(preds[mask_s], obs[mask_s], stock[mask_s])
        m['store_id'] = sid
        m['product_id'] = pid
        records.append(m)

    ps = pd.DataFrame(records)
    per_series_dfs[split_name] = ps
    out_path = os.path.join(RESULTS_DIR, f'mlp_{split_name}_per_series.parquet')
    ps.to_parquet(out_path, index=False)
    print(f'  Salvato: {out_path} ({len(ps):,} serie)')

# ---------------------------------------------------------------------------
# 5. Tabella risultati pooled MLP
# ---------------------------------------------------------------------------
print(format_metrics_table(pooled_results, model_name='MLP Baseline (variant A, no history)'))

# ---------------------------------------------------------------------------
# 6. Distribuzione per-serie MLP
# ---------------------------------------------------------------------------
METRIC_COLS = ['wape_overall', 'wape_instock', 'wape_stockout',
               'wpe_overall', 'wpe_instock', 'wpe_stockout']

print('\n' + '=' * 72)
print('  5. DISTRIBUZIONE METRICHE PER-SERIE — MLP')
print('=' * 72)

print(f'\n  {"Split":<8} {"Metric":<16} {"Mean":>8} {"Median":>8} '
      f'{"Std":>8} {"Q5":>8} {"Q95":>8} {"Valid":>7}')
print('  ' + '-' * 80)

for split_name, ps in per_series_dfs.items():
    for col in METRIC_COLS:
        vals = ps[col].dropna()
        if len(vals) == 0:
            continue
        q5, q95 = np.quantile(vals, [0.05, 0.95])
        print(f'  {split_name:<8} {col:<16} {vals.mean():>8.4f} {vals.median():>8.4f} '
              f'{vals.std():>8.4f} {q5:>8.4f} {q95:>8.4f} {len(vals):>7,}')

# ---------------------------------------------------------------------------
# 7. CONFRONTO CON TUTTI I BASELINE
# ---------------------------------------------------------------------------
print('\n' + '=' * 72)
print('  6. CONFRONTO CON TUTTI I BASELINE')
print('=' * 72)

all_baselines = {
    'Naive (direct)': 'naive_direct',
    'MA K=14 (direct)': 'ma_direct_K14',
    'Global Mean': 'global_mean',
    'DoW Mean': 'dow_mean',
    'MLP (var A)': 'mlp',
}

# --- Table: test split ---
print(f'\n  === TEST SPLIT ===')
print(f'\n  {"Model":<24} {"WAPE_pool":>10} {"WAPE_med_ps":>12} '
      f'{"WPE_pool":>10} {"WPE_med_ps":>12}')
print('  ' + '-' * 72)

comparison_data = {}
for label, prefix in all_baselines.items():
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if not os.path.exists(path):
        print(f'  {label:<24} {"N/A":>10} {"N/A":>12} {"N/A":>10} {"N/A":>12}')
        continue
    ps = pd.read_parquet(path)
    med_wape = ps['wape_overall'].median()
    med_wpe = ps['wpe_overall'].median()

    if prefix == 'mlp':
        wp = pooled_results['test']['wape_overall']
        wpe_p = pooled_results['test']['wpe_overall']
    else:
        # Compute pooled from per-series (approximate)
        wp = np.nan
        wpe_p = np.nan

    comparison_data[label] = {
        'wape_med': med_wape, 'wpe_med': med_wpe,
        'wape_pool': wp, 'wpe_pool': wpe_p,
        'ps': ps,
    }

    print(f'  {label:<24} {wp:>10.4f} {med_wape:>12.4f} '
          f'{wpe_p:>10.4f} {med_wpe:>12.4f}')

# --- Table: val split ---
print(f'\n  === VAL SPLIT ===')
print(f'\n  {"Model":<24} {"WAPE_pool":>10} {"WAPE_med_ps":>12} '
      f'{"WPE_pool":>10} {"WPE_med_ps":>12}')
print('  ' + '-' * 72)

for label, prefix in all_baselines.items():
    path = os.path.join(RESULTS_DIR, f'{prefix}_val_per_series.parquet')
    if not os.path.exists(path):
        print(f'  {label:<24} {"N/A":>10} {"N/A":>12} {"N/A":>10} {"N/A":>12}')
        continue
    ps = pd.read_parquet(path)
    med_wape = ps['wape_overall'].median()
    med_wpe = ps['wpe_overall'].median()

    if prefix == 'mlp':
        wp = pooled_results['val']['wape_overall']
        wpe_p = pooled_results['val']['wpe_overall']
    else:
        wp = np.nan
        wpe_p = np.nan

    print(f'  {label:<24} {wp:>10.4f} {med_wape:>12.4f} '
          f'{wpe_p:>10.4f} {med_wpe:>12.4f}')

# ---------------------------------------------------------------------------
# 8. Figure di confronto
# ---------------------------------------------------------------------------
print('\n7. Generazione figure di confronto...')

# Collect all test per-series data
all_models = {}
colors = {}
color_list = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
for i, (label, prefix) in enumerate(all_baselines.items()):
    path = os.path.join(RESULTS_DIR, f'{prefix}_test_per_series.parquet')
    if os.path.exists(path):
        all_models[label] = pd.read_parquet(path)
        colors[label] = color_list[i % len(color_list)]

# --- Fig 30: Boxplot WAPE + WPE test, tutti i modelli ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Confronto Tutti i Baseline — Test Split (per-serie)', fontsize=14)

# WAPE boxplot
box_data_wape = []
box_labels = []
box_colors = []
medians_wape = []
for label, ps in all_models.items():
    vals = ps['wape_overall'].dropna()
    q99 = vals.quantile(0.99)
    box_data_wape.append(vals.clip(upper=q99).values)
    box_labels.append(label)
    box_colors.append(colors[label])
    medians_wape.append(vals.median())

bp = axes[0].boxplot(box_data_wape, tick_labels=box_labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for ml in bp['medians']:
    ml.set_color('red')
    ml.set_linewidth(2)
for k, med in enumerate(medians_wape):
    axes[0].text(k + 1, med + 0.01, f'{med:.3f}', ha='center', va='bottom',
                 fontsize=8, fontweight='bold', color='red')
axes[0].set_title('WAPE overall — Test')
axes[0].set_ylabel('WAPE')
axes[0].tick_params(axis='x', rotation=25)

# WPE boxplot
box_data_wpe = []
medians_wpe = []
for label, ps in all_models.items():
    vals = ps['wpe_overall'].dropna()
    q01, q99 = vals.quantile(0.01), vals.quantile(0.99)
    box_data_wpe.append(vals.clip(lower=q01, upper=q99).values)
    medians_wpe.append(vals.median())

bp = axes[1].boxplot(box_data_wpe, tick_labels=box_labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for ml in bp['medians']:
    ml.set_color('red')
    ml.set_linewidth(2)
axes[1].axhline(0, color='black', linestyle='-', linewidth=0.8)
for k, med in enumerate(medians_wpe):
    offset = 0.005 if med >= 0 else -0.005
    va = 'bottom' if med >= 0 else 'top'
    axes[1].text(k + 1, med + offset, f'{med:.4f}', ha='center', va=va,
                 fontsize=8, fontweight='bold', color='red')
axes[1].set_title('WPE overall — Test')
axes[1].set_ylabel('WPE')
axes[1].tick_params(axis='x', rotation=25)

fig.tight_layout()
out_path = os.path.join(FIG_DIR, 'fig30_compare_all_baselines_test.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Salvata: {out_path}')

# --- Fig 31: Histogrammi WAPE sovrapposti (test) ---
fig, ax = plt.subplots(figsize=(10, 6))
for label, ps in all_models.items():
    vals = ps['wape_overall'].dropna()
    q99 = vals.quantile(0.99)
    ax.hist(vals.clip(upper=q99), bins=80, alpha=0.4, label=f'{label} (med={vals.median():.3f})',
            color=colors[label], edgecolor='none')
ax.set_xlabel('WAPE overall')
ax.set_ylabel('N serie')
ax.set_title('Distribuzione WAPE per-serie — Test Split')
ax.legend(fontsize=9)
fig.tight_layout()
out_path = os.path.join(FIG_DIR, 'fig31_wape_histograms_all_models_test.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Salvata: {out_path}')

# --- Fig 32: Bar chart mediane WAPE + WPE (test) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

labels = list(all_models.keys())
meds_wape = [all_models[l]['wape_overall'].median() for l in labels]
meds_wpe = [all_models[l]['wpe_overall'].median() for l in labels]

x = np.arange(len(labels))
bar_colors = [colors[l] for l in labels]

axes[0].bar(x, meds_wape, color=bar_colors, alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=25, ha='right')
axes[0].set_ylabel('Median WAPE (per-serie)')
axes[0].set_title('WAPE mediana — Test')
for i, v in enumerate(meds_wape):
    axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[1].bar(x, meds_wpe, color=bar_colors, alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=25, ha='right')
axes[1].set_ylabel('Median WPE (per-serie)')
axes[1].set_title('WPE mediana — Test')
axes[1].axhline(0, color='black', linestyle='-', linewidth=0.8)
for i, v in enumerate(meds_wpe):
    offset = 0.002 if v >= 0 else -0.002
    va = 'bottom' if v >= 0 else 'top'
    axes[1].text(i, v + offset, f'{v:.4f}', ha='center', va=va, fontsize=9, fontweight='bold')

fig.suptitle('Confronto Mediane Per-Serie — Test Split', fontsize=14)
fig.tight_layout()
out_path = os.path.join(FIG_DIR, 'fig32_median_comparison_test.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Salvata: {out_path}')

# --- Fig 33: Boxplot val + test per MLP solo ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('MLP (Variant A) — Distribuzione per-serie', fontsize=14)

for j, (split_name, ps) in enumerate(per_series_dfs.items()):
    # Only 2 splits (val, test)
    if j >= 2:
        break

    # WAPE hist in top
    ax = axes[j]
    vals_wape = ps['wape_overall'].dropna()
    vals_wpe_c = vals_wape.clip(upper=vals_wape.quantile(0.99))
    ax.hist(vals_wpe_c, bins=80, color='steelblue', alpha=0.7, edgecolor='none')
    ax.axvline(vals_wape.median(), color='red', linestyle='--', linewidth=1.5,
               label=f'median={vals_wape.median():.3f}')
    ax.set_title(f'WAPE overall — {split_name}')
    ax.set_xlabel('WAPE')
    ax.set_ylabel('N serie')
    ax.legend(fontsize=9)

fig.tight_layout()
out_path = os.path.join(FIG_DIR, 'fig33_mlp_distributions.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'  Salvata: {out_path}')

print('\n' + '=' * 72)
print('  DONE')
print('=' * 72)
