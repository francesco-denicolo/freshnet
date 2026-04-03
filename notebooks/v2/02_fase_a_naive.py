"""
02_fase_a_naive.py — Fase A: Naive Forecasting Baselines su dati sporchi
========================================================================
Piano: CLAUDE_SEQUENTIAL-2.md, Fase A, punto (1)

Quattro modelli naive su dati sporchi (S_obs con stockout):

1. Global Mean:   profilo = media S_obs su tutti i giorni
2. DoW Mean:      profilo = media S_obs per giorno della settimana (7 profili)
3. Naive Direct:  profilo = S_obs dell'ultimo giorno prima dell'orizzonte
4. MA Direct:     profilo = media ultimi K giorni (K selezionato su val)

Workflow:
  - Val:  profili calcolati su gg 1-83, predizione su gg 84-90
  - Test: profili calcolati su gg 1-90, predizione su eval HF (gg 91-97)
  - MA: K selezionato su val (pooled WAPE), poi test con K*

Metriche: WAPE e WPE (overall, instock, stockout), pooled e mediana per-serie.

Output:
  notebooks/v2/results/<model>_<split>_per_series.parquet

Eseguire con: freshnet/bin/python notebooks/v2/02_fase_a_naive.py
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import functools

print = functools.partial(print, flush=True)

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

from src.evaluation.metrics import compute_metrics, format_metrics_table


def _accumulate_pooled(acc_dict, preds, obs, stock):
    """Accumulate pooled WAPE/WPE statistics for overall/instock/stockout."""
    p_flat = preds.ravel()
    o_flat = obs.ravel()
    s_flat = stock.ravel()
    err = p_flat - o_flat

    for sub, smask in [('overall', np.ones(len(p_flat), dtype=bool)),
                       ('instock', s_flat == 0),
                       ('stockout', s_flat == 1)]:
        acc = acc_dict[sub]
        ef = err[smask]
        of = o_flat[smask]
        acc['sae'] += np.abs(ef).sum()
        acc['sao'] += np.abs(of).sum()
        acc['se'] += ef.sum()
        acc['so'] += of.sum()
        acc['n'] += int(smask.sum())


# ---- Config ----
K_CANDIDATES = [3, 5, 7, 10, 14, 21, 28, 42, 56, 83]

print("=" * 72)
print("  FASE A — NAIVE FORECASTING BASELINES (dati sporchi)")
print("=" * 72)

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
df_full['dow'] = df_full['dt_parsed'].dt.dayofweek  # 0=Mon

n_series = df_full.groupby(['store_id', 'product_id']).ngroups
n_days = len(all_dates)
print(f"  Train: {len(df_train):,} righe, giorni 1-90")
print(f"  Eval:  {len(df_eval):,} righe, giorni 91-97")
print(f"  Full:  {len(df_full):,} righe, {n_days} giorni, {n_series:,} serie")
print(f"  Tempo loading: {time.time()-t0:.1f}s")

del df_train, df_eval

# =========================================================================
# 2. Loop su serie: calcolo predizioni per tutti i 4 modelli
# =========================================================================
print("\n2. Calcolo predizioni per tutti i modelli naive...")
print("   Val:  profili da gg 1-83, predizione gg 84-90")
print("   Test: profili da gg 1-90, predizione gg 91-97")
print(f"   MA K candidati: {K_CANDIDATES}")

# Accumulatori pooled per ogni modello e split
MODELS = ['global_mean', 'dow_mean', 'naive_direct', 'ma_direct']
SPLITS = ['val', 'test']

pooled = {
    model: {
        split: {
            sub: {'sae': 0., 'sao': 0., 'se': 0., 'so': 0., 'n': 0}
            for sub in ['overall', 'instock', 'stockout']
        }
        for split in SPLITS
    }
    for model in MODELS
}

per_series = {model: {split: [] for split in SPLITS} for model in MODELS}

# MA: accumulatori per K selection su val
ma_pooled_val = {
    K: {'sae': 0., 'sao': 0., 'se': 0., 'so': 0.}
    for K in K_CANDIDATES
}
ma_ps_wapes_val = {K: [] for K in K_CANDIDATES}

groups = df_full.groupby(['store_id', 'product_id'], sort=False)
n_groups = len(groups)

t1 = time.time()
for i, ((sid, pid), grp) in enumerate(groups):
    if (i + 1) % 10000 == 0:
        elapsed = time.time() - t1
        print(f"    ... {i+1:,}/{n_groups:,} serie ({elapsed:.0f}s)")

    grp_s = grp.sort_values('day_num')
    sales = np.array(grp_s['hours_sale'].tolist(), dtype=np.float64)  # (N, 24)
    stock = np.array(grp_s['hours_stock_status'].tolist())
    days = grp_s['day_num'].values
    dows = grp_s['dow'].values

    # ---- Precompute profiles ----
    train_mask = days <= 83
    trainval_mask = days <= 90

    if not train_mask.any():
        continue

    # Global Mean profiles
    gm_profile_tv = sales[train_mask].mean(axis=0)      # for val
    gm_profile_test = sales[trainval_mask].mean(axis=0)  # for test

    # DoW Mean profiles
    dow_profiles_tv = {}
    for d in range(7):
        dm = train_mask & (dows == d)
        dow_profiles_tv[d] = sales[dm].mean(axis=0) if dm.any() else gm_profile_tv

    dow_profiles_test = {}
    gm_tv_full = sales[trainval_mask].mean(axis=0)
    for d in range(7):
        dm = trainval_mask & (dows == d)
        dow_profiles_test[d] = sales[dm].mean(axis=0) if dm.any() else gm_tv_full

    # Naive Direct profiles
    anchor_83_idx = np.where(days == 83)[0]
    anchor_90_idx = np.where(days == 90)[0]
    nd_profile_val = sales[anchor_83_idx[0]] if len(anchor_83_idx) > 0 else None
    nd_profile_test = sales[anchor_90_idx[0]] if len(anchor_90_idx) > 0 else None

    # MA: K selection on val (prima del loop split)
    val_mask = (days >= 84) & (days <= 90)
    if val_mask.any():
        val_obs = sales[val_mask]
        n_val = val_obs.shape[0]
        for K in K_CANDIDATES:
            start_day = max(1, 83 - K + 1)
            hist_m = (days >= start_day) & (days <= 83)
            if not hist_m.any():
                ma_ps_wapes_val[K].append(np.nan)
                continue
            profile = sales[hist_m].mean(axis=0)
            preds = np.tile(profile, (n_val, 1))
            err = preds.ravel() - val_obs.ravel()
            ma_pooled_val[K]['sae'] += np.abs(err).sum()
            ma_pooled_val[K]['sao'] += np.abs(val_obs.ravel()).sum()
            ma_pooled_val[K]['se'] += err.sum()
            ma_pooled_val[K]['so'] += val_obs.ravel().sum()
            s_abs_obs = np.abs(val_obs).sum()
            if s_abs_obs > 0:
                ma_ps_wapes_val[K].append(np.abs(preds - val_obs).sum() / s_abs_obs)
            else:
                ma_ps_wapes_val[K].append(np.nan)
    else:
        for K in K_CANDIDATES:
            ma_ps_wapes_val[K].append(np.nan)

    # ---- Evaluate each model on val and test ----
    for split_name, d_min, d_max in [('val', 84, 90), ('test', 91, 97)]:
        mask = (days >= d_min) & (days <= d_max)
        if not mask.any():
            continue

        obs = sales[mask]       # (n_split, 24)
        stk = stock[mask]       # (n_split, 24)
        split_dows = dows[mask]
        n_split = mask.sum()

        # --- Global Mean ---
        gm_prof = gm_profile_test if split_name == 'test' else gm_profile_tv
        gm_pred = np.tile(gm_prof, (n_split, 1))
        _accumulate_pooled(pooled['global_mean'][split_name], gm_pred, obs, stk)
        m = compute_metrics(gm_pred, obs, stk)
        m['store_id'] = sid
        m['product_id'] = pid
        per_series['global_mean'][split_name].append(m)

        # --- DoW Mean ---
        dow_profs = dow_profiles_test if split_name == 'test' else dow_profiles_tv
        dow_pred = np.array([dow_profs[d] for d in split_dows])  # (n_split, 24)
        _accumulate_pooled(pooled['dow_mean'][split_name], dow_pred, obs, stk)
        m = compute_metrics(dow_pred, obs, stk)
        m['store_id'] = sid
        m['product_id'] = pid
        per_series['dow_mean'][split_name].append(m)

        # --- Naive Direct ---
        nd_prof = nd_profile_test if split_name == 'test' else nd_profile_val
        if nd_prof is not None:
            nd_pred = np.tile(nd_prof, (n_split, 1))
            _accumulate_pooled(pooled['naive_direct'][split_name], nd_pred, obs, stk)
            m = compute_metrics(nd_pred, obs, stk)
            m['store_id'] = sid
            m['product_id'] = pid
            per_series['naive_direct'][split_name].append(m)

        # MA Direct: will be filled after K selection

print(f"\n  Loop serie completato in {time.time()-t1:.0f}s")


# =========================================================================
# 3. MA Direct: K selection su val
# =========================================================================
print("\n3. MA Direct — K selection su val...")
print(f"\n  {'K':>4} {'WAPE_pooled':>14} {'WAPE_med_ps':>14} {'WPE_pooled':>12}")
print("  " + "-" * 48)

k_selection = {}
for K in K_CANDIDATES:
    pa = ma_pooled_val[K]
    wape_p = pa['sae'] / pa['sao'] if pa['sao'] > 0 else np.nan
    wpe_p = pa['se'] / pa['so'] if pa['so'] > 0 else np.nan
    wape_m = np.nanmedian(ma_ps_wapes_val[K])
    k_selection[K] = {'wape_pooled': wape_p, 'wape_median': wape_m, 'wpe_pooled': wpe_p}
    print(f"  {K:>4} {wape_p:>14.6f} {wape_m:>14.6f} {wpe_p:>12.6f}")

best_K_pooled = min(K_CANDIDATES, key=lambda k: k_selection[k]['wape_pooled'])
best_K_median = min(K_CANDIDATES, key=lambda k: k_selection[k]['wape_median'])

print(f"\n  Best K (WAPE pooled):  K={best_K_pooled} "
      f"(WAPE={k_selection[best_K_pooled]['wape_pooled']:.6f})")
print(f"  Best K (WAPE median):  K={best_K_median} "
      f"(WAPE={k_selection[best_K_median]['wape_median']:.6f})")

if best_K_pooled == best_K_median:
    K_star = best_K_pooled
    print(f"\n  Criteri concordano: K*={K_star}")
else:
    K_star = best_K_pooled
    print(f"\n  Criteri discordanti. Uso K*={K_star} (pooled).")

# =========================================================================
# 4. MA Direct: test evaluation con K*
# =========================================================================
print(f"\n4. MA Direct — Evaluation con K*={K_star}...")
print(f"   Val:  profilo = media ultimi {K_star} gg prima di gg 84")
print(f"   Test: profilo = media ultimi {K_star} gg prima di gg 91")

# MA val: use the K* accumulators from section 2
pooled['ma_direct']['val'] = {
    'overall':  {'sae': 0., 'sao': 0., 'se': 0., 'so': 0., 'n': 0},
    'instock':  {'sae': 0., 'sao': 0., 'se': 0., 'so': 0., 'n': 0},
    'stockout': {'sae': 0., 'sao': 0., 'se': 0., 'so': 0., 'n': 0},
}

# Need a second pass for MA Direct (val + test with K*)
t2 = time.time()
groups2 = df_full.groupby(['store_id', 'product_id'], sort=False)
for i, ((sid, pid), grp) in enumerate(groups2):
    if (i + 1) % 10000 == 0:
        print(f"    ... {i+1:,}/{n_groups:,} serie")

    grp_s = grp.sort_values('day_num')
    sales = np.array(grp_s['hours_sale'].tolist(), dtype=np.float64)
    stock = np.array(grp_s['hours_stock_status'].tolist())
    days = grp_s['day_num'].values

    for split_name, anchor_day, d_min, d_max in [
        ('val', 83, 84, 90),
        ('test', 90, 91, 97),
    ]:
        target_mask = (days >= d_min) & (days <= d_max)
        if not target_mask.any():
            continue

        start_day = max(1, anchor_day - K_star + 1)
        hist_mask = (days >= start_day) & (days <= anchor_day)
        if not hist_mask.any():
            continue

        profile = sales[hist_mask].mean(axis=0)
        obs = sales[target_mask]
        stk = stock[target_mask]
        n_target = target_mask.sum()
        preds = np.tile(profile, (n_target, 1))

        _accumulate_pooled(pooled['ma_direct'][split_name], preds, obs, stk)
        m = compute_metrics(preds, obs, stk)
        m['store_id'] = sid
        m['product_id'] = pid
        per_series['ma_direct'][split_name].append(m)

print(f"  MA Direct loop: {time.time()-t2:.0f}s")


# =========================================================================
# 5. Salvataggio risultati per-serie
# =========================================================================
print("\n5. Salvataggio risultati per-serie...")

per_series_dfs = {}
for model in MODELS:
    per_series_dfs[model] = {}
    for split in SPLITS:
        if per_series[model][split]:
            df_ps = pd.DataFrame(per_series[model][split])
            per_series_dfs[model][split] = df_ps
            out_path = os.path.join(RESULTS_DIR, f'{model}_{split}_per_series.parquet')
            df_ps.to_parquet(out_path, index=False)
            print(f"  Salvato: {model}_{split}_per_series.parquet ({len(df_ps):,} serie)")


# =========================================================================
# 6. Tabelle risultati pooled
# =========================================================================
print("\n" + "=" * 72)
print("  6. RISULTATI POOLED")
print("=" * 72)

pooled_results = {}
for model in MODELS:
    pooled_results[model] = {}
    for split in SPLITS:
        r = {}
        for sub in ['overall', 'instock', 'stockout']:
            acc = pooled[model][split][sub]
            r[f'wape_{sub}'] = acc['sae'] / acc['sao'] if acc['sao'] > 0 else np.nan
            r[f'wpe_{sub}'] = acc['se'] / acc['so'] if acc['so'] > 0 else np.nan
            r[f'n_{sub}'] = acc['n']
        pooled_results[model][split] = r

for model in MODELS:
    model_label = model.replace('_', ' ').title()
    if model == 'ma_direct':
        model_label += f' (K={K_star})'
    print(format_metrics_table(pooled_results[model], model_name=model_label))


# =========================================================================
# 7. Distribuzione per-serie (mediana)
# =========================================================================
print("\n" + "=" * 72)
print("  7. DISTRIBUZIONE PER-SERIE")
print("=" * 72)

METRIC_COLS = ['wape_overall', 'wape_instock', 'wpe_overall', 'wpe_instock']

for split in SPLITS:
    print(f"\n  --- {split.upper()} ---")
    print(f"  {'Modello':<20} {'WAPE_all_med':>14} {'WAPE_in_med':>14} "
          f"{'WPE_all_med':>14} {'WPE_in_med':>14} {'N_serie':>8}")
    print("  " + "-" * 88)

    for model in MODELS:
        if model not in per_series_dfs or split not in per_series_dfs[model]:
            continue
        ps = per_series_dfs[model][split]
        label = model.replace('_', ' ').title()
        if model == 'ma_direct':
            label += f' (K={K_star})'

        vals = {}
        for col in METRIC_COLS:
            vals[col] = ps[col].median() if col in ps.columns else np.nan

        print(f"  {label:<20} {vals['wape_overall']:>14.4f} {vals['wape_instock']:>14.4f} "
              f"{vals['wpe_overall']:>14.4f} {vals['wpe_instock']:>14.4f} {len(ps):>8,}")


# =========================================================================
# 8. Tabella di confronto finale
# =========================================================================
print("\n" + "=" * 72)
print("  8. CONFRONTO FINALE (TEST)")
print("=" * 72)

print(f"\n  {'Modello':<20} {'WAPE_in_pool':>14} {'WAPE_in_med':>14} "
      f"{'WPE_in_pool':>14} {'WPE_in_med':>14} {'WAPE_all_med':>14}")
print("  " + "-" * 94)

for model in MODELS:
    label = model.replace('_', ' ').title()
    if model == 'ma_direct':
        label += f' (K={K_star})'

    pr = pooled_results[model].get('test', {})
    ps = per_series_dfs.get(model, {}).get('test', None)

    wape_in_pool = pr.get('wape_instock', np.nan)
    wpe_in_pool = pr.get('wpe_instock', np.nan)
    wape_in_med = ps['wape_instock'].median() if ps is not None else np.nan
    wpe_in_med = ps['wpe_instock'].median() if ps is not None else np.nan
    wape_all_med = ps['wape_overall'].median() if ps is not None else np.nan

    print(f"  {label:<20} {wape_in_pool:>14.4f} {wape_in_med:>14.4f} "
          f"{wpe_in_pool:>14.4f} {wpe_in_med:>14.4f} {wape_all_med:>14.4f}")

print(f"\n  Tempo totale: {time.time()-t0:.0f}s")
print("=" * 72)
