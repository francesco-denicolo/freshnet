"""
15_generate_mnar_masks.py — Generate MNAR masks for demand recovery evaluation
===============================================================================
Genera DUE set di maschere MNAR predefinite per la Traccia A.
Le stesse maschere vengono usate da TUTTI i modelli per un confronto equo.

Pattern MNAR: p(mask|hour=h) proportional to distribuzione empirica stockout per ora h
              (piu' alta ore pomeridiane/serali, piu' bassa ore notturne)
Missing rate: 30% delle ore in-stock

Set 1 — VAL:  seed=42,  giorni 84-90 (per selezione modello di imputation)
Set 2 — TEST: seed=123, giorni 1-90  (per valutazione finale, Traccia A)

Output:
  data/mnar_masks_val.parquet   (giorni 84-90, seed=42)
  data/mnar_masks_test.parquet  (giorni 1-90,  seed=123)

Eseguire con: freshnet/bin/python notebooks/15_generate_mnar_masks.py
"""

import os
import numpy as np
import pandas as pd
import time
import functools

print = functools.partial(print, flush=True)

# ---- Config ----
MISSING_RATE = 0.30

# Two mask sets as per CLAUDE_SEQUENTIAL-2.md
MASK_SETS = [
    {'name': 'val',  'seed': 42,  'day_start': 84, 'day_end': 90,
     'file': 'mnar_masks_val.parquet'},
    {'name': 'test', 'seed': 123, 'day_start': 1,  'day_end': 90,
     'file': 'mnar_masks_test.parquet'},
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

print("=" * 70)
print("FASE 0 — Step 0.1: Generazione maschere MNAR predefinite")
print("=" * 70)
print(f"  Missing rate: {MISSING_RATE*100:.0f}%")
for ms in MASK_SETS:
    print(f"  Set {ms['name']:4s}: seed={ms['seed']}, giorni {ms['day_start']}-{ms['day_end']}, "
          f"file={ms['file']}")

# =========================================================================
# 1. Caricamento dati (tutti i 90 giorni del train HF)
# =========================================================================
t0 = time.time()
print("\n1. Caricamento dati (tutti i 90 giorni)...")
df = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df['dt_parsed'] = pd.to_datetime(df['dt'])

# Day number (1-indexed)
min_date = df['dt_parsed'].min()
df['day_num'] = (df['dt_parsed'] - min_date).dt.days + 1
N_total = len(df)
n_days = df['day_num'].nunique()
print(f"  Righe totali: {N_total:,}")
print(f"  Serie uniche: {df.groupby(['store_id', 'product_id']).ngroups:,}")
print(f"  Giorni unici: {n_days} (1-{df['day_num'].max()})")

# Parse hourly arrays (all 90 days)
print("  Parsing arrays orari...")
sales_all = np.array(df['hours_sale'].tolist(), dtype=np.float32)      # (N, 24)
stock_all = np.array(df['hours_stock_status'].tolist(), dtype=np.float32)  # (N, 24)
day_nums = df['day_num'].values
store_ids = df['store_id'].values
product_ids = df['product_id'].values
dt_vals = df['dt'].values
print(f"  Shape sales: {sales_all.shape}, stock: {stock_all.shape}")
print(f"  Tempo loading: {time.time()-t0:.1f}s")

# =========================================================================
# 2. Distribuzione empirica stockout per ora (calcolata su TUTTI i 90 giorni)
# =========================================================================
print("\n2. Distribuzione empirica stockout per ora (tutti i 90 giorni)...")
stockout_rate_per_hour = stock_all.mean(axis=0)  # (24,)

print(f"  {'Ora':>4} {'Stockout Rate':>14} {'In-stock %':>11}")
print(f"  {'----':>4} {'-'*14:>14} {'-'*11:>11}")
for h in range(24):
    print(f"  {h:4d} {stockout_rate_per_hour[h]:14.4f} {(1-stockout_rate_per_hour[h])*100:10.1f}%")

overall_stockout = stock_all.mean()
print(f"\n  Stockout complessivo: {overall_stockout*100:.2f}%")


# =========================================================================
# 3. Funzione di generazione maschere
# =========================================================================
def generate_mnar_masks(sales, stock, day_nums_arr, store_ids_arr,
                        product_ids_arr, dt_arr, stockout_rate,
                        day_start, day_end, seed, missing_rate):
    """Generate MNAR masks for a specific day range and seed."""

    # Filter to day range
    day_mask = (day_nums_arr >= day_start) & (day_nums_arr <= day_end)
    sales_sub = sales[day_mask]
    stock_sub = stock[day_mask]
    N_sub = day_mask.sum()

    # In-stock mask
    instock_mask = (stock_sub == 0)
    total_instock = instock_mask.sum()

    # MNAR probability calibration
    weights_broadcast = np.broadcast_to(stockout_rate, (N_sub, 24))
    avg_weight_instock = weights_broadcast[instock_mask].mean()
    scale = missing_rate / avg_weight_instock

    mask_probs = np.clip(scale * stockout_rate, 0.0, 0.95)

    # Vectorized Bernoulli sampling
    rng = np.random.default_rng(seed=seed)
    rand_all = rng.random(stock_sub.shape).astype(np.float32)
    mask_probs_broadcast = np.broadcast_to(mask_probs.astype(np.float32), (N_sub, 24))
    mask_matrix = instock_mask & (rand_all < mask_probs_broadcast)

    n_masked = mask_matrix.sum()
    actual_rate = n_masked / total_instock if total_instock > 0 else 0

    # Build dataframe
    row_idx, hour_idx = np.where(mask_matrix)
    day_mask_indices = np.where(day_mask)[0]

    mask_df = pd.DataFrame({
        'store_id': store_ids_arr[day_mask_indices[row_idx]],
        'product_id': product_ids_arr[day_mask_indices[row_idx]],
        'dt': dt_arr[day_mask_indices[row_idx]],
        'hour': hour_idx.astype(np.int32),
        'is_masked': np.ones(len(row_idx), dtype=bool),
        'ground_truth': sales_sub[row_idx, hour_idx],
    })

    return mask_df, {
        'n_rows_period': int(N_sub),
        'total_instock': int(total_instock),
        'n_masked': int(n_masked),
        'actual_rate': float(actual_rate),
        'gt_mean': float(mask_df['ground_truth'].mean()),
        'gt_gt0_pct': float((mask_df['ground_truth'] > 0).mean() * 100),
    }


# =========================================================================
# 4. Generazione dei due set di maschere
# =========================================================================
for ms in MASK_SETS:
    print(f"\n{'='*70}")
    print(f"  Generazione maschere {ms['name'].upper()}: "
          f"seed={ms['seed']}, giorni {ms['day_start']}-{ms['day_end']}")
    print(f"{'='*70}")

    t1 = time.time()
    mask_df, stats = generate_mnar_masks(
        sales_all, stock_all, day_nums, store_ids, product_ids, dt_vals,
        stockout_rate_per_hour,
        ms['day_start'], ms['day_end'], ms['seed'], MISSING_RATE)

    print(f"  Righe periodo:   {stats['n_rows_period']:,} "
          f"({ms['day_end']-ms['day_start']+1} giorni)")
    print(f"  Slot in-stock:   {stats['total_instock']:,}")
    print(f"  Slot mascherati: {stats['n_masked']:,}")
    print(f"  Rate effettivo:  {stats['actual_rate']*100:.2f}% (target: {MISSING_RATE*100:.0f}%)")
    print(f"  Serie con mask:  {mask_df.groupby(['store_id', 'product_id']).ngroups:,}")
    print(f"  GT mean:         {stats['gt_mean']:.4f}")
    print(f"  GT>0:            {stats['gt_gt0_pct']:.1f}%")

    # Diagnostica per ora
    print(f"\n  Rate mascheramento per ora:")
    print(f"  {'Ora':>4} {'Mascherati':>12} {'Rate':>8}")
    print(f"  {'----':>4} {'----------':>12} {'------':>8}")
    day_mask_full = (day_nums >= ms['day_start']) & (day_nums <= ms['day_end'])
    stock_sub = stock_all[day_mask_full]
    instock_sub = (stock_sub == 0)
    for h in range(24):
        n_instock_h = int(instock_sub[:, h].sum())
        n_masked_h = int((mask_df['hour'] == h).sum())
        rate_h = n_masked_h / n_instock_h if n_instock_h > 0 else 0
        print(f"  {h:4d} {n_masked_h:12,} {rate_h:8.4f}")

    # Save
    out_path = os.path.join(DATA_DIR, ms['file'])
    mask_df.to_parquet(out_path, index=False, engine='pyarrow')
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n  Salvato: {out_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Tempo: {time.time()-t1:.1f}s")


# =========================================================================
# Riepilogo
# =========================================================================
print(f"\n{'='*70}")
print("RIEPILOGO MASCHERE MNAR")
print(f"{'='*70}")
print(f"  Missing rate target: {MISSING_RATE*100:.0f}%")
print(f"  Pattern: MNAR (proporzionale a distribuzione empirica stockout)")
print(f"  Distribuzione calcolata su: tutti i 90 giorni del train HF")
print()
for ms in MASK_SETS:
    out_path = os.path.join(DATA_DIR, ms['file'])
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    n_rows = pd.read_parquet(out_path, columns=['store_id']).shape[0]
    print(f"  {ms['name'].upper():4s}: seed={ms['seed']}, giorni {ms['day_start']}-{ms['day_end']}, "
          f"{n_rows:,} slot, {size_mb:.1f} MB")

print(f"\n  Tempo totale: {time.time()-t0:.1f}s")
print("=" * 70)
