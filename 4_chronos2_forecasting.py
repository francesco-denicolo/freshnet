#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNAE 98 - STEP 4: CHRONOS-2 TIME SERIES FORECASTING (BATCH PROCESSING)
=======================================================================

Performs sliding window forecasting using Chronos-2 model on the 6 cluster hourly time series.
Uses BATCH PROCESSING to forecast all 6 clusters simultaneously for each window.

Input:
------
- cluster_<ID>_hourly_timeseries.parquet: Hourly time series for each cluster

Output:
-------
- forecasts_chronos/: Directory containing forecast results
  - cluster_<ID>_forecasts.parquet: Forecasts for each cluster
  - forecasts_summary.csv: Summary statistics

Configuration:
--------------
- Model: Chronos-2 (amazon-chronos-t5-small via AutoGluon)
- Context length: 512 hours (~21 days)
- Warmup period: 12 weeks (2,016 hours) - only last 512 will be used
- Forecast horizon: 168 hours (1 week)
- Stride: 168 hours (1 week)
- Batch processing: All 6 clusters forecasted simultaneously per window
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from chronos import BaseChronosPipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION
# ========================================

BASE_DIR = Path(__file__).parent.parent.parent
TIMESERIES_DIR = BASE_DIR / 'forecasting_pipeline' / 'output' / 'clusters_by_cnae' / 'hourly_timeseries'
OUTPUT_DIR = Path(__file__).parent / 'output'
FORECASTS_DIR = OUTPUT_DIR / 'forecasts_chronos'
VIZ_DIR = OUTPUT_DIR / 'visualizations'

# Create output directories
FORECASTS_DIR.mkdir(exist_ok=True, parents=True)
VIZ_DIR.mkdir(exist_ok=True, parents=True)

# Final 6 clusters
FINAL_CLUSTERS = ['0.0', '0.1.0', '0.1.1', '0.1.3', '0.1.5', '2.1']

# Forecasting parameters
WARMUP_WEEKS = 12                    # 12 weeks warmup (but only last 512 hours used by Chronos-2)
WARMUP_HOURS = WARMUP_WEEKS * 7 * 24  # 2,016 hours
FORECAST_HORIZON = 168               # 1 week = 168 hours
STRIDE = 168                         # Move forward by 1 week each iteration
CONTEXT_LENGTH = 512                 # Chronos-2 default context length

# Quantile levels for prediction intervals
QUANTILE_LEVELS = [0.025, 0.5, 0.975]  # 95% prediction interval + median

# Model configuration
MODEL_NAME = "amazon/chronos-t5-small"  # Chronos-2 model
DEVICE = "cpu"  # or "cuda" if GPU available

print("="*80)
print(" CNAE 98 - STEP 4: CHRONOS-2 FORECASTING ".center(80))
print("="*80)
print(f"\n Configuration:")
print(f"   Model: {MODEL_NAME}")
print(f"   Batch processing: ALL 6 CLUSTERS per window")
print(f"   Warmup period: {WARMUP_WEEKS} weeks ({WARMUP_HOURS:,} hours)")
print(f"   Chronos-2 context length: {CONTEXT_LENGTH} hours")
print(f"   Forecast horizon: {FORECAST_HORIZON} hours (1 week)")
print(f"   Stride: {STRIDE} hours (1 week)")
print(f"   Quantiles: {QUANTILE_LEVELS}")

# ========================================
# STEP 1: LOAD CHRONOS-2 MODEL
# ========================================

print(f"\n Loading Chronos-2 model from AutoGluon S3...")
try:
    pipeline = BaseChronosPipeline.from_pretrained(
        "s3://autogluon/chronos-2/",  # AutoGluon version with predict_df support
        device_map=DEVICE
    )
    print(" Model loaded successfully")
except Exception as e:
    print(f" Error loading model: {e}")
    print("\n To install Chronos-2, run:")
    print("   pip install chronos-forecasting")
    exit(1)

# ========================================
# STEP 2: LOAD TIME SERIES DATA
# ========================================

print(f"\n Loading hourly time series for {len(FINAL_CLUSTERS)} clusters...")

cluster_data = {}
for cluster_id in FINAL_CLUSTERS:
    filename = f'cluster_{cluster_id.replace(".", "_")}_hourly_timeseries.parquet'
    filepath = TIMESERIES_DIR / filename

    if not filepath.exists():
        print(f"   Warning: File not found: {filename}")
        continue

    df = pd.read_parquet(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    cluster_data[cluster_id] = df
    print(f"   {cluster_id}: {len(df):,} hours ({df['timestamp'].min()} to {df['timestamp'].max()})")

print(f"\n Loaded {len(cluster_data)} clusters")

# ========================================
# STEP 3: CALCULATE FORECAST WINDOWS
# ========================================

print(f"\n Calculating forecast windows...")

# Find common temporal range across all clusters
all_start_times = [df['timestamp'].min() for df in cluster_data.values()]
all_end_times = [df['timestamp'].max() for df in cluster_data.values()]
common_start = max(all_start_times)
common_end = min(all_end_times)

print(f"   Common temporal range: {common_start} to {common_end}")

# EXCLUDE W41 (problematic window with data quality issues on June 5-7, 2022)
CUTOFF_DATE = pd.Timestamp('2022-06-04 23:00:00')
if common_end > CUTOFF_DATE:
    print(f"\n   Excluding W41 (data quality issues)")
    print(f"   Original end: {common_end}")
    print(f"   New end: {CUTOFF_DATE}")
    common_end = CUTOFF_DATE

# Filter all clusters to common range
for cluster_id in list(cluster_data.keys()):
    df = cluster_data[cluster_id]
    df_filtered = df[(df['timestamp'] >= common_start) & (df['timestamp'] <= common_end)].copy()
    cluster_data[cluster_id] = df_filtered.reset_index(drop=True)
    print(f"   {cluster_id}: {len(df_filtered):,} hours in common range")

# Calculate number of windows
n_hours = len(next(iter(cluster_data.values())))
min_hours_needed = WARMUP_HOURS + FORECAST_HORIZON

if n_hours < min_hours_needed:
    print(f"   Insufficient data: {n_hours} hours < {min_hours_needed} required")
    exit(1)

available_for_forecasts = n_hours - WARMUP_HOURS
n_windows = (available_for_forecasts - FORECAST_HORIZON) // STRIDE + 1

print(f"\n   Total hours (common): {n_hours:,}")
print(f"   Warmup hours: {WARMUP_HOURS:,}")
print(f"   Available for forecasts: {available_for_forecasts:,} hours")
print(f"   Number of forecast windows: {n_windows}")

# ========================================
# STEP 4: BATCH FORECASTING
# ========================================

print(f"\n{'='*80}")
print(" BATCH FORECASTING: ALL 6 CLUSTERS PER WINDOW ".center(80))
print('='*80)

all_forecasts = {cluster_id: [] for cluster_id in FINAL_CLUSTERS}

for window_idx in range(n_windows):
    # Calculate window boundaries
    forecast_start_idx = WARMUP_HOURS + (window_idx * STRIDE)
    forecast_end_idx = forecast_start_idx + FORECAST_HORIZON

    if forecast_end_idx > n_hours:
        print(f"   Window exceeds data range, stopping.")
        break

    # Extract warmup data for all clusters
    warmup_end_idx = forecast_start_idx
    warmup_start_idx = warmup_end_idx - WARMUP_HOURS

    if warmup_start_idx < 0:
        warmup_start_idx = 0

    # Print progress
    if (window_idx + 1) % 5 == 0 or window_idx == 0 or (window_idx + 1) == n_windows:
        first_cluster = next(iter(cluster_data.values()))
        forecast_start_time = first_cluster.iloc[forecast_start_idx]['timestamp']
        print(f"\n   Window {window_idx + 1}/{n_windows}: {forecast_start_time.date()}")

    # Prepare batch DataFrame for predict_df
    batch_data = []
    for cluster_id, df_cluster in cluster_data.items():
        warmup_data = df_cluster.iloc[warmup_start_idx:warmup_end_idx].copy()

        # Determine target column
        if 'total_consumption' in df_cluster.columns:
            target_col = 'total_consumption'
        elif 'hourly_total_consumption' in df_cluster.columns:
            target_col = 'hourly_total_consumption'
        else:
            target_col = df_cluster.columns[-1]

        for idx, row in warmup_data.iterrows():
            batch_data.append({
                'item_id': cluster_id,
                'timestamp': row['timestamp'],
                'target': row[target_col]
            })

    df_batch = pd.DataFrame(batch_data)

    # Get actual values for evaluation
    actual_values = {}
    for cluster_id, df_cluster in cluster_data.items():
        actual_data = df_cluster.iloc[forecast_start_idx:forecast_end_idx].copy()
        actual_values[cluster_id] = actual_data

    try:
        # Use predict_df for batch processing
        pred_df = pipeline.predict_df(
            df_batch,
            prediction_length=FORECAST_HORIZON,
            quantile_levels=QUANTILE_LEVELS
        )

        # Process predictions for each cluster
        for cluster_id in FINAL_CLUSTERS:
            # Filter predictions for this cluster
            cluster_preds = pred_df[pred_df['item_id'] == cluster_id].copy()

            if len(cluster_preds) == 0:
                continue

            # Get actual values
            actual_data = actual_values[cluster_id]

            # Determine target column
            if 'total_consumption' in actual_data.columns:
                target_col = 'total_consumption'
            elif 'hourly_total_consumption' in actual_data.columns:
                target_col = 'hourly_total_consumption'
            else:
                target_col = actual_data.columns[-1]

            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'cluster_id': cluster_id,
                'window_idx': window_idx,
                'forecast_hour': range(len(cluster_preds)),
                'timestamp': actual_data['timestamp'].values,
                'actual': actual_data[target_col].values,
                'forecast_median': cluster_preds['0.5'].values,
                'forecast_lower': cluster_preds['0.025'].values,
                'forecast_upper': cluster_preds['0.975'].values,
            })

            # Calculate errors
            forecast_df['error'] = forecast_df['actual'] - forecast_df['forecast_median']
            forecast_df['abs_error'] = np.abs(forecast_df['error'])
            forecast_df['squared_error'] = forecast_df['error'] ** 2
            forecast_df['pct_error'] = np.abs(forecast_df['error'] / forecast_df['actual']) * 100

            all_forecasts[cluster_id].append(forecast_df)

        # Print MAE summary
        if (window_idx + 1) % 5 == 0 or window_idx == 0 or (window_idx + 1) == n_windows:
            mae_str = ", ".join([
                f"{cid}:{np.mean([f['abs_error'].mean() for f in all_forecasts[cid][-1:]]):.0f}"
                for cid in FINAL_CLUSTERS if len(all_forecasts[cid]) > 0
            ])
            print(f"   MAE: {mae_str}")

    except Exception as e:
        print(f"   Error in batch forecast: {e}")
        import traceback
        traceback.print_exc()
        continue

# ========================================
# STEP 5: CONSOLIDATE AND SAVE FORECASTS
# ========================================

print(f"\n{'='*80}")
print(" CONSOLIDATING RESULTS ".center(80))
print('='*80)

forecast_summary = []

for cluster_id in FINAL_CLUSTERS:
    if len(all_forecasts[cluster_id]) == 0:
        print(f"   No forecasts for {cluster_id}")
        continue

    # Combine all windows for this cluster
    cluster_forecast_df = pd.concat(all_forecasts[cluster_id], ignore_index=True)

    # Save cluster forecasts
    output_file = FORECASTS_DIR / f'cluster_{cluster_id.replace(".", "_")}_forecasts.parquet'
    cluster_forecast_df.to_parquet(output_file, index=False)

    # Calculate overall metrics
    overall_mae = cluster_forecast_df['abs_error'].mean()
    overall_rmse = np.sqrt(cluster_forecast_df['squared_error'].mean())
    overall_mape = cluster_forecast_df['pct_error'].mean()

    forecast_summary.append({
        'cluster_id': cluster_id,
        'n_windows': len(all_forecasts[cluster_id]),
        'total_forecasts': len(cluster_forecast_df),
        'mae': overall_mae,
        'rmse': overall_rmse,
        'mape': overall_mape,
        'mean_actual': cluster_forecast_df['actual'].mean(),
        'mean_forecast': cluster_forecast_df['forecast_median'].mean(),
    })

    print(f"   {cluster_id}: {len(all_forecasts[cluster_id])} windows, MAE={overall_mae:,.2f}, MAPE={overall_mape:.2f}%")

# ========================================
# STEP 6: SAVE SUMMARY
# ========================================

print(f"\n{'='*80}")
print(" SAVING RESULTS ".center(80))
print('='*80)

if len(forecast_summary) > 0:
    df_summary = pd.DataFrame(forecast_summary)
    summary_file = FORECASTS_DIR / 'forecasts_summary.csv'
    df_summary.to_csv(summary_file, index=False)
    print(f"\n Summary saved: {summary_file}")

    print(f"\n{'='*80}")
    print(" FORECAST SUMMARY (CHRONOS-2) ".center(80))
    print('='*80)
    print(df_summary.to_string(index=False))

# ========================================
# FINAL SUMMARY
# ========================================

print(f"\n{'='*80}")
print(" CNAE 98 - STEP 4 COMPLETE (CHRONOS-2) ".center(80))
print('='*80)

print(f"\n Chronos-2 batch forecasting completed for {len(cluster_data)} clusters")
print(f"   Total windows processed: {n_windows}")
print(f"   Forecasts per cluster: {n_windows * FORECAST_HORIZON} hours")
print(f"\n Output directory: {FORECASTS_DIR}")

print(f"\n{'='*80}")
