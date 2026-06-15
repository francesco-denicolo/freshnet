"""
42d_tft_dynamics_analysis.py — Test ipotesi "TFT preferisce imputer dinamici"
==============================================================================
Per ogni imputer, calcola DYNAMICITY:
  - Per ciascuna serie i:
      Decomponi i valori imputati come y_imp(d,h) = μ_imp(h) + ε_imp(d,h)
      Decomponi i valori osservati come y_obs(d,h) = μ_obs(h) + ε_obs(d,h)
      DYN_i = std(ε_imp) / std(ε_obs)   (= dinamicità imputata vs naturale)
  - DYNAMICITY(imputer) = median(DYN_i across series)

Output:
  - rq2_imputer_dynamicity.parquet   13 righe (imputer, DYNAMICITY, n_series)
  - fig_rq2_imputer_dynamicity.png   barre + ranking confronto

Test ipotesi:
  Spearman ρ tra DYNAMICITY ranking e TFT forecasting ranking (atteso: alto pos.)
  Spearman ρ tra DYNAMICITY ranking e Recovery ranking         (atteso: ≈ 0)
  Per altri forecaster (MLP_M5, LGB_M5, Chronos, ecc.):        (atteso: bassi)
"""
import os, glob, functools, time
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COMPLETED_DIR = os.path.join(DATA_DIR, 'completed_sales_622')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

H_START, H_END = 6, 23
N_HOURS = H_END - H_START

# Recovery values (same as 42b)
recovery = {
    'mediana_glob':   0.8090,
    'mediana_cond':   0.8462,
    'lgb':            0.8472,
    'imputeformer':   0.8666,
    'itransformer':   0.9302,
    'saits':          0.9431,
    'media_glob':     0.9454,
    'media_cond':     0.9560,
    'dlinear':        0.9513,
    'timesnet':       1.0405,
    'linear_interp':  1.0473,
    'seasonal_naive': 1.0638,
    'forward_fill':   1.1878,
}

# Minimum stockouts & in-stock per series
MIN_STOCKOUTS = 10
MIN_INSTOCK = 10

# ----------------------------------------------------------------------
# Dynamicity per imputer
# ----------------------------------------------------------------------
def compute_dynamicity(imputer_name):
    """Return median DYN_i across series, and counts."""
    fpath = os.path.join(COMPLETED_DIR, f'{imputer_name}.parquet')
    if not os.path.exists(fpath):
        print(f'  MISSING: {fpath}')
        return None
    df = pd.read_parquet(fpath)
    n_rows = len(df)

    # Stack array columns to (n_rows, N_HOURS) arrays
    sales = np.stack(df['hours_sale'].values).astype(np.float64)
    # hours_sale already covers hours 6-22 (17 hours) post-processing
    if sales.shape[1] == 24:
        sales = sales[:, H_START:H_END]
    stock = np.stack(df['hours_stock_status'].values).astype(np.int8)
    if stock.shape[1] == 24:
        stock = stock[:, H_START:H_END]

    # Build flat arrays
    series_idx_row = pd.Categorical(
        df['store_id'].astype(str) + '_' + df['product_id'].astype(str)
    ).codes
    n_series_total = series_idx_row.max() + 1
    # Each row repeated N_HOURS times
    series_flat = np.repeat(series_idx_row, N_HOURS).astype(np.int32)
    hour_flat = np.tile(np.arange(N_HOURS, dtype=np.int8), n_rows)
    sales_flat = sales.ravel()
    stock_flat = stock.ravel()

    # Build long DataFrame for groupby (compact dtypes)
    long_df = pd.DataFrame({
        'series': series_flat,
        'hour': hour_flat,
        'sale': sales_flat.astype(np.float32),
        'stock': stock_flat,
    })

    # Split
    in_stock = long_df[long_df['stock'] == 0]
    stockout = long_df[long_df['stock'] == 1]

    # Compute hourly mean per (series, hour) and subtract
    in_stock = in_stock.copy()
    in_stock['mu'] = in_stock.groupby(['series', 'hour'])['sale'].transform('mean')
    in_stock['eps'] = in_stock['sale'] - in_stock['mu']

    stockout = stockout.copy()
    stockout['mu'] = stockout.groupby(['series', 'hour'])['sale'].transform('mean')
    stockout['eps'] = stockout['sale'] - stockout['mu']

    # Aggregate std per series
    agg_obs = in_stock.groupby('series')['eps'].agg(['std', 'count']).rename(
        columns={'std': 'std_obs', 'count': 'n_obs'})
    agg_imp = stockout.groupby('series')['eps'].agg(['std', 'count']).rename(
        columns={'std': 'std_imp', 'count': 'n_imp'})

    merged = agg_obs.join(agg_imp, how='inner').dropna()
    # Filter
    merged = merged[(merged['n_obs'] >= MIN_INSTOCK) &
                    (merged['n_imp'] >= MIN_STOCKOUTS) &
                    (merged['std_obs'] > 0)]
    merged['DYN'] = merged['std_imp'] / merged['std_obs']

    dyn_med = float(merged['DYN'].median())
    dyn_q25, dyn_q75 = float(merged['DYN'].quantile(0.25)), float(merged['DYN'].quantile(0.75))
    return {
        'imputer': imputer_name,
        'DYN_median': dyn_med,
        'DYN_q25': dyn_q25,
        'DYN_q75': dyn_q75,
        'n_series_valid': len(merged),
        'n_series_total': n_series_total,
    }

# ----------------------------------------------------------------------
# Forecasting ranks per imputer × forecaster
# ----------------------------------------------------------------------
def get_forecasting_rank(forecaster):
    """For each imputer in `recovery`, get WAPE_h_med under `forecaster`,
    then rank ascending. Returns dict imputer -> rank (1 = best)."""
    out = {}
    for imp in recovery:
        # Try hpo first, then plain
        for suffix in ['_hpo_test_per_series', '_test_per_series']:
            p = f'{RESULTS_DIR}/{imp}__{forecaster}{suffix}.parquet'
            if os.path.exists(p):
                df = pd.read_parquet(p)
                out[imp] = float(df['hourly_wape'].median())
                break
    # Ranks
    sorted_imps = sorted(out, key=lambda k: out[k])
    ranks = {imp: i + 1 for i, imp in enumerate(sorted_imps)}
    return out, ranks

# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
print('1. Computing DYNAMICITY per imputer...')
t0 = time.time()
results = []
for imp in recovery:
    print(f'\n  --- {imp} ---')
    t1 = time.time()
    res = compute_dynamicity(imp)
    if res is None: continue
    results.append(res)
    print(f'    DYN median = {res["DYN_median"]:.4f}  IQR [{res["DYN_q25"]:.4f}, {res["DYN_q75"]:.4f}]')
    print(f'    series valid: {res["n_series_valid"]:,} / {res["n_series_total"]:,}')
    print(f'    time: {time.time() - t1:.1f}s')

dyn_df = pd.DataFrame(results).sort_values('DYN_median').reset_index(drop=True)
dyn_df['DYN_rank'] = dyn_df['DYN_median'].rank(ascending=True).astype(int)
print(f'\n  Total time: {time.time() - t0:.0f}s')

# ----------------------------------------------------------------------
# Cross-reference with forecasting ranks
# ----------------------------------------------------------------------
print('\n2. Cross-reference DYN ranking with each forecaster ranking...')
forecasters = ['tft', 'mlp_m5lags', 'lgb_m5lags', 'chronos_bolt', 'timesfm',
               'global_mean', 'dow_mean', 'ma_k21']

# Build long dyn_df with forecasting ranks per imputer
dyn_df['recovery'] = dyn_df['imputer'].map(recovery)
dyn_df['recovery_rank'] = dyn_df['recovery'].rank(ascending=True).astype(int)

for fc in forecasters:
    wape_med, ranks = get_forecasting_rank(fc)
    dyn_df[f'{fc}_wape'] = dyn_df['imputer'].map(wape_med)
    dyn_df[f'{fc}_rank'] = dyn_df['imputer'].map(ranks)

# ----------------------------------------------------------------------
# Spearman correlations: DYN ranking vs each forecaster ranking
# ----------------------------------------------------------------------
from scipy.stats import spearmanr
print('\n3. Spearman ρ (DYN ranking vs other rankings):\n')
corr_rows = []
print(f"{'comparison':<35} {'Spearman ρ':>12} {'p-value':>10}  {'n':>4}")
print('-' * 70)

# DYN vs recovery
sub = dyn_df.dropna(subset=['DYN_rank', 'recovery_rank'])
rho, pval = spearmanr(sub['DYN_rank'], sub['recovery_rank'])
corr_rows.append({'comparison': 'DYN vs Recovery', 'rho': rho, 'pval': pval, 'n': len(sub)})
print(f"{'DYN vs Recovery':<35} {rho:>+12.4f} {pval:>10.4f}  {len(sub):>4}")

for fc in forecasters:
    sub = dyn_df.dropna(subset=['DYN_rank', f'{fc}_rank'])
    if len(sub) < 5:
        continue
    rho, pval = spearmanr(sub['DYN_rank'], sub[f'{fc}_rank'])
    # We want negative ρ for "DYN high → forecast rank low (good)"
    corr_rows.append({'comparison': f'DYN vs {fc}_rank', 'rho': rho, 'pval': pval, 'n': len(sub)})
    print(f"{'DYN vs ' + fc + ' rank':<35} {rho:>+12.4f} {pval:>10.4f}  {len(sub):>4}")

corr_df = pd.DataFrame(corr_rows)
corr_df.to_parquet(f'{RESULTS_DIR}/rq2_imputer_dynamicity_correlations.parquet', index=False)

# ----------------------------------------------------------------------
# Save tables
# ----------------------------------------------------------------------
dyn_df.to_parquet(f'{RESULTS_DIR}/rq2_imputer_dynamicity.parquet', index=False)
print(f'\n4. Saved: rq2_imputer_dynamicity.parquet ({len(dyn_df)} imputers)')
print(f'   Saved: rq2_imputer_dynamicity_correlations.parquet ({len(corr_df)} comparisons)')

# Pretty print full table
print('\n=== DYNAMICITY ranking table ===')
display_cols = ['imputer', 'DYN_median', 'DYN_rank', 'recovery_rank',
                'tft_rank', 'mlp_m5lags_rank', 'chronos_bolt_rank', 'timesfm_rank']
print(dyn_df[display_cols].to_string(index=False, float_format='%.4f'))

# ----------------------------------------------------------------------
# Figure: bar chart of DYN per imputer + scatter with TFT rank
# ----------------------------------------------------------------------
print('\n5. Building figure...')
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left panel: DYN per imputer (bars)
ax = axes[0]
order = dyn_df.sort_values('DYN_median')['imputer'].tolist()
xs = np.arange(len(order))
yvals = dyn_df.set_index('imputer').loc[order]['DYN_median'].values
q25 = dyn_df.set_index('imputer').loc[order]['DYN_q25'].values
q75 = dyn_df.set_index('imputer').loc[order]['DYN_q75'].values
# Color by family (static vs dynamic — visually based on DYN value)
colors = []
for v in yvals:
    if v < 0.05:
        colors.append('#d73027')   # red = static
    elif v < 0.30:
        colors.append('#fdae61')   # orange = intermediate
    else:
        colors.append('#4575b4')   # blue = dynamic
ax.barh(xs, yvals, xerr=[yvals - q25, q75 - yvals],
        color=colors, edgecolor='black', alpha=0.85, error_kw={'capsize': 4})
ax.set_yticks(xs)
ax.set_yticklabels(order, fontsize=10)
ax.invert_yaxis()
ax.axvline(1.0, color='gray', linestyle='--', alpha=0.6, label='Naturale (DYN=1)')
ax.set_xlabel('DYN median = std_imp / std_obs (per serie)', fontsize=12)
ax.set_title('DYNAMICITY per imputer (variabilità residuo rispetto al naturale)',
             fontsize=12)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, axis='x', alpha=0.3, linestyle='--')

# Right panel: DYN rank vs TFT rank scatter
ax = axes[1]
sub = dyn_df.dropna(subset=['DYN_rank', 'tft_rank']).copy()
xv = sub['DYN_rank'].values
yv = sub['tft_rank'].values  # rank 1 = best forecaster
ax.scatter(xv, yv, s=180, c='#4575b4', edgecolor='black', linewidth=1.5, zorder=3)
for _, r in sub.iterrows():
    ax.annotate(r['imputer'], (r['DYN_rank'], r['tft_rank']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
# Diagonal (perfect negative correlation: high DYN_rank → low tft_rank = best)
mx = max(sub['DYN_rank'].max(), sub['tft_rank'].max())
ax.plot([1, mx], [mx, 1], '--', color='gray', alpha=0.5,
         label='Perfect: alto DYN → top TFT')
# Spearman ρ annotation
rho_tft = corr_df[corr_df.comparison == 'DYN vs tft_rank']['rho'].iloc[0] if 'DYN vs tft_rank' in corr_df.comparison.values else float('nan')
ax.set_xlabel('DYN rank  (1 = più statico, max = più dinamico)', fontsize=12)
ax.set_ylabel('TFT rank (1 = best, max = worst)', fontsize=12)
ax.set_title(f'DYN rank vs TFT rank   Spearman ρ = {rho_tft:+.3f}',
             fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
out_fig = f'{FIG_DIR}/fig_rq2_imputer_dynamicity.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'   Saved: {out_fig}')

print('\nDONE')
