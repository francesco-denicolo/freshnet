"""
11_fase_c_stratified.py — Analisi stratificata per volume e tasso di stockout
==============================================================================
Divide le serie in 4×4 = 16 gruppi (quartili di volume × quartili di stockout_rate)
e calcola WAPE/WPE mediana per 4 forecaster × 3 imputer = 12 celle per gruppo.

Forecaster: DoW Mean, LGB M5, MLP M5, Chronos-bolt
Imputer: No imputation, Mediana condizionata, DLinear
"""
import os, functools, numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H_START, H_END = 6, 23; N_HOURS = H_END - H_START  # 17
N_DAYS_TRAIN = 90  # days 1-90 for stratification

# ===========================================================================
# 1. Compute stratification variables (volume and stockout_rate) per series
# ===========================================================================
print('='*72)
print('  FASE C — ANALISI STRATIFICATA (volume × stockout_rate, quartili)')
print('='*72)

print('\n1. Computing stratification variables...')
df_train = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_train = df_train.sort_values(['store_id', 'product_id', 'dt']).reset_index(drop=True)

# Parse hourly arrays and slice to 6-22
sales = np.array(df_train['hours_sale'].tolist(), dtype=np.float32)[:, H_START:H_END]
stock = np.array(df_train['hours_stock_status'].tolist(), dtype=np.float32)[:, H_START:H_END]

# Group by series (store_id, product_id)
series_keys = df_train.groupby(['store_id','product_id'], sort=False).indices
n_series = len(series_keys)
print(f'  Series: {n_series:,}')

vols = np.zeros(n_series, dtype=np.float32)
stockout_rates = np.zeros(n_series, dtype=np.float32)
sids = np.zeros(n_series, dtype=np.int64)
pids = np.zeros(n_series, dtype=np.int64)

for i, ((sid, pid), idx) in enumerate(series_keys.items()):
    s = sales[idx]  # (n_days_series, 17)
    k = stock[idx]  # (n_days_series, 17)
    instock = k == 0
    # Volume: mean sales on in-stock hours
    total_instock_sales = s[instock].sum()
    n_instock_hours = instock.sum()
    vols[i] = total_instock_sales / max(n_instock_hours, 1)
    # Stockout rate: fraction of stockout hours (6-22, in training)
    stockout_rates[i] = 1.0 - (n_instock_hours / max(s.size, 1))
    sids[i] = sid
    pids[i] = pid
    if (i+1) % 10000 == 0:
        print(f'    {i+1:,}/{n_series:,}')

# Save stratification data
strat = pd.DataFrame({
    'store_id': sids, 'product_id': pids,
    'volume': vols, 'stockout_rate': stockout_rates,
})

# Compute quartiles
vol_25, vol_50, vol_75 = np.percentile(vols, [25, 50, 75])
so_25, so_50, so_75 = np.percentile(stockout_rates, [25, 50, 75])

def vol_bin(v):
    if v <= vol_25: return 'Q1'
    elif v <= vol_50: return 'Q2'
    elif v <= vol_75: return 'Q3'
    else: return 'Q4'

def so_bin(s):
    if s <= so_25: return 'Q1'
    elif s <= so_50: return 'Q2'
    elif s <= so_75: return 'Q3'
    else: return 'Q4'

strat['vol_bin'] = strat['volume'].apply(vol_bin)
strat['so_bin'] = strat['stockout_rate'].apply(so_bin)

print(f'\n  Volume quartili:    [{vol_25:.4f}, {vol_50:.4f}, {vol_75:.4f}]')
print(f'  Stockout quartili:  [{so_25:.4f}, {so_50:.4f}, {so_75:.4f}]')

# Counts per group
print('\n  Serie per gruppo (volume × stockout_rate):')
pivot_cnt = strat.pivot_table(
    index='vol_bin', columns='so_bin',
    values='volume', aggfunc='count',
    observed=False
).reindex(index=['Q1','Q2','Q3','Q4'], columns=['Q1','Q2','Q3','Q4'])
print(pivot_cnt.to_string())

# Save stratification
strat_path = os.path.join(RESULTS_DIR, 'stratification.parquet')
strat.to_parquet(strat_path, index=False)
print(f'\n  Salvato: {strat_path}')

del df_train, sales, stock

# ===========================================================================
# 2. Load per-series results for selected cells
# ===========================================================================
print('\n2. Caricamento per-series results (celle selezionate)...')

FORECASTERS = ['DoW Mean', 'LGB (M5 lags)', 'MLP (M5 lags)', 'Chronos-bolt']
IMPUTERS = ['No imputation', 'Mediana condizionata', 'DLinear']

FC_FILE = {
    'DoW Mean': 'dow_mean',
    'LGB (M5 lags)': 'lgb_m5lags',
    'MLP (M5 lags)': 'mlp_m5lags',
    'Chronos-bolt': 'chronos_bolt',
}
IMP_FILE = {
    'Mediana condizionata': 'mediana_cond',
    'DLinear': 'dlinear',
}

def get_path(imp, fc):
    fc_safe = FC_FILE[fc]
    if fc == 'Chronos-bolt':
        imp_safe = 'no_imp' if imp == 'No imputation' else IMP_FILE[imp]
        return os.path.join(RESULTS_DIR, f'{imp_safe}__{fc_safe}_test_per_series.parquet')
    if imp == 'No imputation':
        if fc == 'DoW Mean':
            return os.path.join(RESULTS_DIR, f'naive_{fc_safe}_test_per_series.parquet')
        else:
            return os.path.join(RESULTS_DIR, f'{fc_safe}_test_per_series.parquet')
    imp_safe = IMP_FILE[imp]
    return os.path.join(RESULTS_DIR, f'{imp_safe}__{fc_safe}_test_per_series.parquet')

# Load all 12 cells
cells = {}
for imp in IMPUTERS:
    for fc in FORECASTERS:
        path = get_path(imp, fc)
        if not os.path.exists(path):
            print(f'  MISSING: {imp} × {fc} -> {path}')
            continue
        ps = pd.read_parquet(path)
        cells[(imp, fc)] = ps
print(f'  Loaded {len(cells)} cells')

# ===========================================================================
# 3. Compute stratified metrics (16 groups × 12 cells)
# ===========================================================================
print('\n3. Calcolo metriche stratificate...')

results = []
for vol_b in ['Q1', 'Q2', 'Q3', 'Q4']:
    for so_b in ['Q1', 'Q2', 'Q3', 'Q4']:
        # Series in this group
        mask = (strat['vol_bin'] == vol_b) & (strat['so_bin'] == so_b)
        grp_keys = strat[mask][['store_id', 'product_id']].copy()
        n_grp = len(grp_keys)

        for (imp, fc), ps in cells.items():
            # Merge group with cell results
            merged = grp_keys.merge(ps, on=['store_id', 'product_id'], how='inner')
            wape_med = merged['hourly_wape'].dropna().median()
            wpe_med = merged['hourly_wpe'].dropna().median()
            results.append({
                'vol_bin': vol_b, 'so_bin': so_b,
                'imputer': imp, 'forecaster': fc,
                'wape_med': wape_med, 'wpe_med': wpe_med,
                'n_series': n_grp,
            })

df_res = pd.DataFrame(results)
df_res.to_parquet(os.path.join(RESULTS_DIR, 'stratified_results.parquet'), index=False)
print(f'  Salvato: stratified_results.parquet ({len(df_res)} righe)')

# ===========================================================================
# 4. Stampa tabella riassuntiva
# ===========================================================================
print('\n4. Tabelle riassuntive (WAPE mediana per gruppo)...')

BINS = ['Q1', 'Q2', 'Q3', 'Q4']
GROUP_ORDER = [(v, s) for v in BINS for s in BINS]

for vol_b, so_b in GROUP_ORDER:
    sub = df_res[(df_res['vol_bin'] == vol_b) & (df_res['so_bin'] == so_b)]
    if sub.empty:
        continue
    n_grp = sub['n_series'].iloc[0]
    print(f'\n  Volume={vol_b}, Stockout={so_b}  (n={n_grp:,})')
    pivot = sub.pivot(index='imputer', columns='forecaster', values='wape_med')
    pivot = pivot.reindex(index=IMPUTERS, columns=FORECASTERS)
    print(pivot.round(4).to_string())

# ===========================================================================
# 5. Visualizzazione: 9 heatmap (3×3 griglia)
# ===========================================================================
print('\n5. Generating heatmaps...')

fig, axes = plt.subplots(4, 4, figsize=(24, 18))

# Find global min/max for consistent coloring
all_wape = df_res['wape_med'].dropna()
vmin = all_wape.min() - 0.01
vmax = all_wape.max() + 0.01
im = None

for i, vol_b in enumerate(['Q4', 'Q3', 'Q2', 'Q1']):  # Q4 (high) on top
    for j, so_b in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):  # Q1 (low) on left
        ax = axes[i, j]
        sub = df_res[(df_res['vol_bin'] == vol_b) & (df_res['so_bin'] == so_b)]
        if sub.empty:
            ax.set_visible(False); continue
        n_grp = sub['n_series'].iloc[0]
        mat = sub.pivot(index='imputer', columns='forecaster', values='wape_med')
        mat = mat.reindex(index=IMPUTERS, columns=FORECASTERS).values

        im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(FORECASTERS)))
        ax.set_yticks(range(len(IMPUTERS)))
        ax.set_xticklabels(FORECASTERS, rotation=30, ha='right', fontsize=7)
        ax.set_yticklabels(IMPUTERS, fontsize=7)

        # Annotate best cell
        best_val = np.nanmin(mat)
        for ii in range(len(IMPUTERS)):
            for jj in range(len(FORECASTERS)):
                v = mat[ii, jj]
                if np.isnan(v): continue
                is_best = abs(v - best_val) < 1e-5
                txt = f'{v:.3f}' + (' ★' if is_best else '')
                color = 'white' if (v - vmin) / (vmax - vmin) > 0.6 else 'black'
                ax.text(jj, ii, txt, ha='center', va='center',
                        fontsize=6, fontweight='bold' if is_best else 'normal',
                        color=color)

        ax.set_title(f'Vol={vol_b}, SO={so_b}  (n={n_grp:,})', fontsize=9)

# Add row/column headers
for j, so_b in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    axes[0, j].annotate(f'Stockout {so_b}', xy=(0.5, 1.25), xycoords='axes fraction',
                        ha='center', fontsize=11, fontweight='bold')

for i, vol_b in enumerate(['Q4', 'Q3', 'Q2', 'Q1']):
    axes[i, 0].annotate(f'Volume {vol_b}', xy=(-0.35, 0.5), xycoords='axes fraction',
                        ha='right', va='center', rotation=90, fontsize=11, fontweight='bold')

fig.suptitle('Matrice stratificata: WAPE mediana per gruppo (quartili volume × stockout)',
             fontsize=14, y=1.01)
# Shared colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label='WAPE mediana')

fig.savefig(os.path.join(FIG_DIR, 'fig09_stratified_matrix.png'),
            dpi=140, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig09_stratified_matrix.png')

# ===========================================================================
# 6. Trend plot: WAPE vs stockout_rate per forecaster (best imputer)
# ===========================================================================
print('\n6. Trend plot...')

fig, ax = plt.subplots(figsize=(12, 6))
COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
MARKERS = ['o', 's', '^', 'D']

# For each forecaster, plot best imputer's WAPE vs stockout_rate group
for idx, fc in enumerate(FORECASTERS):
    for imp in IMPUTERS:
        sub = df_res[(df_res['forecaster'] == fc) & (df_res['imputer'] == imp)]
        # Aggregate across volume bins for simpler view: pooled across vol_bins
        # For each stockout level, take mean of 3 volume groups
        agg = sub.groupby('so_bin')['wape_med'].mean().reindex(['Q1','Q2','Q3','Q4'])
        label = f'{fc} + {imp}'
        linestyle = '-' if imp == 'No imputation' else ('--' if imp == 'Mediana condizionata' else ':')
        ax.plot([0, 1, 2, 3], agg.values, color=COLORS[idx],
                marker=MARKERS[idx], linestyle=linestyle,
                label=label if imp == 'No imputation' else None,
                alpha=0.7 if imp == 'No imputation' else 0.4)

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['SO Q1', 'SO Q2', 'SO Q3', 'SO Q4'])
ax.set_xlabel('Stockout rate (quartile)', fontsize=12)
ax.set_ylabel('WAPE mediana (media sui 4 livelli di volume)', fontsize=12)
ax.set_title('Trend WAPE vs Stockout rate per forecaster (solo No imputation, media sui 4 livelli di volume)', fontsize=12)
ax.grid(alpha=0.3)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig10_trend_stockout.png'),
            dpi=140, bbox_inches='tight')
plt.close(fig)
print('  Salvata: fig10_trend_stockout.png')

# ===========================================================================
# 7. Findings: dove aiuta l'imputation?
# ===========================================================================
print('\n7. Dove aiuta l\'imputation?')
print('  (Δ WAPE med = no_imp - best_imputer, positivo = imputation aiuta)')

for vol_b, so_b in GROUP_ORDER:
    print(f'\n  Volume={vol_b}, Stockout={so_b}:')
    for fc in FORECASTERS:
        sub = df_res[(df_res['vol_bin'] == vol_b) & (df_res['so_bin'] == so_b) &
                      (df_res['forecaster'] == fc)]
        if sub.empty:
            continue
        no_imp = sub[sub['imputer'] == 'No imputation']['wape_med'].values
        other = sub[sub['imputer'] != 'No imputation']
        if len(no_imp) == 0 or other.empty:
            continue
        best_other = other.loc[other['wape_med'].idxmin()]
        delta = no_imp[0] - best_other['wape_med']
        sign = '✓' if delta > 0.001 else ('✗' if delta < -0.001 else '~')
        print(f'    {fc:<18} no_imp={no_imp[0]:.4f}  best_imp={best_other["wape_med"]:.4f} '
              f'({best_other["imputer"]})  Δ={delta:+.4f} {sign}')

print('\n' + '='*72)
print('  DONE — 11_fase_c_stratified.py')
print('='*72)
