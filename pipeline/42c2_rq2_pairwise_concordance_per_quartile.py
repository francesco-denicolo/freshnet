"""
42c2_rq2_pairwise_concordance_per_quartile.py — RQ2 stratified per volume quartile
====================================================================================
Stessa metodologia di 42c (pairwise concordance recovery vs forecasting),
applicata stratificata per quartile di volume Q1-Q4.

Output:
  - rq2_pairwise_concordance_per_quartile.parquet   summary (8 fc × 4 Q = 32 righe)
  - rq2_pairwise_concordance_per_quartile_pairs.parquet  pair detail (~2500 righe)
  - fig_rq2_pairwise_concordance_per_quartile.png   4x2 grid

Risponde a: «Il pattern recovery → forecasting cambia con il regime di volume?»
"""
import os, glob, functools, itertools, time
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# Recovery values (same set as 42b, n=13)
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
imputers_all = list(recovery.keys())

NON_HPO_FC = {'chronos_bolt', 'timesfm', 'global_mean', 'dow_mean', 'ma_k56'}

def find_parquet(imp, fc):
    p_hpo = f'{RESULTS_DIR}/{imp}__{fc}_hpo_test_per_series.parquet'
    p_plain = f'{RESULTS_DIR}/{imp}__{fc}_test_per_series.parquet'
    if os.path.exists(p_hpo): return p_hpo
    if os.path.exists(p_plain): return p_plain
    return None

panels = ['mlp_m5lags', 'lgb_m5lags', 'tft', 'chronos_bolt', 'timesfm',
          'global_mean', 'dow_mean', 'ma_k56']
panel_titles = {
    'mlp_m5lags': 'MLP_M5', 'lgb_m5lags': 'LGB_M5', 'tft': 'TFT',
    'chronos_bolt': 'Chronos-bolt', 'timesfm': 'TimesFM',
    'global_mean': 'GlobalMean', 'dow_mean': 'DoWMean', 'ma_k56': 'MA_K56',
}

# ----------------------------------------------------------------------
# Volume quartiles
# ----------------------------------------------------------------------
print('1. Computing volume quartiles (gg 1-83)...')
df_tr = pd.read_parquet(os.path.join(DATA_DIR, 'frn50k_train.parquet'))
df_tr['dt_parsed'] = pd.to_datetime(df_tr['dt'])
df_tr = df_tr.sort_values(['store_id','product_id','dt_parsed']).reset_index(drop=True)
df_tr['day_num'] = df_tr.groupby(['store_id','product_id']).cumcount() + 1
vol = (df_tr[df_tr.day_num <= 83]
       .groupby(['store_id','product_id'])['sale_amount'].sum().reset_index())
vol['quartile'] = pd.qcut(vol['sale_amount'], q=4, labels=['Q1','Q2','Q3','Q4']).astype(str)
quart_map = vol.set_index(['store_id','product_id'])['quartile']
del df_tr
print(f'   Quartile counts: {quart_map.value_counts().to_dict()}')

# ----------------------------------------------------------------------
# Bootstrap CI for the median of N values
# ----------------------------------------------------------------------
def bootstrap_median_ci(arr, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        boots[k] = np.median(arr[rng.integers(0, n, n)])
    return np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])

# ----------------------------------------------------------------------
# Main per-forecaster × per-quartile loop
# ----------------------------------------------------------------------
print('\n2. Pairwise concordance per (forecaster, quartile)...')
results = []
all_pairs_rows = []

for fc in panels:
    print(f'\n=== {panel_titles[fc]} ===')

    # Load all imputer cells once, get the (store, product) index
    dfs = {}
    for imp in imputers_all:
        p = find_parquet(imp, fc)
        if p is None: continue
        dfs[imp] = pd.read_parquet(p).set_index(['store_id', 'product_id'])['hourly_wape']
    if len(dfs) < 5:
        print(f'  SKIP: only {len(dfs)} imputer cells available')
        continue
    avail = [imp for imp in imputers_all if imp in dfs]
    common = dfs[avail[0]].index
    for imp in avail[1:]:
        common = common.intersection(dfs[imp].index)

    # Add quartile to each series
    series_q = quart_map.reindex(common)
    print(f'  Common series across {len(avail)} imputers: {len(common):,}')

    # Build wide matrix once
    W = np.full((len(common), len(avail)), np.nan, dtype=np.float64)
    for j, imp in enumerate(avail):
        W[:, j] = dfs[imp].loc[common].values

    # Recovery vector
    rec_vec = np.array([recovery[i] for i in avail])

    # All pairs (i, j) with i < j
    pair_indices = list(itertools.combinations(range(len(avail)), 2))
    n_pairs = len(pair_indices)

    # Process per quartile
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        mask_q = (series_q == q).values
        W_q = W[mask_q]
        valid_rows = np.all(~np.isnan(W_q), axis=1)
        W_q = W_q[valid_rows]
        N_q = W_q.shape[0]
        if N_q < 100:
            print(f'    {q}: SKIP (only {N_q} valid)')
            continue

        pair_concordance = np.empty(n_pairs)
        for k, (i, j) in enumerate(pair_indices):
            sgn_r = np.sign(rec_vec[i] - rec_vec[j])
            sgn_f = np.sign(W_q[:, i] - W_q[:, j])
            concord = (sgn_r == sgn_f) & (sgn_r != 0) & (sgn_f != 0)
            pair_concordance[k] = concord.mean()

        # Save per-pair rows
        for k, (i, j) in enumerate(pair_indices):
            all_pairs_rows.append({
                'forecaster': panel_titles[fc], 'quartile': q,
                'imp_a': avail[i], 'imp_b': avail[j],
                'p_concordant': float(pair_concordance[k]),
                'n_series': int(N_q),
            })

        med = float(np.median(pair_concordance))
        q25, q75 = (float(np.percentile(pair_concordance, 25)),
                    float(np.percentile(pair_concordance, 75)))
        mn, mx = float(pair_concordance.min()), float(pair_concordance.max())
        above_50 = float((pair_concordance > 0.5).mean())
        ci_lo, ci_hi = bootstrap_median_ci(pair_concordance, n_boot=1000,
                                            seed=42 + hash(fc + q) % 1000)
        print(f'    {q}  N={N_q:,}  median={med:.3f}  CI[{ci_lo:.3f},{ci_hi:.3f}]  '
              f'IQR[{q25:.3f},{q75:.3f}]  pairs>0.5={above_50:.0%}')
        results.append({
            'forecaster': panel_titles[fc], 'quartile': q,
            'n_imputers': len(avail), 'n_pairs': n_pairs, 'n_series': N_q,
            'median': med, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'q25': q25, 'q75': q75, 'min': mn, 'max': mx,
            'pct_above_50': above_50,
        })

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
summary = pd.DataFrame(results)
summary.to_parquet(f'{RESULTS_DIR}/rq2_pairwise_concordance_per_quartile.parquet', index=False)
pairs_df = pd.DataFrame(all_pairs_rows)
pairs_df.to_parquet(f'{RESULTS_DIR}/rq2_pairwise_concordance_per_quartile_pairs.parquet', index=False)
print(f'\nSaved: rq2_pairwise_concordance_per_quartile.parquet ({len(summary)} rows)')

# ----------------------------------------------------------------------
# Figure: 4x2 grid (Q1-Q4 × {naive group, ML group}) — boxplots
# ----------------------------------------------------------------------
print('\n3. Building figure...')
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
quartiles = ['Q1', 'Q2', 'Q3', 'Q4']
order_fc = panels  # mlp_m5lags, lgb_m5lags, tft, chronos_bolt, timesfm, global_mean, dow_mean, ma_k56
colors = ['#4575b4', '#f46d43', '#7b3294', '#d73027', '#b15928',
          '#2ca02c', '#bcbd22', '#17becf']

# Top row: ML+lag + foundation + TFT (5 forecaster)
# Bottom row: naive aggregati (3 forecaster)
groups = [
    ('ML + DL + Foundation', ['mlp_m5lags', 'lgb_m5lags', 'tft', 'chronos_bolt', 'timesfm']),
    ('Naive aggregati', ['global_mean', 'dow_mean', 'ma_k56'])
]

for row, (group_title, group_fcs) in enumerate(groups):
    for col, q in enumerate(quartiles):
        ax = axes[row, col]
        data = []; labels = []; cols = []
        for fc in group_fcs:
            title = panel_titles[fc]
            sub = pairs_df[(pairs_df.forecaster == title) & (pairs_df.quartile == q)]
            if len(sub) == 0: continue
            data.append(sub.p_concordant.values)
            labels.append(title)
            cols.append(colors[order_fc.index(fc)])
        if not data: continue
        positions = np.arange(len(data)) + 1
        parts = ax.violinplot(data, positions=positions, widths=0.7,
                              showmeans=False, showmedians=True, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(cols[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.8)
        parts['cmedians'].set_color('white')
        parts['cmedians'].set_linewidth(2)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6, linewidth=1)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9, rotation=20)
        ax.set_ylim(-0.05, 1.05)
        if col == 0:
            ax.set_ylabel(f'{group_title}\nP(concordant)', fontsize=11)
        ax.set_title(f'{q}   (N={int(summary[(summary.quartile==q)].n_series.iloc[0]):,})', fontsize=11)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')

fig.suptitle('RQ2 — Pairwise concordance recovery vs forecasting, per volume quartile\n'
             'Mediana di 78 P_pair per forecaster × quartile; ciascuna calcolata su N≈12.5K serie',
             fontsize=13, y=1.005)
plt.tight_layout()
out_fig = f'{FIG_DIR}/fig_rq2_pairwise_concordance_per_quartile.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'Saved: {out_fig}')

# ----------------------------------------------------------------------
# Pivot summary for readability
# ----------------------------------------------------------------------
print('\n=== Summary table (rows = forecaster, cols = quartile, values = median P_pair) ===')
piv = summary.pivot(index='forecaster', columns='quartile', values='median')
# Add globale row from existing rq2_pairwise_concordance.parquet if available
glob_path = f'{RESULTS_DIR}/rq2_pairwise_concordance.parquet'
if os.path.exists(glob_path):
    glob = pd.read_parquet(glob_path).set_index('forecaster')['median']
    piv['Globale'] = glob
    piv = piv[['Globale','Q1','Q2','Q3','Q4']]
print(piv.to_string(float_format='%.4f'))

print('\nDONE')
