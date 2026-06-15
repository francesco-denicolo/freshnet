"""
42c_rq2_pairwise_concordance.py — RQ2 via pairwise concordance probability
==========================================================================
Alternativa robusta al Spearman ρ aggregato (n=13 → fragile). Per ogni
forecaster:
  - 13 imputer → 78 coppie distinte (A, B) di imputer
  - Per ciascuna coppia: P(concordant) calcolata su N=50K serie
       concordant_i = [ sign(rec_A − rec_B) == sign(fc_A − fc_B) ]
       P_pair = mean(concordant) across series
  - Distribuzione di 78 P_pair per forecaster, ciascuna con N=50K → robusta

Vantaggi rispetto a Spearman aggregato:
  - Ogni misura è basata su 50K serie (non 13 imputer)
  - 78 misure per forecaster (non 1) → distribuzione interpretabile
  - SE statistico ~±0.002 (vs ±0.32 per Spearman su n=13)
  - Resistente all'inclusione/esclusione di singoli imputer

Output:
  - rq2_pairwise_concordance.parquet   summary per forecaster (n_pairs, median,
                                       IQR, min, max, %above 0.5, bootstrap CI)
  - rq2_pairwise_concordance_pairs.parquet   tutte le 78 P_pair per forecaster
  - fig_rq2_pairwise_concordance.png    boxplot/violin distribuzione
"""
import os, glob, functools, itertools, time
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
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
rec_vec_all = np.array([recovery[i] for i in imputers_all])
print(f'{len(imputers_all)} imputer con WAPE_recovery')

NON_HPO_FC = {'chronos_bolt', 'timesfm', 'global_mean', 'dow_mean', 'ma_k21'}

def find_parquet(imp, fc):
    p_hpo = f'{RESULTS_DIR}/{imp}__{fc}_hpo_test_per_series.parquet'
    p_plain = f'{RESULTS_DIR}/{imp}__{fc}_test_per_series.parquet'
    if os.path.exists(p_hpo): return p_hpo
    if os.path.exists(p_plain): return p_plain
    return None

panels = ['mlp_m5lags', 'lgb_m5lags', 'tft', 'chronos_bolt', 'timesfm',
          'global_mean', 'dow_mean', 'ma_k21']
panel_titles = {
    'mlp_m5lags': 'MLP_M5', 'lgb_m5lags': 'LGB_M5', 'tft': 'TFT',
    'chronos_bolt': 'Chronos-bolt', 'timesfm': 'TimesFM',
    'global_mean': 'GlobalMean', 'dow_mean': 'DoWMean', 'ma_k21': 'MA_K21',
}

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
# Main per-forecaster loop
# ----------------------------------------------------------------------
results = []
all_pairs_rows = []
for fc in panels:
    print(f'\n=== {panel_titles[fc]} ===')
    # Build wide matrix: rows = series, cols = imputer (subset of imputers_all
    # for which the cell exists)
    dfs = {}
    for imp in imputers_all:
        p = find_parquet(imp, fc)
        if p is None:
            continue
        dfs[imp] = pd.read_parquet(p).set_index(['store_id', 'product_id'])['hourly_wape']
    if len(dfs) < 5:
        print(f'  SKIP: only {len(dfs)} imputer cells available')
        continue
    avail = [imp for imp in imputers_all if imp in dfs]
    common = dfs[avail[0]].index
    for imp in avail[1:]:
        common = common.intersection(dfs[imp].index)
    N = len(common)
    print(f'  Common series across {len(avail)} imputers: {N:,}')
    W = np.full((N, len(avail)), np.nan, dtype=np.float64)
    for j, imp in enumerate(avail):
        W[:, j] = dfs[imp].loc[common].values
    valid_rows = np.all(~np.isnan(W), axis=1)
    W = W[valid_rows]
    N_valid = W.shape[0]
    print(f'  Valid series (no NaN any imputer): {N_valid:,}')

    rec_vec = np.array([recovery[i] for i in avail])

    # All pairs (i, j) with i < j
    pair_indices = list(itertools.combinations(range(len(avail)), 2))
    n_pairs = len(pair_indices)
    print(f'  Pairs: {n_pairs}  (= {len(avail)} choose 2)')

    pair_concordance = np.empty(n_pairs)
    pair_names = []
    for k, (i, j) in enumerate(pair_indices):
        sgn_r = np.sign(rec_vec[i] - rec_vec[j])
        sgn_f = np.sign(W[:, i] - W[:, j])  # vector
        # Concordance: same sign (and not both zero)
        concord = (sgn_r == sgn_f) & (sgn_r != 0) & (sgn_f != 0)
        pair_concordance[k] = concord.mean()
        pair_names.append(f'{avail[i]}__VS__{avail[j]}')

    # Save per-pair rows
    for k, (i, j) in enumerate(pair_indices):
        all_pairs_rows.append({
            'forecaster': panel_titles[fc],
            'pair': pair_names[k],
            'imp_a': avail[i], 'imp_b': avail[j],
            'rec_a': float(rec_vec[i]), 'rec_b': float(rec_vec[j]),
            'p_concordant': float(pair_concordance[k]),
            'n_series': int(N_valid),
        })

    # Aggregate stats on 78 (or fewer) P_pair
    med = float(np.median(pair_concordance))
    q25, q75 = float(np.percentile(pair_concordance, 25)), float(np.percentile(pair_concordance, 75))
    mn, mx = float(pair_concordance.min()), float(pair_concordance.max())
    above_50 = float((pair_concordance > 0.5).mean())
    ci_lo, ci_hi = bootstrap_median_ci(pair_concordance, n_boot=1000, alpha=0.05, seed=42 + hash(fc) % 1000)

    print(f'  P_pair (78 = pair concordance probabilities):')
    print(f'    median       = {med:.4f}  CI95 [{ci_lo:.4f}, {ci_hi:.4f}]')
    print(f'    IQR          = [{q25:.4f}, {q75:.4f}]')
    print(f'    min / max    = {mn:.4f} / {mx:.4f}')
    print(f'    % with > 0.5 = {above_50:.2%}')

    results.append({
        'forecaster': panel_titles[fc],
        'n_imputers': len(avail),
        'n_pairs': n_pairs,
        'n_series': N_valid,
        'median': med, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'q25': q25, 'q75': q75,
        'min': mn, 'max': mx,
        'pct_above_50': above_50,
    })

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
summary = pd.DataFrame(results)
summary.to_parquet(f'{RESULTS_DIR}/rq2_pairwise_concordance.parquet', index=False)
print(f'\nSaved: rq2_pairwise_concordance.parquet ({len(summary)} forecaster)')

pairs_df = pd.DataFrame(all_pairs_rows)
pairs_df.to_parquet(f'{RESULTS_DIR}/rq2_pairwise_concordance_pairs.parquet', index=False)
print(f'Saved: rq2_pairwise_concordance_pairs.parquet ({len(pairs_df)} pair rows)')

# ----------------------------------------------------------------------
# Figure: boxplot/violin of P_pair distribution per forecaster
# ----------------------------------------------------------------------
print('\nBuilding figure...')
fig, ax = plt.subplots(figsize=(14, 8))
data = []
labels = []
for fc in panels:
    if fc not in [r['forecaster'] for r in results]:
        # Maybe stored with panel_title
        pass
    title = panel_titles[fc]
    sub = pairs_df[pairs_df.forecaster == title]
    if len(sub) == 0: continue
    data.append(sub.p_concordant.values)
    labels.append(title)

positions = np.arange(len(data)) + 1
parts = ax.violinplot(data, positions=positions, widths=0.7,
                       showmeans=False, showmedians=True, showextrema=False)
# Color
colors = ['#4575b4', '#f46d43', '#7b3294', '#d73027', '#b15928',
          '#2ca02c', '#bcbd22', '#17becf']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i % len(colors)])
    pc.set_alpha(0.6)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.2)
parts['cmedians'].set_color('white')
parts['cmedians'].set_linewidth(2.5)

# Reference at 0.5
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6, linewidth=1)

# Annotate per forecaster
for i, fc_title in enumerate(labels):
    r = summary[summary.forecaster == fc_title].iloc[0]
    txt = (f'median = {r["median"]:.3f}\n'
           f'CI95 [{r["ci_lo"]:.3f}, {r["ci_hi"]:.3f}]\n'
           f'{r["pct_above_50"]:.0%} pairs > 0.5\n'
           f'(n_pairs={int(r["n_pairs"])}, N_serie={int(r["n_series"]):,})')
    ax.text(positions[i], 0.03, txt, ha='center', va='bottom',
            fontsize=8.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.9, edgecolor='gray'))

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(-0.02, 1.02)
ax.set_ylabel('P(concordant)  per pair across N≈50K series', fontsize=12)
ax.set_title('RQ2 — Pairwise concordance recovery vs forecasting\n'
             'P_pair = P(sign(rec_A−rec_B) == sign(fc_A−fc_B)) across series',
             fontsize=13, pad=12)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.tick_params(axis='y', labelsize=11)

plt.tight_layout()
out_fig = f'{FIG_DIR}/fig_rq2_pairwise_concordance.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f'Saved: {out_fig}')

# ----------------------------------------------------------------------
# Pretty summary table
# ----------------------------------------------------------------------
print('\n=== Summary table: pairwise concordance ===')
print(summary[['forecaster', 'n_pairs', 'median', 'ci_lo', 'ci_hi',
               'q25', 'q75', 'min', 'max', 'pct_above_50']].to_string(index=False,
              float_format='%.4f'))

print('\nDONE')
