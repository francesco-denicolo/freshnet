"""
22_pareto_frontier_per_quartile.py — Frontiera di Pareto per ogni quartile di volume
======================================================================================
Per ogni quartile di volume (Q1-Q4), calcola la frontiera di Pareto WAPE × |WPE|
sulle serie di quel quartile. Confronta come cambia la frontiera per regime.
"""
import os, functools, numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMPUTERS = ['No imputation', 'Media condizionata', 'Media globale',
            'Mediana condizionata', 'Mediana globale',
            'LGB imputer', 'DLinear',
            'Forward Fill', 'Seasonal Naive', 'Linear Interp', 'SAITS']
FORECASTERS = ['Global Mean', 'DoW Mean', 'MA (K=21)',
               'LGB (no lags)', 'LGB (M5 lags)',
               'MLP (no lags)', 'MLP (M5 lags)', 'Chronos-bolt']

FC_FILE = {'Global Mean':'global_mean','DoW Mean':'dow_mean','MA (K=21)':'ma_k21',
           'LGB (no lags)':'lgb_nolags','LGB (M5 lags)':'lgb_m5lags',
           'MLP (no lags)':'mlp_nolags','MLP (M5 lags)':'mlp_m5lags',
           'Chronos-bolt':'chronos_bolt'}
IMP_FILE = {'Media condizionata':'media_cond','Media globale':'media_glob',
            'Mediana condizionata':'mediana_cond','Mediana globale':'mediana_glob',
            'LGB imputer':'lgb','DLinear':'dlinear',
            'Forward Fill':'forward_fill','Seasonal Naive':'seasonal_naive',
            'Linear Interp':'linear_interp','SAITS':'saits'}

FC_COLORS = {
    'Global Mean':    '#bcbcbc',
    'DoW Mean':       '#969696',
    'MA (K=21)':      '#737373',
    'LGB (no lags)':  '#fdae61',
    'LGB (M5 lags)':  '#f46d43',
    'MLP (no lags)':  '#74add1',
    'MLP (M5 lags)':  '#4575b4',
    'Chronos-bolt':   '#d73027',
}
FC_MARKERS = {
    'Global Mean':   's',
    'DoW Mean':      's',
    'MA (K=21)':     's',
    'LGB (no lags)': '^',
    'LGB (M5 lags)': '^',
    'MLP (no lags)': 'D',
    'MLP (M5 lags)': 'D',
    'Chronos-bolt':  'o',
}

def get_path(imp, fc):
    fc_safe = FC_FILE[fc]
    if fc == 'Chronos-bolt':
        imp_safe = 'no_imp' if imp == 'No imputation' else IMP_FILE[imp]
        return os.path.join(RESULTS_DIR, f'{imp_safe}__chronos_bolt_test_per_series.parquet')
    if imp == 'No imputation':
        if fc in ['Global Mean','DoW Mean','MA (K=21)']:
            return os.path.join(RESULTS_DIR, f'naive_{fc_safe}_test_per_series.parquet')
        return os.path.join(RESULTS_DIR, f'{fc_safe}_test_per_series.parquet')
    imp_safe = IMP_FILE[imp]
    if fc in ['LGB (no lags)','MLP (no lags)']:
        return os.path.join(RESULTS_DIR, f'{fc_safe}_test_per_series.parquet')
    return os.path.join(RESULTS_DIR, f'{imp_safe}__{fc_safe}_test_per_series.parquet')

# Load stratification
print('Loading stratification...')
strat = pd.read_parquet(os.path.join(RESULTS_DIR, 'stratification.parquet'))[['store_id','product_id','vol_bin']]

# Load all 88 cells with per-series WAPE/WPE
print('Loading 88 cells with per-series data...')
cells_data = {}
for imp in IMPUTERS:
    for fc in FORECASTERS:
        path = get_path(imp, fc)
        if not os.path.exists(path): continue
        ps = pd.read_parquet(path)
        merged = strat.merge(ps[['store_id','product_id','hourly_wape','hourly_wpe']],
                              on=['store_id','product_id'], how='inner').dropna()
        cells_data[(imp, fc)] = merged

print(f'  Loaded {len(cells_data)} cells')

# Compute WAPE/|WPE| per quartile
print('\nComputing metrics per quartile...')
quartile_data = {}
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    points = []
    for (imp, fc), df in cells_data.items():
        sub = df[df['vol_bin'] == q]
        if len(sub) == 0: continue
        wape_med = sub['hourly_wape'].median()
        wpe_med = sub['hourly_wpe'].median()
        points.append({
            'imputer': imp, 'forecaster': fc,
            'wape_med': wape_med, 'wpe_med': wpe_med,
            'abs_wpe_med': abs(wpe_med),
            'n': len(sub)
        })
    quartile_data[q] = pd.DataFrame(points)
    print(f'  Q{q[1]}: {len(quartile_data[q])} cells, n_serie ≈ {quartile_data[q]["n"].iloc[0]:,}')

# Compute Pareto frontier for each quartile
def pareto_mask(df):
    n = len(df)
    is_pareto = np.ones(n, dtype=bool)
    wape = df['wape_med'].values
    abs_wpe = df['abs_wpe_med'].values
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if (wape[j] <= wape[i] and abs_wpe[j] <= abs_wpe[i] and
                (wape[j] < wape[i] or abs_wpe[j] < abs_wpe[i])):
                is_pareto[i] = False
                break
    return is_pareto

print('\nComputing Pareto frontiers per quartile...')
pareto_per_q = {}
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    df_q = quartile_data[q]
    df_q['pareto'] = pareto_mask(df_q)
    pareto_per_q[q] = df_q
    print(f'  Q{q[1]}: {df_q["pareto"].sum()} cells on Pareto frontier')

# Save full data
all_data = []
for q, df_q in pareto_per_q.items():
    df_q['quartile'] = q
    all_data.append(df_q)
df_all = pd.concat(all_data, ignore_index=True)
df_all.to_parquet(os.path.join(RESULTS_DIR, 'pareto_per_quartile.parquet'), index=False)
print(f'\nSalvato: pareto_per_quartile.parquet ({len(df_all)} righe)')

# === FIGURA: 2x2 grid, una frontiera per quartile ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

quartile_order = ['Q1', 'Q2', 'Q3', 'Q4']
quartile_labels = {
    'Q1': 'Q1 — Volume basso',
    'Q2': 'Q2 — Volume medio-basso',
    'Q3': 'Q3 — Volume medio-alto',
    'Q4': 'Q4 — Volume alto'
}

# Compute global axis limits
all_wape_pareto = pd.concat([df_q[df_q['pareto']]['wape_med'] for df_q in pareto_per_q.values()])
all_wpe_pareto = pd.concat([df_q[df_q['pareto']]['abs_wpe_med'] for df_q in pareto_per_q.values()])
all_wape_all = pd.concat([df_q['wape_med'] for df_q in pareto_per_q.values()])
all_wpe_all = pd.concat([df_q['abs_wpe_med'] for df_q in pareto_per_q.values()])
xlim = (all_wape_all.min() * 0.98, all_wape_all.max() * 1.02)
ylim = (all_wpe_all.min() * 0.95, all_wpe_all.max() * 1.05)

for idx, q in enumerate(quartile_order):
    ax = axes[idx // 2, idx % 2]
    df_q = pareto_per_q[q]
    df_pareto = df_q[df_q['pareto']].sort_values('wape_med')
    df_dom = df_q[~df_q['pareto']]

    # Dominated cells (background)
    ax.scatter(df_dom['wape_med'], df_dom['abs_wpe_med'],
               s=40, color='lightgray', alpha=0.5, edgecolor='gray',
               linewidth=0.4, zorder=1)

    # Pareto cells (foreground, colored by forecaster)
    plotted_fc = set()
    for fc in FORECASTERS:
        sub = df_pareto[df_pareto['forecaster'] == fc]
        if len(sub) == 0: continue
        label = fc if fc not in plotted_fc else None
        plotted_fc.add(fc)
        ax.scatter(sub['wape_med'], sub['abs_wpe_med'],
                   s=110, color=FC_COLORS[fc], marker=FC_MARKERS[fc],
                   edgecolor='black', linewidth=0.7, label=label, zorder=3)

    # Connect Pareto frontier
    ax.plot(df_pareto['wape_med'], df_pareto['abs_wpe_med'],
            color='black', linestyle='--', linewidth=1.2, alpha=0.5, zorder=2)

    # Highlight min WAPE
    p_min = df_pareto.iloc[0]
    ax.scatter(p_min['wape_med'], p_min['abs_wpe_med'],
               s=300, marker='*', color='gold', edgecolor='black',
               linewidth=1.2, zorder=5)
    ax.annotate(f"{p_min['imputer'][:12]}\n+ {p_min['forecaster'][:12]}",
                (p_min['wape_med'], p_min['abs_wpe_med']),
                xytext=(8, -10), textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))

    # Stats annotation
    n_pareto = df_q['pareto'].sum()
    n_serie = df_q['n'].iloc[0]
    ax.text(0.02, 0.98,
            f'{quartile_labels[q]}\n'
            f'n_serie = {n_serie:,}\n'
            f'Pareto: {n_pareto}/{len(df_q)} cells\n'
            f'Min WAPE = {p_min["wape_med"]:.4f}',
            transform=ax.transAxes,
            ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

    ax.set_xlabel('WAPE mediana per-serie', fontsize=10)
    ax.set_ylabel('|WPE| mediana per-serie', fontsize=10)
    ax.set_title(quartile_labels[q], fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if idx == 0:
        ax.legend(loc='lower right', fontsize=8, ncol=2)

fig.suptitle('Frontiera di Pareto WAPE × |WPE| per quartile di volume',
             fontsize=14, y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig20_pareto_per_quartile.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSalvata: fig20_pareto_per_quartile.png')

# === Sintesi tabella ===
print('\n' + '='*80)
print('  SINTESI: come cambia la frontiera per quartile')
print('='*80)

for q in quartile_order:
    df_q = pareto_per_q[q]
    df_p = df_q[df_q['pareto']].sort_values('wape_med')
    n_pareto = len(df_p)
    p_min_wape = df_p.iloc[0]
    p_min_wpe = df_p.iloc[-1]
    print(f'\n  {quartile_labels[q]} (n_serie = {df_q["n"].iloc[0]:,})')
    print(f'    Cells on Pareto frontier: {n_pareto}/{len(df_q)}')
    print(f'    Min WAPE:    {p_min_wape["imputer"]:<22} + {p_min_wape["forecaster"]:<18}'
          f'  WAPE={p_min_wape["wape_med"]:.4f}  |WPE|={p_min_wape["abs_wpe_med"]:.4f}')
    print(f'    Min |WPE|:   {p_min_wpe["imputer"]:<22} + {p_min_wpe["forecaster"]:<18}'
          f'  WAPE={p_min_wpe["wape_med"]:.4f}  |WPE|={p_min_wpe["abs_wpe_med"]:.4f}')

# Forecaster representation on each frontier
print('\n' + '='*80)
print('  RAPPRESENTAZIONE FORECASTER SULLE FRONTIERE')
print('='*80)
print(f'\n  {"":12}{"  ".join(quartile_order)}')
for fc in FORECASTERS:
    counts = [pareto_per_q[q][(pareto_per_q[q]['forecaster']==fc) &
                                (pareto_per_q[q]['pareto'])].shape[0]
              for q in quartile_order]
    print(f'  {fc:<22}  {counts[0]:>2}    {counts[1]:>2}    {counts[2]:>2}    {counts[3]:>2}')

print('\n' + '='*80)
print('  IMPUTER MIGLIORE (min WAPE) PER QUARTILE')
print('='*80)
for q in quartile_order:
    df_q = pareto_per_q[q]
    df_p = df_q[df_q['pareto']].sort_values('wape_med').head(5)
    print(f'\n  {quartile_labels[q]} - top 5 Pareto:')
    for _, r in df_p.iterrows():
        print(f'    {r["imputer"]:<22} + {r["forecaster"]:<18}'
              f'  WAPE={r["wape_med"]:.4f}  |WPE|={r["abs_wpe_med"]:.4f}')

print('\n' + '='*80)
print('  DONE')
print('='*80)
