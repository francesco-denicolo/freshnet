"""
20_pareto_frontier_plot.py — Visualizzazione frontiera di Pareto WAPE × |WPE|
==============================================================================
Genera fig19_pareto_frontier.png:
- 56 celle dominate in grigio (sottofondo)
- 32 celle Pareto-ottimali colorate per forecaster
- Linea che connette i punti della frontiera
- 3 punti rappresentativi (min WAPE, knee, min |WPE|) evidenziati
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
FORECASTERS = ['Global Mean', 'DoW Mean', 'MA (K=56)',
               'LGB (no lags)', 'LGB (M5 lags)',
               'MLP (no lags)', 'MLP (M5 lags)', 'Chronos-bolt']

FC_FILE = {'Global Mean':'global_mean','DoW Mean':'dow_mean','MA (K=56)':'ma_k56',
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
    'MA (K=56)':      '#737373',
    'LGB (no lags)':  '#fdae61',
    'LGB (M5 lags)':  '#f46d43',
    'MLP (no lags)':  '#74add1',
    'MLP (M5 lags)':  '#4575b4',
    'Chronos-bolt':   '#d73027',
}
FC_MARKERS = {
    'Global Mean':   's',
    'DoW Mean':      's',
    'MA (K=56)':     's',
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
        if fc in ['Global Mean','DoW Mean','MA (K=56)']:
            return os.path.join(RESULTS_DIR, f'naive_{fc_safe}_test_per_series.parquet')
        return os.path.join(RESULTS_DIR, f'{fc_safe}_test_per_series.parquet')
    imp_safe = IMP_FILE[imp]
    if fc in ['LGB (no lags)','MLP (no lags)']:
        return os.path.join(RESULTS_DIR, f'{fc_safe}_test_per_series.parquet')
    return os.path.join(RESULTS_DIR, f'{imp_safe}__{fc_safe}_test_per_series.parquet')

# Load all 88 cells
print('Loading 88 cells...')
points = []
for imp in IMPUTERS:
    for fc in FORECASTERS:
        path = get_path(imp, fc)
        if not os.path.exists(path): continue
        ps = pd.read_parquet(path)
        wape_med = ps['hourly_wape'].dropna().median()
        wpe_med = ps['hourly_wpe'].dropna().median()
        points.append({
            'imputer': imp, 'forecaster': fc,
            'wape_med': wape_med, 'wpe_med': wpe_med,
            'abs_wpe_med': abs(wpe_med)
        })
df = pd.DataFrame(points)

# Compute Pareto frontier (minimize wape_med AND abs_wpe_med)
def is_dominated(p, df):
    for _, q in df.iterrows():
        if (q['wape_med'] <= p['wape_med'] and q['abs_wpe_med'] <= p['abs_wpe_med'] and
            (q['wape_med'] < p['wape_med'] or q['abs_wpe_med'] < p['abs_wpe_med'])):
            return True
    return False

df['pareto'] = [not is_dominated(p, df.drop(i)) for i, p in df.iterrows()]
print(f'Pareto frontier: {df["pareto"].sum()}/88 cells')

df_pareto = df[df['pareto']].sort_values('wape_med').reset_index(drop=True)
df_dominated = df[~df['pareto']]

# === Figure ===
fig, ax = plt.subplots(figsize=(13, 8))

# Dominated cells (background, gray)
ax.scatter(df_dominated['wape_med'], df_dominated['abs_wpe_med'],
           s=50, color='lightgray', alpha=0.6, edgecolor='gray',
           linewidth=0.5, label='Dominated (56 cells)', zorder=1)

# Pareto cells (foreground, colored by forecaster)
plotted_fc = set()
for fc in FORECASTERS:
    sub = df_pareto[df_pareto['forecaster'] == fc]
    if len(sub) == 0: continue
    label = fc if fc not in plotted_fc else None
    plotted_fc.add(fc)
    ax.scatter(sub['wape_med'], sub['abs_wpe_med'],
               s=120, color=FC_COLORS[fc], marker=FC_MARKERS[fc],
               edgecolor='black', linewidth=0.8, label=label, zorder=3)

# Connect Pareto frontier with line
ax.plot(df_pareto['wape_med'], df_pareto['abs_wpe_med'],
        color='black', linestyle='--', linewidth=1.5, alpha=0.5,
        label='Pareto frontier (32 cells)', zorder=2)

# Highlight 3 key points
# 1. Min WAPE
p_min_wape = df_pareto.iloc[0]
ax.scatter(p_min_wape['wape_med'], p_min_wape['abs_wpe_med'],
           s=400, marker='*', color='gold', edgecolor='black', linewidth=1.5,
           zorder=5, label='Min WAPE')
ax.annotate(f"Min WAPE\n{p_min_wape['imputer']}\n+ {p_min_wape['forecaster']}",
            (p_min_wape['wape_med'], p_min_wape['abs_wpe_med']),
            xytext=(15, -10), textcoords='offset points', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# 2. Knee point — closest to origin in normalized (WAPE, |WPE|) space
_xn = (df_pareto['wape_med'] - df_pareto['wape_med'].min()) / (
    df_pareto['wape_med'].max() - df_pareto['wape_med'].min() + 1e-9)
_yn = (df_pareto['abs_wpe_med'] - df_pareto['abs_wpe_med'].min()) / (
    df_pareto['abs_wpe_med'].max() - df_pareto['abs_wpe_med'].min() + 1e-9)
p_knee = df_pareto.assign(_dist=(_xn**2 + _yn**2)**0.5).sort_values('_dist').iloc[0]
ax.scatter(p_knee['wape_med'], p_knee['abs_wpe_med'],
           s=400, marker='*', color='lime', edgecolor='black', linewidth=1.5,
           zorder=5, label='Knee point')
ax.annotate(f"Knee point\n{p_knee['imputer']}\n+ {p_knee['forecaster']}",
            (p_knee['wape_med'], p_knee['abs_wpe_med']),
            xytext=(15, 10), textcoords='offset points', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.6))

# 3. Min |WPE|
p_min_wpe = df_pareto.iloc[-1]
ax.scatter(p_min_wpe['wape_med'], p_min_wpe['abs_wpe_med'],
           s=400, marker='*', color='cyan', edgecolor='black', linewidth=1.5,
           zorder=5, label='Min |WPE|')
ax.annotate(f"Min |WPE|\n{p_min_wpe['imputer']}\n+ {p_min_wpe['forecaster']}",
            (p_min_wpe['wape_med'], p_min_wpe['abs_wpe_med']),
            xytext=(-50, 15), textcoords='offset points', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8))

# Annotate Pareto regions with shaded zones
ax.axhspan(0.7, 1.0, alpha=0.05, color='red', zorder=0)  # Chronos zone
ax.axhspan(0.2, 0.45, alpha=0.05, color='blue', zorder=0)  # MLP zone
ax.axhspan(0.05, 0.2, alpha=0.05, color='green', zorder=0)  # Naive zone

ax.text(1.005, 0.85, 'Zone A: Chronos-bolt\n(min WAPE, max |WPE|)',
        fontsize=10, color='darkred', fontweight='bold', alpha=0.7)
ax.text(1.005, 0.30, 'Zone B: MLP M5\n(knee — best trade-off)',
        fontsize=10, color='darkblue', fontweight='bold', alpha=0.7)
ax.text(1.105, 0.10, 'Zone C: Naive/Mediana\n(min |WPE|)',
        fontsize=10, color='darkgreen', fontweight='bold', alpha=0.7)

ax.set_xlabel('WAPE mediana per-serie (accuratezza)', fontsize=12)
ax.set_ylabel('|WPE| mediana per-serie (bias)', fontsize=12)
ax.set_title('Frontiera di Pareto: trade-off WAPE × |WPE| sulla matrice 11×8 = 88 celle\n'
             '32 celle Pareto-ottimali (linea tratteggiata) vs 56 dominate (grigio)',
             fontsize=13, pad=12)
ax.grid(alpha=0.3, zorder=0)
ax.legend(loc='upper right', fontsize=9, ncol=2)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig19_pareto_frontier.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Salvata: fig19_pareto_frontier.png')

# Save data
df.to_parquet(os.path.join(RESULTS_DIR, 'pareto_frontier_full.parquet'), index=False)
print(f'Salvato: pareto_frontier_full.parquet ({len(df)} righe con flag pareto)')
