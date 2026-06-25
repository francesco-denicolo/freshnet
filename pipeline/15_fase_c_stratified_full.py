"""
15_fase_c_stratified_full.py — Analisi stratificata su 72 celle (9 imputer × 8 forecaster)
=============================================================================================
Rifa Opzione A (best per quartile di volume) e Opzione B (best per 16 gruppi vol×stockout)
con la matrice completa.
"""
import os, functools, numpy as np, pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

strat = pd.read_parquet(os.path.join(RESULTS_DIR, 'stratification.parquet'))

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
            'LGB imputer':'lgb',
            'DLinear':'dlinear',
            'Forward Fill':'forward_fill','Seasonal Naive':'seasonal_naive',
            'Linear Interp':'linear_interp','SAITS':'saits'}

def get_path(imp, fc):
    fc_safe = FC_FILE[fc]
    if fc == 'Chronos-bolt':
        imp_safe = 'no_imp' if imp == 'No imputation' else IMP_FILE[imp]
        return os.path.join(RESULTS_DIR, f'{imp_safe}__{fc_safe}_test_per_series.parquet')
    if imp == 'No imputation':
        if fc in ['Global Mean','DoW Mean','MA (K=56)']:
            return os.path.join(RESULTS_DIR, f'naive_{fc_safe}_test_per_series.parquet')
        return os.path.join(RESULTS_DIR, f'{fc_safe}_test_per_series.parquet')
    imp_safe = IMP_FILE[imp]
    if fc in ['LGB (no lags)', 'MLP (no lags)']:
        return os.path.join(RESULTS_DIR, f'{fc_safe}_test_per_series.parquet')
    return os.path.join(RESULTS_DIR, f'{imp_safe}__{fc_safe}_test_per_series.parquet')

print('='*72)
print('  ANALISI STRATIFICATA — 72 celle (9 imputer × 8 forecaster)')
print('='*72)

print('\n1. Caricamento celle...')
cells = {}
for imp in IMPUTERS:
    for fc in FORECASTERS:
        p = get_path(imp, fc)
        if os.path.exists(p):
            ps = pd.read_parquet(p)
            merged = strat[['store_id','product_id','vol_bin','so_bin']].merge(
                ps[['store_id','product_id','hourly_wape']], on=['store_id','product_id']
            ).dropna()
            cells[(imp, fc)] = merged
print(f'  Loaded {len(cells)}/72 cells')

# ===========================================================================
# OPZIONE A — Miglior combinazione per ogni quartile di volume
# ===========================================================================
print('\n' + '='*72)
print('  OPZIONE A — Best combinazione per quartile di VOLUME (4 gruppi)')
print('='*72)

for q in ['Q1','Q2','Q3','Q4']:
    rows = []
    for (imp, fc), df in cells.items():
        sub = df[df['vol_bin'] == q]
        if len(sub) == 0: continue
        w = sub['hourly_wape'].dropna()
        rows.append({'imp':imp, 'fc':fc, 'wape_med':w.median(), 'n':len(sub)})
    df_q = pd.DataFrame(rows).sort_values('wape_med').reset_index(drop=True)
    n_grp = int(df_q['n'].iloc[0]) if len(df_q) > 0 else 0
    print(f'\n  QUARTILE VOLUME {q}  (n={n_grp:,} serie)')
    print(f'  Top 10 combinazioni:')
    print(f'  {"Rank":<5}{"Imputer":<22}{"Forecaster":<18}{"WAPE med":>10}')
    print('  ' + '-'*55)
    for i, r in df_q.head(10).iterrows():
        mark = ' ★' if i == 0 else ''
        print(f'  {i+1:<5}{r["imp"]:<22}{r["fc"]:<18}{r["wape_med"]:>10.4f}{mark}')

# Summary table
print('\n  RIEPILOGO — Miglior cella per quartile di volume:')
for q in ['Q1','Q2','Q3','Q4']:
    rows = []
    for (imp, fc), df in cells.items():
        sub = df[df['vol_bin'] == q]
        if len(sub) == 0: continue
        rows.append({'imp':imp, 'fc':fc, 'w':sub['hourly_wape'].dropna().median()})
    df_q = pd.DataFrame(rows).sort_values('w')
    best = df_q.iloc[0]
    print(f'    {q}: {best["imp"]:<22} + {best["fc"]:<18} → {best["w"]:.4f}')

# ===========================================================================
# OPZIONE B — 16 gruppi (volume × stockout)
# ===========================================================================
print('\n' + '='*72)
print('  OPZIONE B — Best combinazione per gruppo (Volume × Stockout = 16)')
print('='*72)

results_16 = []
for vol in ['Q1','Q2','Q3','Q4']:
    for so in ['Q1','Q2','Q3','Q4']:
        rows = []
        for (imp, fc), df in cells.items():
            sub = df[(df['vol_bin']==vol) & (df['so_bin']==so)]
            if len(sub) == 0: continue
            rows.append({'imp':imp, 'fc':fc, 'w':sub['hourly_wape'].dropna().median(), 'n':len(sub)})
        df_g = pd.DataFrame(rows).sort_values('w').reset_index(drop=True)
        best = df_g.iloc[0]
        results_16.append({'vol':vol, 'so':so, 'n':int(best['n']),
                            'best_imp':best['imp'], 'best_fc':best['fc'], 'wape':best['w']})

df_res = pd.DataFrame(results_16)
df_res.to_parquet(os.path.join(RESULTS_DIR, 'best_per_group_16_v2.parquet'), index=False)

print(f'\n  Tabella 4x4 (righe=volume, colonne=stockout):')
print(f'  {"":12} {"SO=Q1":<28} {"SO=Q2":<28} {"SO=Q3":<28} {"SO=Q4":<28}')
print('  ' + '-' * 124)
for vol in ['Q4','Q3','Q2','Q1']:
    line = f'  Vol={vol}:      '
    for so in ['Q1','Q2','Q3','Q4']:
        r = df_res[(df_res['vol']==vol) & (df_res['so']==so)].iloc[0]
        combo = f'{r["best_imp"][:8]}+{r["best_fc"][:10]}'
        line += f'{combo:<20}{r["wape"]:.3f} (n={r["n"]:>4}) '
    print(line[:200])

# Winners count
print('\n  Vincitori aggregati (su 16 gruppi):')
print('  Forecaster:')
wins_fc = df_res['best_fc'].value_counts()
for fc, n in wins_fc.items(): print(f'    {fc:<24} {n}/16')
print('  Imputer:')
wins_imp = df_res['best_imp'].value_counts()
for imp, n in wins_imp.items(): print(f'    {imp:<24} {n}/16')

# Compact 4x4 matrix (only combo name)
print('\n  Matrice 4×4 compatta (solo miglior combinazione):')
print(f'  {"":12} {"SO=Q1":<30} {"SO=Q2":<30} {"SO=Q3":<30} {"SO=Q4":<30}')
for vol in ['Q4','Q3','Q2','Q1']:
    line = f'  Vol={vol}:      '
    for so in ['Q1','Q2','Q3','Q4']:
        r = df_res[(df_res['vol']==vol) & (df_res['so']==so)].iloc[0]
        combo = f'{r["best_imp"][:8]}+{r["best_fc"][:8]}={r["wape"]:.3f}'
        line += f'{combo:<32}'
    print(line)

# ===========================================================================
# Test stratificato per Chronos vs ML (per ogni imputer × quartile volume)
# ===========================================================================
print('\n' + '='*72)
print('  CROSSOVER: Chronos-bolt vs MLP M5, stratificato per imputer × volume')
print('='*72)

def rank_biserial(x, y):
    from scipy import stats
    diff = x - y; diff = diff[diff != 0]
    if len(diff) == 0: return 0.0
    ranks = stats.rankdata(np.abs(diff))
    T_plus = ranks[diff > 0].sum(); T_minus = ranks[diff < 0].sum()
    T_total = T_plus + T_minus
    return (T_plus - T_minus) / T_total if T_total > 0 else 0.0

def cliff_effect_label(d):
    a = abs(d)
    if a < 0.147: return 'neg '
    if a < 0.33:  return 'smal'
    if a < 0.474: return 'med '
    return 'LARG'

print(f'\n  {"Imputer":<24}{"Q":>3} {"n":>7} {"ΔMLP-Chr":>10} {"r_rb":>8} {"Effect":>7}  Dominio')
print('  ' + '-'*80)

crossover_results = []
for imp in IMPUTERS:
    if (imp, 'MLP (M5 lags)') not in cells or (imp, 'Chronos-bolt') not in cells:
        continue
    mlp_df = cells[(imp, 'MLP (M5 lags)')][['store_id','product_id','vol_bin','hourly_wape']].rename(columns={'hourly_wape':'wape_mlp'})
    chr_df = cells[(imp, 'Chronos-bolt')][['store_id','product_id','hourly_wape']].rename(columns={'hourly_wape':'wape_chr'})
    merged = mlp_df.merge(chr_df, on=['store_id','product_id']).dropna()
    for q in ['Q1','Q2','Q3','Q4']:
        sub = merged[merged['vol_bin'] == q]
        diff = sub['wape_mlp'].values - sub['wape_chr'].values
        # Cliff's delta paired
        pos = np.sum(diff > 0); neg = np.sum(diff < 0)
        d_cliff = (pos - neg) / len(diff) if len(diff) > 0 else 0.0
        r_rb = rank_biserial(sub['wape_mlp'].values, sub['wape_chr'].values)
        delta_med = np.median(sub['wape_mlp'].values) - np.median(sub['wape_chr'].values)
        eff = cliff_effect_label(d_cliff)
        winner = 'Chronos wins' if d_cliff > 0.147 else ('MLP wins' if d_cliff < -0.147 else 'tie')
        imp_label = imp if q == 'Q1' else ''
        print(f'  {imp_label:<24}{q:>3} {len(sub):>7,} {delta_med:>+10.4f} {r_rb:>+8.3f} {eff:>7}  {winner}')
        crossover_results.append({'imp':imp, 'q':q, 'delta':delta_med, 'r_rb':r_rb, 'cliff':d_cliff, 'winner':winner})
    print()

pd.DataFrame(crossover_results).to_parquet(os.path.join(RESULTS_DIR, 'crossover_vs_mlp_9imp.parquet'), index=False)

print('\n'+'='*72)
print('  DONE — 15_fase_c_stratified_full.py')
print('='*72)
