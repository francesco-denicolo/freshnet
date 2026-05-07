"""
21_stratified_mc_correction.py — Correzione MC sui test stratificati
=====================================================================
Aggiunge Holm-Bonferroni e Benjamini-Hochberg ai p-value di:
  - spearman_p
  - jt_p_decreasing
nei 176 test stratificati di volume_stockout_tests.parquet.

Famiglia di test: 88 celle × 2 dimensioni (volume, stockout).
Multiple comparisons corretto SEPARATAMENTE per dimensione (88 test cad.).
"""
import os, functools, numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

print('='*72)
print('  CORREZIONE MULTIPLE COMPARISONS — test stratificati (176 test)')
print('='*72)

df = pd.read_parquet(os.path.join(RESULTS_DIR, 'volume_stockout_tests.parquet'))
print(f'\nLoaded {len(df)} test')
print(f'Dimensioni: {df["dimension"].unique().tolist()}')
print(f'Forecaster × imputer: {df["forecaster"].nunique()} × {df["imputer"].nunique()}')

# Apply MC correction SEPARATELY per dimension (volume vs stockout)
def add_mc(df_sub, p_col, prefix):
    p = df_sub[p_col].fillna(1.0).clip(lower=1e-300).values
    df_sub[f'{prefix}_holm'] = multipletests(p, method='holm')[1]
    df_sub[f'{prefix}_fdr_bh'] = multipletests(p, method='fdr_bh')[1]
    df_sub[f'{prefix}_sig_holm_05'] = df_sub[f'{prefix}_holm'] < 0.05
    df_sub[f'{prefix}_sig_fdr_05'] = df_sub[f'{prefix}_fdr_bh'] < 0.05
    return df_sub

results = []
for dim in ['volume', 'stockout']:
    sub = df[df['dimension'] == dim].copy()
    sub = add_mc(sub, 'spearman_p', 'spearman')
    sub = add_mc(sub, 'jt_p_decreasing', 'jt')
    results.append(sub)

df_corr = pd.concat(results, ignore_index=True)
df_corr.to_parquet(os.path.join(RESULTS_DIR, 'volume_stockout_tests_mc.parquet'), index=False)
print(f'\nSalvato: volume_stockout_tests_mc.parquet ({len(df_corr)} righe)')

# Sintesi
print('\n' + '='*72)
print('  SINTESI MULTIPLE COMPARISONS (88 test per dimensione)')
print('='*72)

for dim in ['volume', 'stockout']:
    sub = df_corr[df_corr['dimension'] == dim]
    n = len(sub)
    print(f'\n--- DIMENSIONE: {dim.upper()} ({n} test) ---')

    print(f'\n  SPEARMAN ρ:')
    print(f'    Significativi raw (p<0.05):           {(sub["spearman_p"]<0.05).sum()}/{n}')
    print(f'    Significativi Holm-Bonferroni (<0.05): {sub["spearman_sig_holm_05"].sum()}/{n}')
    print(f'    Significativi Benjamini-Hochberg:      {sub["spearman_sig_fdr_05"].sum()}/{n}')

    print(f'\n  JONCKHEERE-TERPSTRA (decreasing):')
    print(f'    Significativi raw (p<0.05):           {(sub["jt_p_decreasing"]<0.05).sum()}/{n}')
    print(f'    Significativi Holm-Bonferroni (<0.05): {sub["jt_sig_holm_05"].sum()}/{n}')
    print(f'    Significativi Benjamini-Hochberg:      {sub["jt_sig_fdr_05"].sum()}/{n}')

# Interesting cases: which cells become NON significant after correction?
print('\n' + '='*72)
print('  CELLE CHE PERDONO SIGNIFICATIVITÀ DOPO HOLM-BONFERRONI')
print('='*72)

for dim in ['volume', 'stockout']:
    sub = df_corr[df_corr['dimension'] == dim]
    lost_spearman = sub[(sub['spearman_p']<0.05) & (~sub['spearman_sig_holm_05'])]
    lost_jt = sub[(sub['jt_p_decreasing']<0.05) & (~sub['jt_sig_holm_05'])]
    print(f'\n  {dim.upper()}:')
    print(f'    Spearman: {len(lost_spearman)} celle perdono significatività')
    if len(lost_spearman) > 0:
        for _, r in lost_spearman.iterrows():
            print(f'      {r["imputer"]:<22} × {r["forecaster"]:<18}  ρ={r["spearman_rho"]:+.3f} '
                  f'(p_raw={r["spearman_p"]:.2e}, p_holm={r["spearman_holm"]:.2e})')
    print(f'    JT: {len(lost_jt)} celle perdono significatività')

# Effect size summary tables
print('\n' + '='*72)
print('  EFFECT SIZE (Cliff δ) DOPO CORREZIONE MC')
print('='*72)

for dim in ['volume', 'stockout']:
    sub = df_corr[df_corr['dimension'] == dim]
    print(f'\n  {dim.upper()}:')
    print(f'    Cliff δ large    (|δ|≥0.474): {(sub["cliff_effect"]=="large").sum()}/{len(sub)}')
    print(f'    Cliff δ medium   (|δ|≥0.33):  {(sub["cliff_effect"]=="medium").sum()}/{len(sub)}')
    print(f'    Cliff δ small    (|δ|≥0.147): {(sub["cliff_effect"]=="small").sum()}/{len(sub)}')
    print(f'    Cliff δ negligible:           {(sub["cliff_effect"]=="negligible").sum()}/{len(sub)}')

# Combined criterion: significant + large effect
print('\n' + '='*72)
print('  CRITERI COMBINATI: SIGNIFICATIVITÀ × EFFECT SIZE')
print('='*72)

for dim in ['volume', 'stockout']:
    sub = df_corr[df_corr['dimension'] == dim]
    n = len(sub)
    print(f'\n  {dim.upper()}:')
    sig_holm = sub['jt_sig_holm_05']
    large_eff = sub['cliff_effect'].isin(['large'])
    med_eff = sub['cliff_effect'].isin(['large', 'medium'])
    print(f'    Sig (Holm) + large effect:    {(sig_holm & large_eff).sum()}/{n}')
    print(f'    Sig (Holm) + medium+ effect:  {(sig_holm & med_eff).sum()}/{n}')
    print(f'    Sig (Holm) + small effect:    {(sig_holm & sub["cliff_effect"].isin(["small"])).sum()}/{n}')
    print(f'    Sig (Holm) + negligible:      {(sig_holm & sub["cliff_effect"].isin(["negligible"])).sum()}/{n}')
    print(f'    NOT sig (Holm):              {(~sig_holm).sum()}/{n}')

print('\n' + '='*72)
print('  DONE')
print('='*72)
