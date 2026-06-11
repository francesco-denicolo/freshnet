"""
49_rq1_from_friedman.py — RQ1 riformulato in stile Friedman+Nemenyi
=====================================================================
Per ciascun forecaster × livello (globale, Q1, Q2, Q3, Q4):
  - Friedman best (lowest mean rank tra i k imputer, inclusi no_imp)
  - posizione di no_imp nel ranking
  - Δ mean rank tra no_imp e best
  - is no_imp CD-indistinguishable from best?

Risposta RQ1 per (forecaster, livello):
  no_imp ∈ CD-equiv set  → nessun imputer batte no_imp → IMPUTER NON AIUTA
  no_imp ∉ CD-equiv set  → almeno un imputer batte no_imp → IMPUTER AIUTA
"""
import os, functools
import pandas as pd
print = functools.partial(print, flush=True)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

fr = pd.read_parquet(f'{RESULTS_DIR}/friedman_per_forecaster.parquet')
print(f'Loaded {len(fr)} rows from friedman_per_forecaster.parquet')

rows = []
for (level, fc), g in fr.groupby(['level', 'forecaster']):
    g = g.sort_values('mean_rank')
    best = g.iloc[0]
    no_imp_row = g[g.imputer == 'no_imp']
    if len(no_imp_row) == 0:
        # Some forecasters (lgb_nolags, mlp_nolags) only have no_imp → SKIP
        continue
    no_imp_row = no_imp_row.iloc[0]
    delta_rank = float(no_imp_row.mean_rank - best.mean_rank)
    rows.append({
        'level': level,
        'forecaster': fc,
        'k_imputers': int(best.k),
        'N_series': int(best.N),
        'kendall_W': float(best.kendall_W),
        'kendall_cat': best.kendall_cat,
        'CD': float(best.CD),
        'friedman_best_imputer': best.imputer,
        'best_mean_rank': float(best.mean_rank),
        'no_imp_mean_rank': float(no_imp_row.mean_rank),
        'no_imp_rank_position': int(no_imp_row.rank_position),
        'delta_rank_no_imp_vs_best': delta_rank,
        'no_imp_is_cd_equiv': bool(no_imp_row.cd_indistinguishable),
        'imputer_helps': not bool(no_imp_row.cd_indistinguishable),
    })

out = pd.DataFrame(rows).sort_values(['level', 'forecaster'])
out.to_parquet(f'{RESULTS_DIR}/rq1_friedman_summary.parquet', index=False)
print(f'\nSaved: rq1_friedman_summary.parquet ({len(out)} rows)')

# Pretty print
for level in ['global', 'quartile_Q1', 'quartile_Q2', 'quartile_Q3', 'quartile_Q4']:
    sub = out[out.level == level]
    if len(sub) == 0: continue
    print(f'\n=== {level} ===')
    print(f"{'forecaster':<14} {'best_imp':<16} {'W':>5} {'CD':>5} "
          f"{'no_imp_pos':>10} {'Δrank':>8} {'no_imp_CD?':>10} {'imp_helps?':>10}")
    print('-' * 90)
    for _, r in sub.iterrows():
        cd_flag = 'YES' if r.no_imp_is_cd_equiv else 'no'
        help_flag = 'YES' if r.imputer_helps else 'no'
        print(f'{r.forecaster:<14} {r.friedman_best_imputer:<16} '
              f'{r.kendall_W:>5.3f} {r.CD:>5.3f} '
              f'{r.no_imp_rank_position:>3}/{r.k_imputers:<5} '
              f'{r.delta_rank_no_imp_vs_best:>+8.3f} '
              f'{cd_flag:>10} {help_flag:>10}')

# Summary across levels
print('\n\n=== SUMMARY: per forecaster, in quanti livelli l\'imputer aiuta? ===')
summary = (out.groupby('forecaster')['imputer_helps']
           .agg(['sum', 'count']).reset_index())
summary.columns = ['forecaster', 'n_levels_helps', 'n_levels_total']
summary['fraction_helps'] = summary.n_levels_helps / summary.n_levels_total
print(summary.to_string(index=False))

print('\nDONE')
