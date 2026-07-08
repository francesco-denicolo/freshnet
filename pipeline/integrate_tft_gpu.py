"""
integrate_tft_gpu.py — integra i risultati TFT-GPU nella matrice e rigenera tutto
=================================================================================
Dopo aver riportato in locale i file `{imp}__tft_gpu_test_per_series.parquet` dal
cloud, questo script:

  1. valida la presenza dei 14 file GPU (avvisa sui mancanti);
  2. fa BACKUP delle celle TFT attuali (CPU) in results/_tft_cpu_backup/ + manifest;
  3. SOSTITUISCE `{imp}__tft_hpo_test_per_series.parquet` con i risultati GPU
     (inclusa la nuova cella imputeformer__tft che chiude il buco della matrice);
  4. ri-esegue la catena statistica/figure nell'ordine corretto:
       45 (Friedman+CD) -> 46 (per-quartile) -> 35 (matrice+Pareto)
       -> 47 (Cliff δ) -> 49 (RQ1) -> paper_heatmap -> paper_extra_figs -> paper_censored_fig
  5. stampa un confronto BEFORE/AFTER (best globale, celle TFT, conteggio matrice).

Uso:
  python pipeline/integrate_tft_gpu.py            # integra + rigenera
  python pipeline/integrate_tft_gpu.py --dry-run  # mostra solo cosa farebbe
  python pipeline/integrate_tft_gpu.py --no-rerun # solo swap, niente ricalcolo
  python pipeline/integrate_tft_gpu.py --restore  # ripristina lo stato CPU dal backup
"""
import os, sys, glob, json, shutil, subprocess, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)

RES = os.path.join(os.path.dirname(__file__), 'results')
PIPE = os.path.dirname(__file__)
BACKUP = os.path.join(RES, '_tft_cpu_backup')
MANIFEST = os.path.join(BACKUP, 'manifest.json')
PY = sys.executable

IMPUTERS = ['no_imp', 'media_glob', 'media_cond', 'mediana_glob', 'mediana_cond',
            'forward_fill', 'seasonal_naive', 'linear_interp', 'lgb',
            'dlinear', 'saits', 'itransformer', 'timesnet', 'imputeformer']

# catena downstream: (script, is_core). core=True -> stop se fallisce.
CHAIN = [
    ('45_friedman_nemenyi.py', True),
    ('46_friedman_nemenyi_per_quartile.py', False),
    ('35_pareto_analysis_hpo.py', True),
    ('47_friedman_cliff_delta.py', False),
    ('49_rq1_from_friedman.py', False),
    ('paper_heatmap.py', True),
    ('paper_extra_figs.py', False),
    ('paper_censored_fig.py', False),
]

def gpu_path(imp):  return os.path.join(RES, f'{imp}__tft_gpu_test_per_series.parquet')
def hpo_path(imp):  return os.path.join(RES, f'{imp}__tft_hpo_test_per_series.parquet')

def snapshot_tft(tag):
    """Legge la matrice corrente e restituisce un dict con best globale + celle TFT."""
    mp = os.path.join(RES, 'hpo_matrix_pareto.parquet')
    if not os.path.exists(mp):
        return None
    m = pd.read_parquet(mp)
    best = m.sort_values('wape_h_med').iloc[0]
    tft = m[m.forecaster == 'tft'][['imputer', 'wape_h_med', 'abs_wpe_med', 'pareto']].copy()
    return {'tag': tag, 'n_cells': len(m), 'best_cell': best['cell'],
            'best_wape': float(best['wape_h_med']),
            'tft': tft.sort_values('wape_h_med').reset_index(drop=True)}

# ---------------------------------------------------------------- restore mode
if '--restore' in sys.argv:
    if not os.path.exists(MANIFEST):
        sys.exit('Nessun manifest di backup: niente da ripristinare.')
    man = json.load(open(MANIFEST))
    for imp in man['overwritten']:
        src = os.path.join(BACKUP, f'{imp}__tft_hpo_test_per_series.parquet')
        shutil.copy2(src, hpo_path(imp))
        print(f'  ripristinato {imp}__tft_hpo (da backup)')
    for imp in man['added']:
        if os.path.exists(hpo_path(imp)):
            os.remove(hpo_path(imp))
            print(f'  rimosso {imp}__tft_hpo (era stato aggiunto)')
    print('Ripristino completato. Ri-esegui la catena a mano se necessario.')
    sys.exit(0)

DRY = '--dry-run' in sys.argv
NO_RERUN = '--no-rerun' in sys.argv

# ---------------------------------------------------------------- 1. validazione
print('=' * 72)
print('  INTEGRAZIONE TFT-GPU')
print('=' * 72)
present = [imp for imp in IMPUTERS if os.path.exists(gpu_path(imp))]
missing = [imp for imp in IMPUTERS if not os.path.exists(gpu_path(imp))]
print(f'\n1. File GPU trovati: {len(present)}/{len(IMPUTERS)}')
if missing:
    print(f'   ⚠ MANCANTI ({len(missing)}): {missing}')
    print('   Queste celle NON verranno sostituite (resta la versione CPU, se esiste).')
if not present:
    sys.exit('Nessun file *__tft_gpu_test_per_series.parquet trovato. Interrompo.')

before = snapshot_tft('BEFORE (CPU)')

# ---------------------------------------------------------------- 2. backup + swap
print(f'\n2. Backup + swap ({"DRY-RUN" if DRY else "esecuzione"})...')
os.makedirs(BACKUP, exist_ok=True)
overwritten, added = [], []
for imp in present:
    tgt = hpo_path(imp)
    if os.path.exists(tgt):
        overwritten.append(imp)
        if not DRY:
            shutil.copy2(tgt, os.path.join(BACKUP, os.path.basename(tgt)))
    else:
        added.append(imp)          # es. imputeformer: nuova cella, chiude il buco
    if not DRY:
        shutil.copy2(gpu_path(imp), tgt)
    print(f'   {imp:15s} -> {"OVERWRITE" if imp in overwritten else "ADD (nuova cella)"}')
if not DRY:
    json.dump({'overwritten': overwritten, 'added': added}, open(MANIFEST, 'w'), indent=2)
print(f'   overwritten={len(overwritten)}  added={len(added)}  (backup in {BACKUP})')

if DRY:
    print('\nDRY-RUN: nessuna modifica scritta. Rilancia senza --dry-run per applicare.')
    sys.exit(0)

# ---------------------------------------------------------------- 3. rerun chain
if NO_RERUN:
    print('\n--no-rerun: swap fatto, salto il ricalcolo.')
    sys.exit(0)

print('\n3. Ri-esecuzione catena statistica/figure...')
for script, is_core in CHAIN:
    path = os.path.join(PIPE, script)
    if not os.path.exists(path):
        print(f'   ⚠ {script} non trovato, salto.'); continue
    print(f'\n   >>> {script} ...')
    r = subprocess.run([PY, path], capture_output=True, text=True)
    tail = '\n'.join(r.stdout.strip().splitlines()[-6:])
    if r.returncode == 0:
        print(f'   OK\n{tail}')
    else:
        errtail = '\n'.join((r.stdout + r.stderr).strip().splitlines()[-15:])
        print(f'   FALLITO (exit {r.returncode}):\n{errtail}')
        if is_core:
            sys.exit(f'\nStep CORE {script} fallito: interrompo. '
                     f'Usa --restore per tornare allo stato CPU.')

# ---------------------------------------------------------------- 4. confronto
print('\n' + '=' * 72)
print('  CONFRONTO BEFORE / AFTER')
print('=' * 72)
after = snapshot_tft('AFTER (GPU)')
if before is not None:
    print(f'\nMatrice: {before["n_cells"]} celle -> {after["n_cells"]} celle '
          f'({"+"+str(after["n_cells"]-before["n_cells"]) if after["n_cells"]>=before["n_cells"] else after["n_cells"]-before["n_cells"]})')
    print(f'Best globale (WAPE med): {before["best_cell"]} ({before["best_wape"]:.4f}) '
          f'-> {after["best_cell"]} ({after["best_wape"]:.4f})')
else:
    print(f'\nMatrice: {after["n_cells"]} celle | Best globale: {after["best_cell"]} ({after["best_wape"]:.4f})')

print('\nCelle TFT (WAPE_h_med) BEFORE vs AFTER:')
b_tft = before['tft'].set_index('imputer')['wape_h_med'].to_dict() if before is not None else {}
a_tft = after['tft'].set_index('imputer')
for imp in a_tft['imputer'] if 'imputer' in a_tft else a_tft.index:
    new = a_tft.loc[imp, 'wape_h_med'] if imp in a_tft.index else float('nan')
    old = b_tft.get(imp, float('nan'))
    par = '  [Pareto]' if (imp in a_tft.index and bool(a_tft.loc[imp, 'pareto'])) else ''
    delta = f'{new-old:+.4f}' if not np.isnan(old) else '  (nuova)'
    print(f'   {imp:15s} {old if not np.isnan(old) else float("nan"):.4f} -> {new:.4f}  ({delta}){par}'
          if not np.isnan(old) else
          f'   {imp:15s}   ---   -> {new:.4f}  (nuova cella){par}')

# best TFT cell + posizione in classifica Friedman, se disponibile
fr_path = os.path.join(RES, 'friedman_nemenyi_ranks.parquet')
if os.path.exists(fr_path):
    fr = pd.read_parquet(fr_path).reset_index(drop=True)
    fr['rank_pos'] = np.arange(1, len(fr) + 1)
    tft_fr = fr[fr.cell.str.endswith('__tft')].sort_values('rank_pos')
    print(f'\nFriedman: {len(fr)} celle | miglior cella TFT = '
          f'{tft_fr.iloc[0]["cell"]} (posizione {int(tft_fr.iloc[0]["rank_pos"])}/{len(fr)})'
          if len(tft_fr) else '\nFriedman: nessuna cella TFT trovata nel ranking')
    equiv = fr[fr.cd_indistinguishable]['cell'].tolist() if 'cd_indistinguishable' in fr else []
    print(f'Equivalence set (Nemenyi CD): {len(equiv)} celle -> {equiv}')

print('\nDONE. Figure aggiornate: fig_cd_diagram, fig_pareto_hpo, fig_results_heatmap, '
      'fig_rq4_crossover, fig_censored_direct.')
print('Rivedi i numeri poi aggiorna il testo del paper (framing TFT, conteggio matrice, caption).')
