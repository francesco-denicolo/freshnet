"""Referee point #5 --- re-score the central finding on "safe" in-stock test hours.

Assumption 1 (an in-stock hour reflects uncensored demand) is most fragile for hours
that themselves sold out later in the day: an in-stock-labelled hour may have depleted
by its end, so the "uncensored ground truth" is itself mildly censored, correlated with
demand. We therefore re-score on safe in-stock hours only:
  safe_A: in-stock hours on days with zero stock-out;
  safe_B: in-stock hours at least k=2 hours before the first stock-out of the day.
If the imputer stays negligible and the forecaster ordering holds there, the limitation
is closed. Consumes safehours_{cell}_test_per_series.parquet from the NV_OVERNIGHT run.
Run: safe_hours_aggregate.py
"""
import os, glob, functools
import numpy as np, pandas as pd
print = functools.partial(print, flush=True)
RES = os.path.join(os.path.dirname(__file__), 'results')
MIN_INSTOCK, MIN_SAFE = 34, 17   # per-series inclusion thresholds (hours)

files = sorted(glob.glob(f'{RES}/safehours_*_test_per_series.parquet'))
print(f'{len(files)} cells with safe-hours re-scoring\n')

rows = []
for f in files:
    cell = os.path.basename(f)[len('safehours_'):-len('_test_per_series.parquet')]
    d = pd.read_parquet(f)
    imp, fc = cell.replace('_hpo', '').split('__', 1)

    def med(wcol, ncol, thr):
        s = d[d[ncol] >= thr][wcol].dropna()
        return (float(s.median()), int(len(s)))
    wi, ni = med('wape_instock', 'n_instock', MIN_INSTOCK)
    wa, na = med('wape_safeA', 'n_safeA', MIN_SAFE)
    wb, nb = med('wape_safeB', 'n_safeB', MIN_SAFE)
    pi, _ = med('wpe_instock', 'n_instock', MIN_INSTOCK)
    pa, _ = med('wpe_safeA', 'n_safeA', MIN_SAFE)
    pb, _ = med('wpe_safeB', 'n_safeB', MIN_SAFE)
    rows.append({'cell': cell, 'imputer': imp, 'forecaster': fc,
                 'wape_instock': wi, 'wape_safeA': wa, 'wape_safeB': wb,
                 'wpe_instock': pi, 'wpe_safeA': pa, 'wpe_safeB': pb,
                 'n_series_instock': ni, 'n_series_safeA': na, 'n_series_safeB': nb})

df = pd.DataFrame(rows).sort_values('wape_instock').reset_index(drop=True)
out = f'{RES}/safehours_summary.parquet'
df.to_parquet(out, index=False)
pd.set_option('display.width', 200)
print(df[['cell', 'wape_instock', 'wape_safeA', 'wape_safeB', 'wpe_instock', 'wpe_safeA', 'wpe_safeB']].to_string(index=False))

print('\n=== does the imputer stay negligible on safe hours? (within-forecaster WAPE spread) ===')
for fc, g in df.groupby('forecaster'):
    print(f'  {fc}:  spread(instock)={g.wape_instock.max()-g.wape_instock.min():.4f}  '
          f'spread(safeA)={g.wape_safeA.max()-g.wape_safeA.min():.4f}  '
          f'spread(safeB)={g.wape_safeB.max()-g.wape_safeB.min():.4f}')

print('\n=== best cell (lowest in-stock WAPE) across scoring regions ===')
b = df.iloc[0]
print(f'  {b.cell}: WAPE instock={b.wape_instock:.4f}  safeA={b.wape_safeA:.4f}  safeB={b.wape_safeB:.4f}')
print(f'\nsaved {out}')
print('Interpretation: if the within-forecaster spread stays small and the ordering holds on '
      'safe hours, the mild endogenous censoring of in-stock labels does not drive the conclusions.')
