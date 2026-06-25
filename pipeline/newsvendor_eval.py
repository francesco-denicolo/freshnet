"""
Order-level / newsvendor evaluation (referee major pt 1 + pt 8).

Goal: re-rank the key cells under an OPERATIONAL objective that (a) weights the
stock-out portion of demand and (b) penalises over- and under-ordering
asymmetrically, to test whether the "imputer is free" / WAPE ranking survives
when the objective is what the recovery step is actually meant to improve.

Pipeline:
  1. Reference completed daily demand y*(series, day) on the TEST horizon:
        y*(s,d) = sum over operational hours of
                    observed sales        if in stock
                    REFERENCE-imputer rec. if stock-out      (pseudo-truth, proxy)
     The reference imputer is fixed (e.g. the best, itransformer) for ALL cells,
     so it is not circular w.r.t. the forecaster being scored. Acknowledged as a
     synthetic proxy (no true stock-out ground truth exists).
  2. Per cell, daily order forecast q(s,d) = sum_h yhat_h  (exported separately;
     see gen_daily_preds.py -- NOT YET RUN; wire after the seed jobs finish).
  3. Newsvendor cost per (s,d) for an underage/overage ratio r = c_u/c_o:
        cost = c_o * max(q - y*, 0) + c_u * max(y* - q, 0)
     Reported as the per-series total cost, summarised by its cross-series median,
     for several r in {1, 2, 5} (perishables: lost sales typically dearer).
  4. Compare cells: does the cost ranking match the WAPE ranking? Does the
     imputer choice (within MLP-M5/LGB-M5) still look immaterial when the
     objective weights stock-out demand?

This file currently provides the cost machinery and the evaluation skeleton; the
daily-prediction export (step 2) and the reference-demand build (step 1) are
wired at run time.
"""
import os, functools, numpy as np, pandas as pd
print = functools.partial(print, flush=True)
RES = os.path.join(os.path.dirname(__file__), 'results')

COST_RATIOS = [1.0, 2.0, 5.0]   # c_u / c_o (underage = lost sales : overage = spoilage)

def newsvendor_cost(q, y, c_u, c_o):
    """Per-observation asymmetric order cost (q = order/forecast, y = realised demand)."""
    over = np.clip(q - y, 0, None)
    under = np.clip(y - q, 0, None)
    return c_o * over + c_u * under

def evaluate(daily_pred, daily_demand, group_keys):
    """daily_pred, daily_demand: 1-D arrays aligned over (series, day) rows;
    group_keys: array of series id per row. Returns per-ratio median per-series cost."""
    out = {}
    for r in COST_RATIOS:
        c = newsvendor_cost(daily_pred, daily_demand, c_u=r, c_o=1.0)
        per_series = pd.Series(c).groupby(group_keys).sum()
        out[f'r={r:g}'] = float(per_series.median())
    return out

if __name__ == '__main__':
    print('newsvendor_eval: cost machinery ready.')
    print('TODO at run time: (1) build reference completed daily demand on days 91-97;')
    print('                  (2) export daily forecasts per cell (gen_daily_preds.py);')
    print('                  (3) call evaluate() per cell and tabulate cost rankings.')
