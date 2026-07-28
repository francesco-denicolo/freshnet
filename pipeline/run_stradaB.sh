#!/bin/bash
# Strada B (review risk #2, hard 50K W): re-train the 13 imputer MLP-M5 cells with the
# PRODUCTION pipeline (08_fase_b2_forecast_mlp.py) using the itransformer-tuned config
# ([256,128], from hpo_mlp_itransformer_best.json), then compute Kendall's W across the
# imputers. If W stays negligible, the tuning basis does not create imputer co-adaptation
# -- with the paper's own production code, at full scale. ~4-6 h.
# Monitor: tail -f pipeline/results/_log_stradaB.txt
cd /Users/utente/Desktop/FreshNetRetail
PY=freshnet/bin/python
LOG=pipeline/results/_log_stradaB.txt
IMPS="itransformer mediana_glob media_glob saits dlinear forward_fill timesnet lgb imputeformer seasonal_naive linear_interp media_cond mediana_cond"
{
  for imp in $IMPS; do
    echo "=== retrain $imp (itransformer config) ==="
    HPO_VARIANT=1 MLP_CONFIG=hpo_mlp_itransformer_best.json CELL_SUFFIX=_itconfig $PY pipeline/08_fase_b2_forecast_mlp.py "$imp"
  done
  echo "=== Kendall W across the 13 itconfig cells ==="
  $PY - <<'PYEOF'
import pandas as pd, glob, numpy as np, os
fs=sorted(glob.glob('pipeline/results/*__mlp_m5lags_hpo_itconfig_test_per_series.parquet'))
cols={}
for f in fs:
    imp=os.path.basename(f).split('__')[0]
    cols[imp]=pd.read_parquet(f).set_index(['store_id','product_id'])['hourly_wape']
M=pd.DataFrame(cols).dropna()
ranks=M.rank(axis=1); n,k=M.shape
Rsum=ranks.sum(0)
W=12*((Rsum-n*(k+1)/2)**2).sum()/(n**2*k*(k**2-1))
print(f'STRADA B RESULT: Kendall W (itransformer-tuned [256,128] config, {k} imputers, {n} series) = {W:.4f}')
print('  reference: paper censored-tuned MLP-M5 W = 0.03 ; 20K ad-hoc = 0.07')
print('  per-imputer median WAPE:'); print(M.median(0).round(4).sort_values().to_string())
PYEOF
  echo "=== STRADA B DONE ==="
} > "$LOG" 2>&1 &
echo "launched Strada B, PID $!"
echo "monitor: tail -f $LOG"
