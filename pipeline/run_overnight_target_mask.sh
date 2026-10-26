#!/usr/bin/env bash
# Internal review (Oct 2026): the censored target of the lag-based forecasters (~16 h, resumable).
#  (ii-a) pinball loss MASKED on stock-out hours, no imputation, tau 0.5-0.8, val+test orders   (8 cells)
#  (ii-b) masked loss for Median global / iTransformer / Linear interp. at tau 0.5 and at tau*,
#         tau* selected on validation WAPE from (ii-a)                                        (6-12 cells)
#  (i)    production MAE cells with the COMPLETED series as training target, 13 imputers      (26 cells)
# Outputs are tagged (_mask, _tgtcomp); the paper's cells are never touched. Finished cells are skipped,
# results are written atomically. Sleep is blocked by caffeinate: keep the Mac on AC, lid open.
# Usage:  nohup bash pipeline/run_overnight_target_mask.sh > pipeline/results/overnight_target_mask.out 2>&1 &
#         tail -f pipeline/results/_log_target_mask.txt
set -u
cd "$(dirname "$0")/.."
PY=freshnet/bin/python
LOG=pipeline/results/_log_target_mask.txt
caffeinate -is -w $$ &
export HPO_VARIANT=1
unset SMOKE SEED TARGET_COMPLETED CELL_SUFFIX
echo "START $(date)" | tee -a "$LOG"

# ---- (ii-a) masked loss, no imputation, tau sweep with validation orders
for tau in 0.5 0.6 0.7 0.8; do
  for fam in mlp lgb; do
    echo "=== [$(date '+%d %H:%M')] (ii-a) masked  tau=$tau  ${fam}-M5  no imputation ===" | tee -a "$LOG"
    MASK_STOCKOUT=1 IMPUTER=no_imp RUN_TAG=_mask SAVE_PREDS=1 SAVE_VAL_PREDS=1 TAUS=$tau \
      $PY pipeline/censored_aware_${fam}.py 2>&1 | grep -vE '^\s+\.\.\.' | tee -a "$LOG"
  done
done
$PY pipeline/referee3_mask.py 2>&1 | tee -a "$LOG"

# ---- (ii-b) masked loss for three imputers at tau = 0.5 and tau* (per family, from the json)
for fam in mlp lgb; do
  TSTAR=$($PY -c "import json;print(json.load(open('pipeline/results/referee3_mask_tau_star.json')).get('$fam','0.5'))" 2>/dev/null || echo 0.5)
  for tau in $(echo "0.5 $TSTAR" | tr ' ' '\n' | sort -u); do
    for imp in mediana_glob itransformer linear_interp; do
      echo "=== [$(date '+%d %H:%M')] (ii-b) masked  tau=$tau  ${fam}-M5 x $imp ===" | tee -a "$LOG"
      MASK_STOCKOUT=1 IMPUTER=$imp RUN_TAG=_mask SAVE_PREDS=1 TAUS=$tau \
        $PY pipeline/censored_aware_${fam}.py 2>&1 | grep -vE '^\s+\.\.\.' | tee -a "$LOG"
    done
  done
done
$PY pipeline/referee3_mask.py 2>&1 | tee -a "$LOG"

# ---- (i) completed series as training target, 13 imputers (production MAE cells)
IMPS="itransformer mediana_glob media_glob saits dlinear forward_fill timesnet lgb imputeformer seasonal_naive linear_interp media_cond mediana_cond"
for imp in $IMPS; do
  echo "=== [$(date '+%d %H:%M')] (i) target=completed  MLP-M5 x $imp ===" | tee -a "$LOG"
  TARGET_COMPLETED=1 CELL_SUFFIX=_tgtcomp $PY pipeline/08_fase_b2_forecast_mlp.py "$imp" 2>&1 | grep -vE '^\s+\.\.\.|Epoch' | tee -a "$LOG"
  echo "=== [$(date '+%d %H:%M')] (i) target=completed  LGB-M5 x $imp ===" | tee -a "$LOG"
  TARGET_COMPLETED=1 CELL_SUFFIX=_tgtcomp $PY pipeline/07_fase_b2_forecast_lgb.py "$imp" 2>&1 | grep -vE '^\s+\.\.\.' | tee -a "$LOG"
done
$PY pipeline/referee3_target_completed.py 2>&1 | tee -a "$LOG"
echo "DONE $(date)" | tee -a "$LOG"
