#!/usr/bin/env bash
# Second referee report, Major #2 follow-up: the MAE-equivalent block (pinball tau = 0.5 = MAE/2,
# same minimiser) of the imputer grid, run with the SAME pipeline as the tau = 0.6/0.7 grid
# (censored_aware_*.py with IMPUTER set) so that RMSSE / scaled pinball can be compared across
# MAE -> 0.6 -> 0.7. {no_imp + 13 imputers} x {MLP-M5, LGB-M5} = 28 cells, ~7-8 h.
# Hourly test predictions -> results/preds/*.npz. Then re-runs the W and RMSSE/SPL analyses.
# Resumable (finished cells are skipped). Sleep blocked by caffeinate: keep the Mac on AC, lid open.
#
# Usage:  nohup bash pipeline/run_overnight_mae_grid.sh > pipeline/results/overnight_mae_grid.out 2>&1 &
#         tail -f pipeline/results/_log_mae_grid.txt
set -u
cd "$(dirname "$0")/.."
PY=freshnet/bin/python
LOG=pipeline/results/_log_mae_grid.txt
caffeinate -is -w $$ &
export HPO_VARIANT=1 SAVE_PREDS=1
unset SMOKE
IMPS="no_imp itransformer mediana_glob media_glob saits dlinear forward_fill timesnet lgb imputeformer seasonal_naive linear_interp media_cond mediana_cond"
echo "START $(date)" | tee -a "$LOG"
for imp in $IMPS; do
  echo "=== [$(date '+%d %H:%M')] tau=0.5  MLP-M5 x $imp ===" | tee -a "$LOG"
  IMPUTER=$imp TAUS=0.5 $PY pipeline/censored_aware_mlp.py 2>&1 | grep -vE '^\s+\.\.\.' | tee -a "$LOG"
  echo "=== [$(date '+%d %H:%M')] tau=0.5  LGB-M5 x $imp ===" | tee -a "$LOG"
  IMPUTER=$imp TAUS=0.5 $PY pipeline/censored_aware_lgb.py 2>&1 | grep -vE '^\s+\.\.\.' | tee -a "$LOG"
done
echo "=== [$(date '+%d %H:%M')] W analysis (WAPE) ===" | tee -a "$LOG"
$PY pipeline/referee2_tau_grid_analysis.py 2>&1 | tee -a "$LOG"
echo "=== [$(date '+%d %H:%M')] RMSSE / scaled pinball ===" | tee -a "$LOG"
$PY pipeline/referee2_rmsse_spl.py 2>&1 | tee -a "$LOG"
echo "DONE $(date)" | tee -a "$LOG"
