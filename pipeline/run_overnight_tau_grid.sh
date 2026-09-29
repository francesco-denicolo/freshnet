#!/usr/bin/env bash
# Second referee report, Major #1: imputer x pinball-tau grid for the two lag-based forecasters.
#
#   Pass 1 (decisive, ~9-10 h): tau = 0.7, {no_imp + 13 imputers} x {MLP-M5, LGB-M5} = 28 cells
#   Ablation (~25 min):         best cell (iTransformer/MLP-M5, MAE) without the realised weather covariates
#   Pass 2 (~8 h):              tau = 0.6, same 28 cells
#   After each pass: referee2_tau_grid_analysis.py (Kendall W across imputers, as RQ1)
#
# Every cell also saves its hourly test predictions (results/preds/*.npz) for RMSSE / scaled pinball.
# Resumable: finished cells are skipped, so after a crash/reboot just launch it again.
# Sleep is blocked with caffeinate for the whole run (keep the Mac on AC power).
#
# Usage:  nohup bash pipeline/run_overnight_tau_grid.sh > pipeline/results/overnight_tau_grid.out 2>&1 &
#         tail -f pipeline/results/_log_tau_grid.txt
set -u
cd "$(dirname "$0")/.."
PY=freshnet/bin/python
LOG=pipeline/results/_log_tau_grid.txt
caffeinate -is -w $$ &
export HPO_VARIANT=1 SAVE_PREDS=1
unset SMOKE
IMPS="no_imp itransformer mediana_glob media_glob saits dlinear forward_fill timesnet lgb imputeformer seasonal_naive linear_interp media_cond mediana_cond"
echo "START $(date)" | tee -a "$LOG"

run_pass () {
  local tau=$1
  for imp in $IMPS; do
    echo "=== [$(date '+%d %H:%M')] tau=$tau  MLP-M5 x $imp ===" | tee -a "$LOG"
    IMPUTER=$imp TAUS=$tau $PY pipeline/censored_aware_mlp.py 2>&1 | grep -vE '^\s+\.\.\.' | tee -a "$LOG"
    echo "=== [$(date '+%d %H:%M')] tau=$tau  LGB-M5 x $imp ===" | tee -a "$LOG"
    IMPUTER=$imp TAUS=$tau $PY pipeline/censored_aware_lgb.py 2>&1 | grep -vE '^\s+\.\.\.' | tee -a "$LOG"
  done
  echo "=== [$(date '+%d %H:%M')] analysis after tau=$tau ===" | tee -a "$LOG"
  $PY pipeline/referee2_tau_grid_analysis.py 2>&1 | tee -a "$LOG"
}

run_pass 0.7

echo "=== [$(date '+%d %H:%M')] weather ablation: iTransformer x MLP-M5 (MAE) without weather ===" | tee -a "$LOG"
NO_WEATHER=1 CELL_SUFFIX=_noweather $PY pipeline/08_fase_b2_forecast_mlp.py itransformer 2>&1 | tail -15 | tee -a "$LOG"

run_pass 0.6
echo "DONE $(date)" | tee -a "$LOG"
