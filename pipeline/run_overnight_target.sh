#!/usr/bin/env bash
# Internal review (Oct 2026), point (b): LGB-M5 and MLP-M5 production cells (MAE, tuned configs)
# re-trained with the COMPLETED series as training target, for the 13 imputers (26 cells, ~10 h).
# No imputation needs no re-run (completed == observed). Outputs tagged _tgtcomp; the paper's cells are
# never touched. Resumable (finished cells are skipped; results written atomically). Keep the Mac on AC.
# Usage:  nohup bash pipeline/run_overnight_target.sh > pipeline/results/overnight_target.out 2>&1 &
#         tail -f pipeline/results/_log_target.txt
set -u
cd "$(dirname "$0")/.."
PY=freshnet/bin/python
LOG=pipeline/results/_log_target.txt
caffeinate -is -w $$ &
export HPO_VARIANT=1 TARGET_COMPLETED=1 CELL_SUFFIX=_tgtcomp
unset SMOKE
IMPS="itransformer mediana_glob media_glob saits dlinear forward_fill timesnet lgb imputeformer seasonal_naive linear_interp media_cond mediana_cond"
echo "START $(date)" | tee -a "$LOG"
for imp in $IMPS; do
  echo "=== [$(date '+%d %H:%M')] MLP-M5 x $imp (target = completed) ===" | tee -a "$LOG"
  $PY pipeline/08_fase_b2_forecast_mlp.py "$imp" 2>&1 | grep -vE '^\s+\.\.\.|Epoch' | tee -a "$LOG"
  echo "=== [$(date '+%d %H:%M')] LGB-M5 x $imp (target = completed) ===" | tee -a "$LOG"
  $PY pipeline/07_fase_b2_forecast_lgb.py "$imp" 2>&1 | grep -vE '^\s+\.\.\.' | tee -a "$LOG"
  $PY pipeline/referee3_target_completed.py 2>&1 | tail -4 | tee -a "$LOG"
done
echo "DONE $(date)" | tee -a "$LOG"
