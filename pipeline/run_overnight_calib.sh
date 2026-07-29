#!/usr/bin/env bash
# Consolidated overnight re-run for referee points #4 (newsvendor critical-fractile
# calibration) and #5 (safe-in-stock-hours re-score).
#
# Re-trains the 26 lag-based forecaster cells (13 imputers x {MLP-M5, LGB-M5}) with
# NV_OVERNIGHT=1, which makes each run additionally dump:
#   - newsvendor_qval_{cell}.parquet            (validation daily orders, for #4)
#   - safehours_{cell}_test_per_series.parquet  (safe-hours WAPE/WPE, for #5)
# then runs the two downstream analyses. Resumable: cells whose three outputs already
# exist are skipped, so re-launching after an interruption continues where it stopped.
#
# Usage:  nohup bash pipeline/run_overnight_calib.sh > pipeline/results/overnight_calib.out 2>&1 &
#         tail -f pipeline/results/overnight_calib_run.log
set -u
cd "$(dirname "$0")/.."
PY=freshnet/bin/python
export HPO_VARIANT=1 NV_OVERNIGHT=1
LOG=pipeline/results/overnight_calib_run.log
echo "START $(date)  |  HPO_VARIANT=$HPO_VARIANT NV_OVERNIGHT=$NV_OVERNIGHT" | tee -a "$LOG"

echo "=== 0. validation reference (days 84-90) ===" | tee -a "$LOG"
$PY pipeline/build_nv_ref_val.py 2>&1 | tee -a "$LOG"

# imputer list = the lag cells that already exist in the matrix (13)
IMPUTERS=$(ls pipeline/results/newsvendor_q_*__mlp_m5lags_hpo.parquet 2>/dev/null \
           | sed 's#.*newsvendor_q_##; s#__mlp_m5lags_hpo.parquet##' | sort)
echo "Imputers ($(echo $IMPUTERS | wc -w | tr -d ' ')): $IMPUTERS" | tee -a "$LOG"

for imp in $IMPUTERS; do
  echo "=== [$(date +%H:%M)] MLP-M5 x $imp ===" | tee -a "$LOG"
  $PY pipeline/nv_mlp.py "$imp" 2>&1 | tee -a "$LOG"
  echo "=== [$(date +%H:%M)] LGB-M5 x $imp ===" | tee -a "$LOG"
  $PY pipeline/nv_lgb.py "$imp" 2>&1 | tee -a "$LOG"
done

echo "=== #4 critical-fractile calibration ===" | tee -a "$LOG"
$PY pipeline/nv_calibrate.py 2>&1 | tee -a "$LOG"
echo "=== #5 safe-hours re-score ===" | tee -a "$LOG"
$PY pipeline/safe_hours_aggregate.py 2>&1 | tee -a "$LOG"
echo "DONE $(date)" | tee -a "$LOG"
