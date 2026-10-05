#!/usr/bin/env bash
# Second referee report, round 2 — points A and C (~6 h).
#   A: direct forecaster with no imputer, pinball tau in {0.5..0.9} x {MLP-M5, LGB-M5}, saving the
#      hourly forecasts on validation (days 84-90) and test (91-97): tau is then selected per cost ratio
#      r on VALIDATION order cost and reported at test.                                   (10 cells)
#   C: the same no-imputer cell at tau = 0.7 under seeds 1-5 (seed 42 comes from A), for a seed band
#      of the order cost.                                                                 (10 cells)
# Outputs are tagged _nvsel so the files cited in the paper are never touched.
# Resumable: a finished cell is skipped; every file is written atomically (temp file + rename), so an
# interruption never leaves a truncated result behind. Sleep is blocked by caffeinate: keep the Mac
# on AC power with the lid open.
#
# Usage:  nohup bash pipeline/run_overnight_nv_tau.sh > pipeline/results/overnight_nv_tau.out 2>&1 &
#         tail -f pipeline/results/_log_nv_tau.txt
set -u
cd "$(dirname "$0")/.."
PY=freshnet/bin/python
LOG=pipeline/results/_log_nv_tau.txt
caffeinate -is -w $$ &
export HPO_VARIANT=1 SAVE_PREDS=1 IMPUTER=no_imp RUN_TAG=_nvsel
unset SMOKE SEED
echo "START $(date)" | tee -a "$LOG"

# ---- A: tau selection on validation cost (seed 42); tau = 0.7 first, it is also the seed-42 member of C
for tau in 0.7 0.8 0.6 0.5 0.9; do
  for fam in mlp lgb; do
    echo "=== [$(date '+%d %H:%M')] A  tau=$tau  ${fam}-M5 (val+test orders) ===" | tee -a "$LOG"
    SAVE_VAL_PREDS=1 TAUS=$tau $PY pipeline/censored_aware_${fam}.py 2>&1 | grep -vE '^\s+\.\.\.' | tee -a "$LOG"
  done
done
echo "=== [$(date '+%d %H:%M')] analysis after A ===" | tee -a "$LOG"
$PY pipeline/referee2_nv_tau_select.py 2>&1 | tee -a "$LOG"

# ---- C: seed band of the order cost at tau = 0.7
for seed in 1 2 3 4 5; do
  for fam in mlp lgb; do
    echo "=== [$(date '+%d %H:%M')] C  tau=0.7  ${fam}-M5  seed=$seed ===" | tee -a "$LOG"
    SEED=$seed TAUS=0.7 $PY pipeline/censored_aware_${fam}.py 2>&1 | grep -vE '^\s+\.\.\.' | tee -a "$LOG"
  done
done
echo "=== [$(date '+%d %H:%M')] final analysis ===" | tee -a "$LOG"
$PY pipeline/referee2_nv_tau_select.py 2>&1 | tee -a "$LOG"
echo "DONE $(date)" | tee -a "$LOG"
