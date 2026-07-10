#!/bin/bash
# Combined overnight batch. Launch under caffeinate so the PC stays awake:
#   caffeinate -i bash pipeline/run_overnight_batch.sh > pipeline/overnight.log 2>&1 &
# Each step skips outputs that already exist, so the batch is safely resumable.
set -e
cd "$(dirname "$0")/.."
PY=freshnet/bin/python
export HPO_VARIANT=1

echo "######## OVERNIGHT BATCH START $(date) ########"

# ---- MAJOR 1: censored-aware DIRECT forecasters (no imputer, quantile/pinball sweep) ----
export TAUS="0.5,0.6,0.7,0.8,0.9"
echo "==== [1/4] censored-aware LGB (taus=$TAUS) $(date) ===="
$PY pipeline/censored_aware_lgb.py
echo "==== [2/4] censored-aware MLP (taus=$TAUS) $(date) ===="
$PY pipeline/censored_aware_mlp.py

# ---- Seed sweep: 20 seeds for both lag-based forecasters on the itransformer cell ----
export SEEDS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"
echo "==== [3/4] seed sweep LGB (20 seeds) $(date) ===="
$PY pipeline/seedrun_lgb_multi.py itransformer
echo "==== [4/4] seed sweep MLP (20 seeds) $(date) ===="
$PY pipeline/seedrun_mlp_multi.py itransformer

echo "######## OVERNIGHT BATCH DONE $(date) ########"
