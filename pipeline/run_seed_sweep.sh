#!/bin/bash
# Overnight seed sweep: 20 seeds for LGB-M5 and MLP-M5 on the itransformer cell.
# Build once, loop seeds (skips seeds whose parquet already exists).
# Launch under caffeinate so the PC stays awake:  caffeinate -i bash pipeline/run_seed_sweep.sh
set -e
cd "$(dirname "$0")/.."
PY=freshnet/bin/python
export SEEDS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"
export HPO_VARIANT=1

echo "######## SEED SWEEP START $(date) | seeds=$SEEDS ########"
echo "---- LGB-M5 ----"
$PY pipeline/seedrun_lgb_multi.py itransformer
echo "---- MLP-M5 ----"
$PY pipeline/seedrun_mlp_multi.py itransformer
echo "######## SEED SWEEP DONE $(date) ########"
