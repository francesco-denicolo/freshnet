#!/bin/bash
# Capacity-sweep robustness check: re-train iTransformer + TimesNet at increased
# capacity, then run both lag-based forecasters (MLP-M5, LGB-M5) on each, with the
# same HPO forecaster configs as the main-benchmark cells. Outputs are *_hicap.
#
# Cells produced (4):
#   itransformer_hicap__mlp_m5lags_hpo, itransformer_hicap__lgb_m5lags_hpo
#   timesnet_hicap__mlp_m5lags_hpo,     timesnet_hicap__lgb_m5lags_hpo
#
# Run from the repo root:  bash pipeline/run_cap_sweep.sh
set -e
cd "$(dirname "$0")/.."
PY=freshnet/bin/python

for IMP in itransformer timesnet; do
  echo "######## CAPACITY SWEEP: $IMP ########"
  $PY pipeline/cap_sweep_imputer.py "$IMP"
  echo "---- forecaster: MLP-M5 on ${IMP}_hicap ----"
  HPO_VARIANT=1 $PY pipeline/08_fase_b2_forecast_mlp.py "${IMP}_hicap"
  echo "---- forecaster: LGB-M5 on ${IMP}_hicap ----"
  HPO_VARIANT=1 $PY pipeline/07_fase_b2_forecast_lgb.py "${IMP}_hicap"
done

echo "######## DONE — 4 hi-cap cells written to pipeline/results/ ########"
