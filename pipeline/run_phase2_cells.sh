#!/bin/bash
# FASE 2: Run 6 forecaster cells on new imputers (CSDI + ImputeFormer)

cd /Users/utente/Desktop/FreshNetRetail
PYBIN="freshnet/bin/python"
export HPO_VARIANT=1
export PYTHONUNBUFFERED=1

IMPUTERS="csdi imputeformer"
T0=$(date +%s)
echo "============================================================"
echo "[$(date +%H:%M:%S)] FASE 2: 6 cells (LGB+MLP+TFT × csdi/imputeformer)"
echo "============================================================"

for imp in $IMPUTERS; do
    t1=$(date +%s); echo ""; echo "[$(date +%H:%M:%S)] LGB_M5 × $imp"
    $PYBIN pipeline/07_fase_b2_forecast_lgb.py "$imp"
    t2=$(date +%s); echo "[$(date +%H:%M:%S)] DONE: LGB_M5 × $imp ($((($t2-$t1)/60))min)"
done

for imp in $IMPUTERS; do
    t1=$(date +%s); echo ""; echo "[$(date +%H:%M:%S)] MLP_M5 × $imp"
    $PYBIN pipeline/08_fase_b2_forecast_mlp.py "$imp"
    t2=$(date +%s); echo "[$(date +%H:%M:%S)] DONE: MLP_M5 × $imp ($((($t2-$t1)/60))min)"
done

for imp in $IMPUTERS; do
    t1=$(date +%s); echo ""; echo "[$(date +%H:%M:%S)] TFT × $imp"
    $PYBIN pipeline/25_tft_full_training.py "$imp"
    t2=$(date +%s); echo "[$(date +%H:%M:%S)] DONE: TFT × $imp ($((($t2-$t1)/60))min)"
done

T_END=$(date +%s)
echo ""
echo "============================================================"
echo "FASE 2 DONE in $(( (T_END - T0) / 60 )) min ($(( (T_END - T0) / 3600 )) h)"
echo "============================================================"
