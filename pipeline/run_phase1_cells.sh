#!/bin/bash
# FASE 1: Run 6 forecaster cells on new imputers (iTransformer + TimesNet)
# LGB_M5 × 2, MLP_M5 × 2, TFT × 2 with HPO HPs

cd /Users/utente/Desktop/FreshNetRetail
PYBIN="freshnet/bin/python"
export HPO_VARIANT=1
export PYTHONUNBUFFERED=1

IMPUTERS="itransformer timesnet"
T0=$(date +%s)
echo "============================================================"
echo "[$(date +%H:%M:%S)] FASE 1: 6 forecaster cells (LGB+MLP+TFT × itr/tsn)"
echo "============================================================"

# LGB_M5 cells
for imp in $IMPUTERS; do
    t1=$(date +%s)
    echo ""
    echo "[$(date +%H:%M:%S)] LGB_M5 × $imp"
    echo "============================================================"
    $PYBIN pipeline/07_fase_b2_forecast_lgb.py "$imp"
    t2=$(date +%s); echo "[$(date +%H:%M:%S)] DONE: LGB_M5 × $imp ($((($t2-$t1)/60))min)"
done

# MLP_M5 cells
for imp in $IMPUTERS; do
    t1=$(date +%s)
    echo ""
    echo "[$(date +%H:%M:%S)] MLP_M5 × $imp"
    echo "============================================================"
    $PYBIN pipeline/08_fase_b2_forecast_mlp.py "$imp"
    t2=$(date +%s); echo "[$(date +%H:%M:%S)] DONE: MLP_M5 × $imp ($((($t2-$t1)/60))min)"
done

# TFT cells
for imp in $IMPUTERS; do
    t1=$(date +%s)
    echo ""
    echo "[$(date +%H:%M:%S)] TFT × $imp"
    echo "============================================================"
    $PYBIN pipeline/25_tft_full_training.py "$imp"
    t2=$(date +%s); echo "[$(date +%H:%M:%S)] DONE: TFT × $imp ($((($t2-$t1)/60))min)"
done

T_END=$(date +%s)
echo ""
echo "============================================================"
echo "FASE 1 DONE in $(( (T_END - T0) / 60 )) min ($(( (T_END - T0) / 3600 )) h)"
echo "============================================================"
