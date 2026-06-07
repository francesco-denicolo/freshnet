#!/bin/bash
# Re-train cells with HPO-tuned HPs (LGB_M5 + MLP_M5 imputer cells)
# Output files have _hpo suffix

cd /Users/utente/Desktop/FreshNetRetail
PYBIN="freshnet/bin/python"
export HPO_VARIANT=1
export PYTHONUNBUFFERED=1

IMPUTERS="dlinear forward_fill lgb linear_interp media_cond media_glob mediana_cond mediana_glob saits seasonal_naive"

T0=$(date +%s)
echo "============================================================"
echo "[$(date +%H:%M:%S)] HPO RE-TRAIN: LGB_M5 (10 cells) + MLP_M5 (9 cells)"
echo "============================================================"

# LGB_M5 cells
i=0
for imp in $IMPUTERS; do
    i=$((i+1))
    t1=$(date +%s)
    echo ""
    echo "[$(date +%H:%M:%S)] [$i/10] LGB_M5 × $imp"
    echo "============================================================"
    $PYBIN pipeline/07_fase_b2_forecast_lgb.py "$imp"
    t2=$(date +%s); dt=$((t2-t1))
    echo "[$(date +%H:%M:%S)] DONE: $imp (${dt}s = $((dt/60))min)"
done

# MLP_M5 cells (skip saits, already done)
i=0
for imp in $IMPUTERS; do
    if [ "$imp" = "saits" ]; then continue; fi
    i=$((i+1))
    t1=$(date +%s)
    echo ""
    echo "[$(date +%H:%M:%S)] [$i/9] MLP_M5 × $imp"
    echo "============================================================"
    $PYBIN pipeline/08_fase_b2_forecast_mlp.py "$imp"
    t2=$(date +%s); dt=$((t2-t1))
    echo "[$(date +%H:%M:%S)] DONE: $imp (${dt}s = $((dt/60))min)"
done

T_END=$(date +%s)
echo ""
echo "============================================================"
echo "ALL DONE in $(( (T_END - T0) / 60 )) min ($(( (T_END - T0) / 3600 )) h)"
echo "============================================================"
