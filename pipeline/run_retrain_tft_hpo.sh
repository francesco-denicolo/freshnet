#!/bin/bash
# Re-train TFT cells with HPO-tuned HPs (11 imputer cells)
# Output: {imp}__tft_hpo_test_per_series.parquet

cd /Users/utente/Desktop/FreshNetRetail
PYBIN="freshnet/bin/python"
export HPO_VARIANT=1
export PYTHONUNBUFFERED=1

IMPUTERS="no_imp dlinear forward_fill lgb linear_interp media_cond media_glob mediana_cond mediana_glob saits seasonal_naive"
TOTAL=11

T0=$(date +%s)
echo "============================================================"
echo "[$(date +%H:%M:%S)] HPO TFT RE-TRAIN: $TOTAL cells"
echo "============================================================"

i=0
for imp in $IMPUTERS; do
    i=$((i+1))
    t1=$(date +%s)
    echo ""
    echo "[$(date +%H:%M:%S)] [$i/$TOTAL] TFT × $imp"
    echo "============================================================"
    $PYBIN pipeline/25_tft_full_training.py "$imp"
    rc=$?
    t2=$(date +%s); dt=$((t2-t1))
    echo "[$(date +%H:%M:%S)] [$i/$TOTAL] DONE: $imp (exit=$rc, ${dt}s = $((dt/60))min)"
    if [ $rc -ne 0 ]; then
        echo "WARN: exit=$rc, continuing"
    fi
done

T_END=$(date +%s)
echo ""
echo "============================================================"
echo "ALL TFT DONE in $(( (T_END - T0) / 60 )) min ($(( (T_END - T0) / 3600 )) h)"
echo "============================================================"
