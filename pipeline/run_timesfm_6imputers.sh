#!/bin/bash
# TimesFM × 6 imputer aggiuntivi (saits, itransformer, dlinear, forward_fill,
# seasonal_naive, linear_interp) per chiudere il design RQ2 a n=9 imputer.
# Idempotente: lo script Python salta in caso di output già esistente.

cd /Users/utente/Desktop/FreshNetRetail
PYBIN="freshnet_timesfm/bin/python"
export PYTHONUNBUFFERED=1
T0=$(date +%s)

echo "============================================================"
echo "[$(date +%H:%M:%S)] TimesFM × 6 imputer aggiuntivi"
echo "============================================================"

for imp in saits itransformer dlinear forward_fill seasonal_naive linear_interp; do
    t1=$(date +%s)
    echo ""
    echo "[$(date +%H:%M:%S)] TimesFM × $imp"
    $PYBIN pipeline/40_fase_b2_forecast_timesfm.py "$imp"
    t2=$(date +%s); echo "[$(date +%H:%M:%S)] DONE: $imp ($((($t2-$t1)/60))min)"
done

T_END=$(date +%s)
echo ""
echo "============================================================"
echo "DONE in $(( (T_END - T0) / 60 )) min"
echo "============================================================"
