#!/bin/bash
# TimesFM × 3 imputer chiave (spot-check)

cd /Users/utente/Desktop/FreshNetRetail
PYBIN="freshnet_timesfm/bin/python"
export PYTHONUNBUFFERED=1
T0=$(date +%s)

echo "============================================================"
echo "[$(date +%H:%M:%S)] TimesFM \xc3\x97 3 imputers chiave"
echo "============================================================"

for imp in imputeformer mediana_glob timesnet; do
    t1=$(date +%s)
    echo ""
    echo "[$(date +%H:%M:%S)] TimesFM \xc3\x97 $imp"
    $PYBIN pipeline/40_fase_b2_forecast_timesfm.py "$imp"
    t2=$(date +%s); echo "[$(date +%H:%M:%S)] DONE: $imp ($((($t2-$t1)/60))min)"
done

T_END=$(date +%s)
echo ""
echo "============================================================"
echo "DONE in $(( (T_END - T0) / 60 )) min"
echo "============================================================"
