#!/bin/bash
# TimesFM × 4 imputer mancanti per chiudere il design su tutti i 14 imputer.
# Imputer senza WAPE_recovery in Traccia A (servono solo per Sezione 1, non Sezione 2).
# Idempotente: lo script Python salta in caso di output già esistente.

cd /Users/utente/Desktop/FreshNetRetail
PYBIN="freshnet_timesfm/bin/python"
export PYTHONUNBUFFERED=1
T0=$(date +%s)

echo "============================================================"
echo "[$(date +%H:%M:%S)] TimesFM × 4 imputer mancanti"
echo "============================================================"

for imp in lgb mediana_cond media_cond media_glob; do
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
