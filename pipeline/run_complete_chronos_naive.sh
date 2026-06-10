#!/bin/bash
# Complete Chronos + naive cells for iTransformer, TimesNet, ImputeFormer (no CSDI)

cd /Users/utente/Desktop/FreshNetRetail
PYBIN="freshnet/bin/python"
export PYTHONUNBUFFERED=1
T0=$(date +%s)

echo "============================================================"
echo "[$(date +%H:%M:%S)] Completing matrix: Chronos × 3 imputers + Naive × 3 imputers"
echo "============================================================"

# Chronos × {itransformer, timesnet, imputeformer}
for imp in itransformer timesnet imputeformer; do
    t1=$(date +%s)
    echo ""; echo "[$(date +%H:%M:%S)] Chronos × $imp"
    $PYBIN pipeline/10_fase_b2_forecast_chronos.py "$imp"
    t2=$(date +%s); echo "[$(date +%H:%M:%S)] DONE Chronos × $imp ($((($t2-$t1)/60))min)"
done

# Naive × {itransformer, timesnet, imputeformer} — script iterates 3 forecasters × 7 imputers internally,
# we rely on existing_cells check to skip already-done ones.
t1=$(date +%s)
echo ""; echo "[$(date +%H:%M:%S)] Naive forecasters (skip-existing)"
$PYBIN pipeline/06_fase_b2_forecast_naive.py
t2=$(date +%s); echo "[$(date +%H:%M:%S)] DONE naive ($((($t2-$t1)/60))min)"

T_END=$(date +%s)
echo ""
echo "============================================================"
echo "DONE in $(( (T_END - T0) / 60 )) min"
echo "============================================================"
