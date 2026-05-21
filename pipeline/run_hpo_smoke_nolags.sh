#!/bin/bash
# Smoke test HPO per LGB_nolags + MLP_nolags varianti.

cd /Users/utente/Desktop/FreshNetRetail
PYBIN="freshnet/bin/python"
T0=$(date +%s)

echo "============================================================"
echo "[$(date +%H:%M:%S)] SMOKE TEST HPO NOLAGS — n_trials=2 × 2 variants"
echo "============================================================"

# LGB nolags
echo ""
echo "[$(date +%H:%M:%S)] [1/2] LGB nolags smoke"
echo "============================================================"
HPO_SMOKE=1 PYTHONUNBUFFERED=1 $PYBIN pipeline/32b_hpo_lgb_nolags.py
T1=$(date +%s); echo "[$(date +%H:%M:%S)] [1/2] DONE in $(( (T1 - T0) / 60 )) min"

# MLP nolags
echo ""
echo "[$(date +%H:%M:%S)] [2/2] MLP nolags smoke"
echo "============================================================"
HPO_SMOKE=1 PYTHONUNBUFFERED=1 $PYBIN pipeline/33b_hpo_mlp_nolags.py
T2=$(date +%s); echo "[$(date +%H:%M:%S)] [2/2] DONE in $(( (T2 - T1) / 60 )) min"

echo ""
echo "============================================================"
echo "SMOKE NOLAGS DONE in $(( (T2 - T0) / 60 )) min"
ls -la pipeline/results/hpo_*nolags_smoke* 2>&1 | head -10
echo "============================================================"
