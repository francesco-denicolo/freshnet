#!/bin/bash
# Smoke test HPO: n_trials=2 per forecaster, budget ridotto.
# Output in pipeline/results/hpo_*_smoke.db / hpo_*_smoke_*.json
# Pre-build cache se necessario.

cd /Users/utente/Desktop/FreshNetRetail
PYBIN="freshnet/bin/python"
T0=$(date +%s)

echo "============================================================"
echo "[$(date +%H:%M:%S)] SMOKE TEST HPO — n_trials=2 × 3 forecaster"
echo "============================================================"

# LGB: build cache + 2 trial
echo ""
echo "[$(date +%H:%M:%S)] [1/3] LGB smoke (con cache build)"
echo "============================================================"
HPO_SMOKE=1 PYTHONUNBUFFERED=1 $PYBIN pipeline/32_hpo_lgb.py
T1=$(date +%s); echo "[$(date +%H:%M:%S)] [1/3] DONE in $(( (T1 - T0) / 60 )) min"

# MLP: usa cache da LGB
echo ""
echo "[$(date +%H:%M:%S)] [2/3] MLP smoke (riusa cache LGB)"
echo "============================================================"
HPO_SMOKE=1 PYTHONUNBUFFERED=1 $PYBIN pipeline/33_hpo_mlp.py
T2=$(date +%s); echo "[$(date +%H:%M:%S)] [2/3] DONE in $(( (T2 - T1) / 60 )) min"

# TFT: cache propria
echo ""
echo "[$(date +%H:%M:%S)] [3/3] TFT smoke (con cache build)"
echo "============================================================"
HPO_SMOKE=1 PYTHONUNBUFFERED=1 $PYBIN pipeline/31_hpo_tft.py
T3=$(date +%s); echo "[$(date +%H:%M:%S)] [3/3] DONE in $(( (T3 - T2) / 60 )) min"

echo ""
echo "============================================================"
echo "SMOKE TEST DONE in $(( (T3 - T0) / 60 )) min"
echo "Output files (smoke versions):"
ls -la pipeline/results/hpo_*_smoke.db pipeline/results/hpo_*_smoke_best.json 2>&1 | head -20
echo "============================================================"
