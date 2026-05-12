#!/bin/bash
# Re-runs 7 forecaster cells that use lgb as imputer (after LGB imputer MAE re-train).

cd /Users/utente/Desktop/FreshNetRetail
PYBIN="freshnet/bin/python"
TOTAL=5
i=0
T0=$(date +%s)

run() {
    i=$((i+1))
    local desc="$1"
    local cmd="$2"
    local t1=$(date +%s)
    echo ""
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] [$i/$TOTAL] START: $desc"
    echo "============================================================"
    eval "$cmd"
    local rc=$?
    local t2=$(date +%s)
    local dt=$((t2 - t1))
    echo "[$(date +%H:%M:%S)] [$i/$TOTAL] DONE: $desc (exit=$rc, ${dt}s = $((dt/60)) min)"
    echo "[$(date +%H:%M:%S)] CUMULATIVE elapsed: $(( (t2 - T0) / 60 )) min"
    if [ $rc -ne 0 ]; then
        echo "WARN: exit=$rc, continuing"
    fi
}

# 3 Naive con script 29 (single imputer) — produce 3 cells in un solo run
run "Naive (3 cells) × lgb"             "PYTHONUNBUFFERED=1 $PYBIN pipeline/29_fase_b2_forecast_naive_single.py lgb"

# LGB_M5 × lgb
run "LGB_M5 × lgb"                       "PYTHONUNBUFFERED=1 $PYBIN pipeline/07_fase_b2_forecast_lgb.py lgb"

# MLP_M5 × lgb
run "MLP_M5 × lgb"                       "PYTHONUNBUFFERED=1 $PYBIN pipeline/08_fase_b2_forecast_mlp.py lgb"

# Chronos × lgb
run "Chronos × lgb"                      "PYTHONUNBUFFERED=1 $PYBIN pipeline/10_fase_b2_forecast_chronos.py lgb"

# TFT × lgb
run "TFT × lgb"                          "PYTHONUNBUFFERED=1 $PYBIN pipeline/25_tft_full_training.py lgb"

T_END=$(date +%s)
echo ""
echo "============================================================"
echo "ALL 5 SCRIPTS DONE in $(( (T_END - T0) / 60 )) min (7 cells produced)"
echo "============================================================"
