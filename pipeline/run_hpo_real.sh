#!/bin/bash
# Run HPO completo per 5 forecaster (30 trial cad).
# Storage SQLite + load_if_exists permettono resume su crash.
# Ordine: dal più veloce al più lento.

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
        echo "WARN: exit=$rc, continuing to next forecaster"
    fi
}

# Ordine: fastest first
run "LGB_no_lags HPO (30 trial)"  "PYTHONUNBUFFERED=1 $PYBIN pipeline/32b_hpo_lgb_nolags.py"
run "LGB_M5 HPO (30 trial)"       "PYTHONUNBUFFERED=1 $PYBIN pipeline/32_hpo_lgb.py"
run "MLP_no_lags HPO (30 trial)"  "PYTHONUNBUFFERED=1 $PYBIN pipeline/33b_hpo_mlp_nolags.py"
run "MLP_M5 HPO (30 trial)"       "PYTHONUNBUFFERED=1 $PYBIN pipeline/33_hpo_mlp.py"
run "TFT HPO (30 trial)"          "PYTHONUNBUFFERED=1 $PYBIN pipeline/31_hpo_tft.py"

# Aggregate
echo ""
echo "============================================================"
echo "[$(date +%H:%M:%S)] AGGREGATE results"
echo "============================================================"
PYTHONUNBUFFERED=1 $PYBIN pipeline/34_hpo_aggregate.py

T_END=$(date +%s)
echo ""
echo "============================================================"
echo "ALL HPO DONE in $(( (T_END - T0) / 60 )) min ($(( (T_END - T0) / 3600 )) h)"
echo "Best configs in: pipeline/results/hpo_*_best.json"
echo "Summary in: pipeline/results/hpo_summary.csv"
echo "============================================================"
