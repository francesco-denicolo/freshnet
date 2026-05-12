#!/bin/bash
# Runs only Fase B2 (20 scripts), since Fase A already completed.

cd /Users/utente/Desktop/FreshNetRetail
PYBIN="freshnet/bin/python"
TOTAL=20
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
    # Don't abort on non-zero exit — some scripts crash AFTER saving parquet
    if [ $rc -ne 0 ]; then
        echo "WARN: exit=$rc, continuing (parquet may already be saved)"
    fi
}

# Fase B2 LGB M5 × 10 imputer (eccetto mediana_cond già fatto)
run "LGB_M5 × dlinear"          "PYTHONUNBUFFERED=1 $PYBIN pipeline/07_fase_b2_forecast_lgb.py dlinear"
run "LGB_M5 × forward_fill"     "PYTHONUNBUFFERED=1 $PYBIN pipeline/07_fase_b2_forecast_lgb.py forward_fill"
run "LGB_M5 × lgb"              "PYTHONUNBUFFERED=1 $PYBIN pipeline/07_fase_b2_forecast_lgb.py lgb"
run "LGB_M5 × linear_interp"    "PYTHONUNBUFFERED=1 $PYBIN pipeline/07_fase_b2_forecast_lgb.py linear_interp"
run "LGB_M5 × media_cond"       "PYTHONUNBUFFERED=1 $PYBIN pipeline/07_fase_b2_forecast_lgb.py media_cond"
run "LGB_M5 × media_glob"       "PYTHONUNBUFFERED=1 $PYBIN pipeline/07_fase_b2_forecast_lgb.py media_glob"
run "LGB_M5 × mediana_glob"     "PYTHONUNBUFFERED=1 $PYBIN pipeline/07_fase_b2_forecast_lgb.py mediana_glob"
run "LGB_M5 × saits"            "PYTHONUNBUFFERED=1 $PYBIN pipeline/07_fase_b2_forecast_lgb.py saits"
run "LGB_M5 × seasonal_naive"   "PYTHONUNBUFFERED=1 $PYBIN pipeline/07_fase_b2_forecast_lgb.py seasonal_naive"

# Fase B2 MLP M5 × 10 imputer (incluso mediana_cond)
run "MLP_M5 × dlinear"          "PYTHONUNBUFFERED=1 $PYBIN pipeline/08_fase_b2_forecast_mlp.py dlinear"
run "MLP_M5 × forward_fill"     "PYTHONUNBUFFERED=1 $PYBIN pipeline/08_fase_b2_forecast_mlp.py forward_fill"
run "MLP_M5 × lgb"              "PYTHONUNBUFFERED=1 $PYBIN pipeline/08_fase_b2_forecast_mlp.py lgb"
run "MLP_M5 × linear_interp"    "PYTHONUNBUFFERED=1 $PYBIN pipeline/08_fase_b2_forecast_mlp.py linear_interp"
run "MLP_M5 × media_cond"       "PYTHONUNBUFFERED=1 $PYBIN pipeline/08_fase_b2_forecast_mlp.py media_cond"
run "MLP_M5 × media_glob"       "PYTHONUNBUFFERED=1 $PYBIN pipeline/08_fase_b2_forecast_mlp.py media_glob"
run "MLP_M5 × mediana_cond"     "PYTHONUNBUFFERED=1 $PYBIN pipeline/08_fase_b2_forecast_mlp.py mediana_cond"
run "MLP_M5 × mediana_glob"     "PYTHONUNBUFFERED=1 $PYBIN pipeline/08_fase_b2_forecast_mlp.py mediana_glob"
run "MLP_M5 × saits"            "PYTHONUNBUFFERED=1 $PYBIN pipeline/08_fase_b2_forecast_mlp.py saits"
run "MLP_M5 × seasonal_naive"   "PYTHONUNBUFFERED=1 $PYBIN pipeline/08_fase_b2_forecast_mlp.py seasonal_naive"

T_END=$(date +%s)
echo ""
echo "============================================================"
echo "ALL 20 SCRIPTS DONE in $(( (T_END - T0) / 60 )) min"
echo "============================================================"
