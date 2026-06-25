#!/bin/bash
# Overnight chain: wait for seed jobs -> recursive (Maj-4, 26 runs) -> aggregate.
# Fully unattended; continue-on-error so one failure does not abort the rest.
PY=/Users/utente/Desktop/FreshNetRetail/freshnet/bin/python
P=/Users/utente/Desktop/FreshNetRetail/pipeline
LOG=$P/chain_overnight.log
SEEDLOG=$P/seedrun.log
: > "$LOG"
echo "WAIT for seeds to finish ($(date))" >> "$LOG"
until grep -q "EXTRA DONE" "$SEEDLOG" 2>/dev/null; do sleep 60; done
echo "SEEDS DONE -> start recursive ($(date))" >> "$LOG"

IMPS="dlinear forward_fill imputeformer itransformer lgb linear_interp media_cond media_glob mediana_cond mediana_glob saits seasonal_naive timesnet"

for imp in $IMPS; do
  echo "=== RECURSIVE LGB $imp ($(date)) ===" >> "$LOG"
  HPO_VARIANT=1 "$PY" "$P/recursive_lgb.py" "$imp" >> "$LOG" 2>&1 || echo "FAIL recursive_lgb $imp" >> "$LOG"
done
for imp in $IMPS; do
  echo "=== RECURSIVE MLP $imp ($(date)) ===" >> "$LOG"
  HPO_VARIANT=1 "$PY" "$P/recursive_mlp.py" "$imp" >> "$LOG" 2>&1 || echo "FAIL recursive_mlp $imp" >> "$LOG"
done

echo "=== AGGREGATE ($(date)) ===" >> "$LOG"
"$PY" "$P/aggregate_robustness.py" >> "$LOG" 2>&1 || echo "FAIL aggregate" >> "$LOG"
echo "PIPELINE DONE ($(date))" >> "$LOG"
