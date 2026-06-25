#!/bin/bash
# Newsvendor chain: export daily orders q for 13 imputers x {LGB-M5, MLP-M5} = 26 cells
# (HPO variant, frozen/direct protocol), then compute the asymmetric order cost.
# Unattended; continue-on-error; skip-existing (q parquet is the skip target).
PY=/Users/utente/Desktop/FreshNetRetail/freshnet/bin/python
P=/Users/utente/Desktop/FreshNetRetail/pipeline
LOG=$P/nv_chain.log
: > "$LOG"
echo "NV CHAIN START ($(date))" >> "$LOG"
IMPS="dlinear forward_fill imputeformer itransformer lgb linear_interp media_cond media_glob mediana_cond mediana_glob saits seasonal_naive timesnet"

for imp in $IMPS; do
  echo "=== NV LGB $imp ($(date)) ===" >> "$LOG"
  HPO_VARIANT=1 "$PY" "$P/nv_lgb.py" "$imp" >> "$LOG" 2>&1 || echo "FAIL nv_lgb $imp" >> "$LOG"
  echo "=== NV MLP $imp ($(date)) ===" >> "$LOG"
  HPO_VARIANT=1 "$PY" "$P/nv_mlp.py" "$imp" >> "$LOG" 2>&1 || echo "FAIL nv_mlp $imp" >> "$LOG"
done

echo "=== NV COST ($(date)) ===" >> "$LOG"
"$PY" "$P/nv_cost.py" >> "$LOG" 2>&1 || echo "FAIL nv_cost" >> "$LOG"
echo "NV CHAIN DONE ($(date))" >> "$LOG"
