#!/bin/bash
PY=/Users/utente/Desktop/FreshNetRetail/freshnet/bin/python
P=/Users/utente/Desktop/FreshNetRetail/pipeline
LOG=$P/mase_chain.log; : > "$LOG"
echo "MASE CHAIN START ($(date))" >> "$LOG"
for imp in itransformer timesnet; do
  echo "=== MASE MLP $imp ($(date)) ===" >> "$LOG"
  HPO_VARIANT=1 "$PY" "$P/mase_mlp.py" "$imp" >> "$LOG" 2>&1 || echo "FAIL mlp $imp" >> "$LOG"
done
for imp in mediana_glob; do
  echo "=== MASE LGB $imp ($(date)) ===" >> "$LOG"
  HPO_VARIANT=1 "$PY" "$P/mase_lgb.py" "$imp" >> "$LOG" 2>&1 || echo "FAIL lgb $imp" >> "$LOG"
done
echo "MASE CHAIN DONE ($(date))" >> "$LOG"
