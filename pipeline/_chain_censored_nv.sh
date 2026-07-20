#!/bin/bash
# Wait for the LGB cross-application (job 1) to free the machine, then export the
# censored-aware MLP/LGB tau=0.7 daily order q and re-run the newsvendor ranking.
cd /Users/utente/Desktop/FreshNetRetail
PY=freshnet/bin/python
LGB_PID=86793
echo "[chain] waiting for job1 (PID $LGB_PID)..."
while ps -p $LGB_PID >/dev/null 2>&1; do sleep 30; done
echo "[chain] job1 done. Exporting censored tau=0.7 q (MLP)..."
HPO_VARIANT=1 TAUS=0.7 EXPORT_Q=1 $PY pipeline/censored_aware_mlp.py 2>&1 | tail -8
echo "[chain] Re-running newsvendor cost over all cells..."
$PY pipeline/nv_cost.py 2>&1 | sed -n '1,12p'
echo "[chain] DONE"
