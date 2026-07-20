#!/bin/bash
# Overnight batch for the two remaining review computations.
#   1) Principled censored-quantile cell: MLP-M5 with tau(h)=0.5+0.5*f(h), tied to
#      the per-hour stock-out frequency (referee 2.4). ~15-25 min.
#   2) Full-50K re-tune co-adaptation check (review risk #2). ~7-9 h.
# Sequential (both use the MPS/GPU) so they do not contend.
# Monitor: tail -f pipeline/results/_log_overnight.txt
cd /Users/utente/Desktop/FreshNetRetail
PY=freshnet/bin/python
LOG=pipeline/results/_log_overnight.txt
{
  echo "=== [1/3] censored-quantile tau(h) MLP-M5 ==="
  HPO_VARIANT=1 TAU_H=1 EXPORT_Q=1 $PY pipeline/censored_aware_mlp.py
  echo "=== [2/3] newsvendor cost (add tau(h) cell) ==="
  $PY pipeline/nv_cost.py | sed -n '1,14p'
  echo "=== [3/3] full-50K re-tune (7-9 h) ==="
  SUB=0 $PY pipeline/retune_itransformer_W.py
  echo "=== ALL OVERNIGHT DONE ==="
} > "$LOG" 2>&1 &
echo "launched overnight batch, PID $!"
echo "monitor: tail -f $LOG"
