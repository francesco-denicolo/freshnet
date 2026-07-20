#!/bin/bash
# Overnight batch for the two remaining review computations.
#   1) Principled censored-quantile cell: MLP-M5 with tau(h)=0.5+0.5*f(h), tied to
#      the per-hour stock-out frequency (referee 2.4). ~15-25 min.
#   2) Co-adaptation check via the PRODUCTION HPO pipeline (Strada A): re-run the MLP
#      hyperparameter search on the itransformer-COMPLETED basis and compare the
#      selected config to the paper's censored-basis config. If they match, the
#      tuning basis is invariant and the paper's cells (W=0.03) already answer the
#      co-adaptation objection -- no re-implementation, no retrains. ~5-7 h (45 trials,
#      full 50K; Optuna storage resumes if interrupted).
# Sequential so they do not contend on the MPS/GPU.
# Monitor: tail -f pipeline/results/_log_overnight.txt
cd /Users/utente/Desktop/FreshNetRetail
PY=freshnet/bin/python
LOG=pipeline/results/_log_overnight.txt
{
  echo "=== [1/3] censored-quantile tau(h) MLP-M5 ==="
  HPO_VARIANT=1 TAU_H=1 EXPORT_Q=1 $PY pipeline/censored_aware_mlp.py
  echo "=== [2/3] newsvendor cost (add tau(h) cell) ==="
  $PY pipeline/nv_cost.py | sed -n '1,14p'
  echo "=== [3/3] co-adaptation: MLP HPO on the itransformer basis (production pipeline) ==="
  TUNE_IMPUTER=itransformer $PY pipeline/33_hpo_mlp.py
  echo "=== config comparison (censored basis vs itransformer basis) ==="
  $PY - <<'PYEOF'
import json, os
R='pipeline/results'
def show(f):
    p=os.path.join(R,f)
    if not os.path.exists(p): return f, None
    d=json.load(open(p)); return f, d.get('best_params'), d.get('best_value')
for f in ['hpo_mlp_best.json','hpo_mlp_itransformer_best.json']:
    print(f, '->', show(f))
PYEOF
  echo "=== ALL OVERNIGHT DONE ==="
} > "$LOG" 2>&1 &
echo "launched overnight batch, PID $!"
echo "monitor: tail -f $LOG"
