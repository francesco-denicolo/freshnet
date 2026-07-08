#!/bin/bash
# ============================================================================
# run_cloud_tft.sh — resourcing "vero" del TFT su GPU cloud
# ============================================================================
# 1) HPO ampia (no cap hidden=32, hidden fino a 256, ~48 trial)
# 2) training delle 14 celle {imputer} × TFT con la config vincente
#
# Uso (dalla root del repo, dopo aver fatto lo staging dei dati — vedi
# cloud_tft_README.md):
#   bash pipeline/run_cloud_tft.sh            # HPO + tutte le 14 celle
#   bash pipeline/run_cloud_tft.sh hpo        # solo HPO
#   bash pipeline/run_cloud_tft.sh cells      # solo le celle (HPO già fatto)
#
# Ogni step è resumable: gli output esistenti vengono saltati, e lo study
# Optuna riprende da hpo_tft_gpu.db.
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python}"                      # override con: PY=freshnet/bin/python ...
MODE="${1:-all}"

# tutte le 14 righe imputer della matrice finale (14 × TFT)
IMPUTERS=(no_imp media_glob media_cond mediana_glob mediana_cond \
          forward_fill seasonal_naive linear_interp lgb \
          dlinear saits itransformer timesnet imputeformer)

echo "######## CLOUD TFT $(date) | mode=$MODE ########"
$PY -c "import torch; print('CUDA available:', torch.cuda.is_available());
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

if [ "$MODE" = "all" ] || [ "$MODE" = "hpo" ]; then
  echo "==== [HPO] TFT GPU $(date) ===="
  $PY pipeline/31_hpo_tft_gpu.py
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "cells" ]; then
  for imp in "${IMPUTERS[@]}"; do
    echo "==== [CELL] $imp × TFT $(date) ===="
    $PY pipeline/25_tft_full_training_gpu.py "$imp"
  done
fi

echo "######## CLOUD TFT DONE $(date) ########"
echo "Output: pipeline/results/*__tft_gpu_test_per_series.parquet + hpo_tft_gpu_best.json"
echo "Scarica questi file in locale per l'integrazione nella matrice."
