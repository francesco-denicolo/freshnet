#!/bin/bash
# =============================================================================
# run_docker_tft.sh — lancia il container TFT-GPU sul g4dn.2xlarge
# =============================================================================
# Prerequisiti sull'host EC2:
#   - NVIDIA driver + nvidia-container-toolkit (la Deep Learning AMI li ha già)
#   - immagine costruita:  docker build -t tft-gpu:latest .
#   - dati montati:  $DATA (frn50k_*, completed_sales_622/) e
#                    $RESULTS (deve contenere stratification.parquet; riceve gli output)
#
# Uso:
#   bash pipeline/run_docker_tft.sh all      # HPO + 14 celle (default)
#   bash pipeline/run_docker_tft.sh hpo      # solo HPO
#   bash pipeline/run_docker_tft.sh cells    # solo le celle
#   SERIES_SUBSAMPLE=15000 bash pipeline/run_docker_tft.sh all   # opzionale (RAM ridotta)
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-tft-gpu:latest}"
DATA="${DATA:-$PWD/data}"
RESULTS="${RESULTS:-$PWD/pipeline/results}"
HIDDEN_CAP="${TFT_HIDDEN_CAP:-256}"
PRECISION="${TFT_PRECISION:-32-true}"

# sanity check input
[ -f "$DATA/frn50k_train.parquet" ] || { echo "MANCA $DATA/frn50k_train.parquet"; exit 1; }
[ -f "$RESULTS/stratification.parquet" ] || { echo "MANCA $RESULTS/stratification.parquet"; exit 1; }
mkdir -p "$RESULTS"

echo "IMAGE=$IMAGE | DATA=$DATA | RESULTS=$RESULTS | HIDDEN_CAP=$HIDDEN_CAP | SUBSAMPLE=${SERIES_SUBSAMPLE:-0}"

docker run --rm --gpus all \
  --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PY=python \
  -e TFT_HIDDEN_CAP="$HIDDEN_CAP" \
  -e TFT_PRECISION="$PRECISION" \
  ${SERIES_SUBSAMPLE:+-e SERIES_SUBSAMPLE="$SERIES_SUBSAMPLE"} \
  -v "$DATA":/app/data:ro \
  -v "$RESULTS":/app/pipeline/results \
  "$IMAGE" \
  bash pipeline/run_cloud_tft.sh "${1:-all}"
