#!/bin/bash
# =============================================================================
# Entrypoint SageMaker Training. SageMaker invoca l'immagine come:
#     docker run <image> train
# quindi questo script è installato come /usr/local/bin/train (eseguibile).
#
# Contratto SageMaker:
#   /opt/ml/input/data/<channel>/    input scaricati da S3 (canale "input")
#   /opt/ml/input/config/hyperparameters.json
#   /opt/ml/checkpoints              sincronizzato con CheckpointConfig.S3Uri (resume su Spot)
#   /opt/ml/model                    -> impacchettato in model.tar.gz su S3 a fine job
#   /opt/ml/output/failure           motivo del fallimento (se esce != 0)
# =============================================================================
set -euo pipefail

IN=/opt/ml/input/data/input
CKPT=/opt/ml/checkpoints
MODEL=/opt/ml/model
CFG=/opt/ml/input/config/hyperparameters.json
FAILFILE=/opt/ml/output/failure

mkdir -p "$CKPT" "$MODEL" /opt/ml/output /tmp/tft_cache
fail() { echo "$1" > "$FAILFILE"; echo "FAILED: $1" >&2; exit 1; }

[ -d "$IN" ] || fail "canale di input mancante: $IN (atteso s3://.../tft-input/)"
[ -f "$IN/data/frn50k_train.parquet" ] || fail "manca $IN/data/frn50k_train.parquet"

# --- hyperparameters SageMaker -> variabili d'ambiente -----------------------
if [ -f "$CFG" ]; then
  eval "$(python - <<'PY'
import json, shlex
try:
    hp = json.load(open('/opt/ml/input/config/hyperparameters.json'))
except Exception:
    hp = {}
for k, v in hp.items():
    key = k.strip().upper().replace('-', '_')
    if key.startswith(('TFT_', 'SERIES_')):
        print(f'export {key}={shlex.quote(str(v).strip(chr(34)))}')
PY
)"
fi

# --- wiring dei path attesi dalla pipeline ----------------------------------
# /app/data            -> input read-only scaricati da S3
# /app/pipeline/results-> checkpoint dir (sincronizzata su S3 => resume su Spot)
rm -rf /app/data /app/pipeline/results
ln -s "$IN/data" /app/data
ln -s "$CKPT"    /app/pipeline/results

# stratification.parquet è un INPUT che la pipeline cerca dentro results/
[ -f "$CKPT/stratification.parquet" ] || cp "$IN/pipeline/results/stratification.parquet" "$CKPT/"

# cache pesanti su disco locale: NON devono finire nella sync S3 dei checkpoint
export TFT_CACHE_DIR=/tmp/tft_cache

echo "======================================================================"
echo " SageMaker TFT | IN=$IN | CKPT=$CKPT (resume) | CACHE=$TFT_CACHE_DIR"
echo " SERIES_SUBSAMPLE=${SERIES_SUBSAMPLE:-0 (tutte)} | TFT_HIDDEN_CAP=${TFT_HIDDEN_CAP:-256}"
echo " TFT_PRECISION=${TFT_PRECISION:-32-true} | MODE=${TFT_MODE:-all}"
echo "======================================================================"
nvidia-smi || echo "(nvidia-smi non disponibile)"

PY=python bash /app/pipeline/run_cloud_tft.sh "${TFT_MODE:-all}"

# --- artefatti finali -> /opt/ml/model (model.tar.gz su S3) ------------------
cp "$CKPT"/*__tft_gpu_test_per_series.parquet "$MODEL"/ 2>/dev/null || true
cp "$CKPT"/hpo_tft_gpu_best.json            "$MODEL"/ 2>/dev/null || true
cp "$CKPT"/hpo_tft_gpu_trials.parquet       "$MODEL"/ 2>/dev/null || true
echo "== DONE: $(ls -1 "$MODEL" | wc -l) artefatti in $MODEL =="
