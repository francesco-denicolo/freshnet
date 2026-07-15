#!/bin/bash
# =============================================================================
# codebuild_build.sh — impacchetta il source, lo carica su S3 e lancia CodeBuild
# =============================================================================
# CodeBuild costruisce l'immagine nativamente su x86_64 (= quello che serve a
# SageMaker) e la pusha su ECR. Niente Docker sul Mac, niente emulazione arm64.
#
# Uso:
#   aws-vault exec evocity -- bash pipeline/codebuild_build.sh          # build + attende
#   aws-vault exec evocity -- bash pipeline/codebuild_build.sh --no-wait
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${PROJECT:-tft-gpu-build}"
BUCKET="${TFT_BUCKET:-forecasting-paper}"
KEY="codebuild-source/tft-gpu-src.zip"
REGION="${AWS_REGION:-eu-central-1}"
ZIP=$(mktemp -t tft-src-XXXX).zip

echo ">> impacchetto il source (solo build context: Dockerfile + buildspec + script)"
rm -f "$ZIP"
zip -q -r "$ZIP" \
  Dockerfile buildspec.yml \
  pipeline/requirements_tft_gpu.txt \
  pipeline/31_hpo_tft_gpu.py pipeline/25_tft_full_training_gpu.py \
  pipeline/run_cloud_tft.sh pipeline/sagemaker_train.sh
echo "   $(du -h "$ZIP" | cut -f1)  ($(unzip -l "$ZIP" | tail -1 | awk '{print $2}') file)"

echo ">> upload su s3://$BUCKET/$KEY"
aws s3 cp "$ZIP" "s3://$BUCKET/$KEY" --only-show-errors
rm -f "$ZIP"

echo ">> avvio build"
ID=$(aws codebuild start-build --project-name "$PROJECT" --region "$REGION" \
      --query 'build.id' --output text)
echo "   build id: $ID"

if [ "${1:-}" = "--no-wait" ]; then
  echo "   (--no-wait) segui con: aws codebuild batch-get-builds --ids $ID"
  exit 0
fi

echo ">> attendo il completamento (log: CloudWatch /aws/codebuild/$PROJECT)"
while true; do
  S=$(aws codebuild batch-get-builds --ids "$ID" --region "$REGION" \
        --query 'builds[0].buildStatus' --output text)
  P=$(aws codebuild batch-get-builds --ids "$ID" --region "$REGION" \
        --query 'builds[0].currentPhase' --output text)
  printf "\r   stato=%s fase=%-20s" "$S" "$P"
  [ "$S" = "IN_PROGRESS" ] || break
  sleep 20
done
echo ""
if [ "$S" = "SUCCEEDED" ]; then
  ACC=$(aws sts get-caller-identity --query Account --output text)
  echo "BUILD OK -> $ACC.dkr.ecr.$REGION.amazonaws.com/tft-gpu:latest"
else
  echo "BUILD $S — log:"
  aws logs tail "/aws/codebuild/$PROJECT" --region "$REGION" --since 30m | tail -30 || true
  exit 1
fi
