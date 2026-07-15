# TFT su SageMaker (Training Job GPU + checkpoint/resume)

Esegue la stessa immagine Docker come **SageMaker Training Job** su `ml.g4dn.2xlarge`.
Input letti da `s3://forecasting-paper/tft-input/`, risultati su S3.

**Perché Training Job (e non Processing):** SageMaker sincronizza `/opt/ml/checkpoints`
su S3 **durante** il run → un'interruzione (o Spot) non perde nulla e il job **riprende**
da dove era (Optuna dal DB, celle già fatte saltate). Un Processing Job carica gli output
solo alla fine: su un run da 5–10 h sarebbe fragile.

## Come l'immagine si adatta a SageMaker
| Contratto SageMaker | Come lo soddisfiamo |
|---|---|
| invoca `docker run <image> train` | `ENTRYPOINT tini` + `/usr/local/bin/train` (= `pipeline/sagemaker_train.sh`) |
| input in `/opt/ml/input/data/input/` | il wrapper fa `ln -s .../data → /app/data` |
| `/opt/ml/checkpoints` sincronizzato su S3 | `ln -s /opt/ml/checkpoints → /app/pipeline/results` (resume) |
| artefatti in `/opt/ml/model` | il wrapper copia lì i parquet + `hpo_tft_gpu_best.json` |
| hyperparameters JSON | letti ed esportati come env (`TFT_*`, `SERIES_*`) |
| gira come **root** | nessun `USER` fisso nell'immagine (su EC2 la non-root resta via `--user`) |
| cache pesanti (~3 GB) | `TFT_CACHE_DIR=/tmp/tft_cache` → **fuori** dalla sync S3 |

## 1. ⚠ Build per x86 (obbligatorio)
Il tuo Mac è **arm64**, le istanze SageMaker sono **x86_64**: l'immagine va costruita
per `linux/amd64`, altrimenti il job fallisce con "exec format error".
```bash
ACCOUNT=463470954204 ; REGION=eu-central-1 ; REPO=tft-gpu
ECR=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com

# repo ECR (una volta)
aws-vault exec evocity -- aws ecr create-repository --repository-name $REPO --region $REGION || true

# login docker su ECR
aws-vault exec evocity -- aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ECR

# build cross-platform + push in un colpo
docker buildx build --platform linux/amd64 -t $ECR/$REPO:latest --push .
```
(La build emulata su Mac è lenta: se preferisci, buildala sull'istanza EC2 x86 — lì
`docker build -t $ECR/$REPO:latest . && docker push $ECR/$REPO:latest` è nativo e veloce.)

## 2. IAM: execution role
Serve un **SageMaker execution role** con:
- `AmazonSageMakerFullAccess` (o equivalente minimo),
- lettura/scrittura sul bucket: `s3://forecasting-paper/tft-input/*` (get),
  `tft-output/*` e `tft-checkpoints/*` (put/get),
- pull da ECR (`AmazonEC2ContainerRegistryReadOnly`).

## 3. Lancia il job
```bash
aws-vault exec evocity -- python pipeline/sagemaker_launch.py \
  --image $ECR/tft-gpu:latest \
  --role  arn:aws:iam::463470954204:role/<SageMakerExecutionRole>
```
Opzioni:
```bash
  --spot                # managed spot (~70% risparmio, resume automatico via checkpoint)
  --subsample 15000     # riduci le serie (default 0 = tutte le 50K)
  --mode hpo|cells|all  # esegui solo una fase
  --instance ml.g4dn.2xlarge
```

## 4. Monitoraggio
```bash
# stato
aws-vault exec evocity -- python pipeline/sagemaker_launch.py --status <job-name>
# log in streaming
aws-vault exec evocity -- aws logs tail /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix <job-name> --follow
```
I checkpoint (parquet delle celle già fatte + DB Optuna) compaiono man mano in
`s3://forecasting-paper/tft-checkpoints/`.

## 5. Risultati → integrazione locale
```bash
# opzione A: artefatti finali impacchettati
aws-vault exec evocity -- aws s3 cp \
  s3://forecasting-paper/tft-output/<job-name>/output/model.tar.gz .
tar xzf model.tar.gz -C pipeline/results/

# opzione B: direttamente dai checkpoint (anche a job in corso)
aws-vault exec evocity -- aws s3 sync \
  s3://forecasting-paper/tft-checkpoints/ pipeline/results/ \
  --exclude "*" --include "*__tft_gpu_test_per_series.parquet" --include "hpo_tft_gpu_best.json"

# integrazione nella matrice + statistiche + figure
freshnet/bin/python pipeline/integrate_tft_gpu.py
```

## Costi indicativi
`ml.g4dn.2xlarge` ≈ $0.94/h on-demand (eu-central-1) → run 5–10 h ≈ **$5–10**
(≈ **$1.5–3** con `--spot`). Il job si spegne da solo a fine run: nessuna istanza da
terminare a mano (a differenza di EC2).

## Le tre modalità dell'immagine
| Dove | Comando | Path |
|---|---|---|
| **EC2** | `bash pipeline/run_docker_tft.sh all` | volumi montati (`-v data`, `-v results`) |
| **SageMaker** | job → `docker run <image> train` | S3 → `/opt/ml/...` (wiring automatico) |
| **Locale/debug** | `docker run --gpus all <image>` | CMD di default |
