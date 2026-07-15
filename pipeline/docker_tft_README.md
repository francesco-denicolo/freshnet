# TFT su EC2 con Docker (g4dn.2xlarge)

Alternativa al setup venv di `ec2_tft_README.md`: un container CUDA riproducibile e
isolato. **I dati non sono nell'immagine** — si montano a runtime, così l'immagine resta
piccola e i dati restano sull'host (e persistono).

## 0. Prerequisiti host (una volta)
Sulla **Deep Learning AMI** sono già presenti Docker + NVIDIA driver + nvidia-container-toolkit.
Su Ubuntu liscia:
```bash
# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER    # poi riesegui il login
# NVIDIA Container Toolkit
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
# verifica GPU nel container:
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## 1. Porta codice + dati sull'host
Il codice (repo) e i dati (~1.1 GB) vanno sull'istanza — vedi §2 di `ec2_tft_README.md`
(rsync dal Mac). Servono, sotto la root del repo:
- `data/frn50k_train.parquet`, `data/frn50k_eval.parquet`
- `data/completed_sales_622/` (i 13 file imputer)
- `pipeline/results/stratification.parquet`  ← input, va nella dir results montata
- `Dockerfile`, `.dockerignore`, `pipeline/` (script)

## 2. Build dell'immagine (dalla root del repo)
```bash
docker build -t tft-gpu:latest .
```
La prima build scarica torch cu121 (~2.5 GB): mettici qualche minuto. Le build successive
usano la cache (i layer delle dipendenze non si ricostruiscono se non cambia
`requirements_tft_gpu.txt`).

## 3. Run completo (50K piene, in tmux)
```bash
tmux new -s tft
bash pipeline/run_docker_tft.sh all 2>&1 | tee tft.log
#  stacca: Ctrl-b poi d   |   riattacca: tmux attach -t tft
```
Lo script monta `data/` (read-only) e `pipeline/results/` (read-write) nel container, gira su
GPU (`--gpus all`), e lancia `run_cloud_tft.sh all` con `TFT_HIDDEN_CAP=256`,
`TFT_PRECISION=32-true`, nessun subsample (50K piene).
Gli output finiscono in `pipeline/results/` **sull'host** (persistono anche se il container esce).

Varianti:
```bash
bash pipeline/run_docker_tft.sh hpo        # solo HPO
bash pipeline/run_docker_tft.sh cells      # solo le celle (HPO già fatto)
SERIES_SUBSAMPLE=15000 bash pipeline/run_docker_tft.sh all   # se vuoi ridurre la RAM
```

## 4. Monitoraggio
```bash
tail -f tft.log                                   # progressi
docker stats                                      # CPU/RAM del container
nvidia-smi                                         # uso GPU (dall'host)
```

## 5. Resume / Spot
- **Resumable**: `run_cloud_tft.sh` salta le celle già presenti e Optuna riprende dal DB
  (`hpo_tft_gpu.db`), tutto nella dir results montata sull'host → nulla si perde se il
  container esce o l'istanza si ferma.
- **Spot**: `tini` come PID 1 propaga il SIGTERM → chiusura pulita. Alla ripartenza,
  rilancia lo stesso comando: riprende da dove era.

## 6. Fine
```bash
# scarica gli output sul Mac (dal Mac):
rsync -avz -e "ssh -i KEY.pem" ubuntu@<IP>:~/FreshNetRetail/pipeline/results/'*__tft_gpu_*.parquet' pipeline/results/
rsync -avz -e "ssh -i KEY.pem" ubuntu@<IP>:~/FreshNetRetail/pipeline/results/hpo_tft_gpu_best.json pipeline/results/
# integrazione in locale:
freshnet/bin/python pipeline/integrate_tft_gpu.py
```
E **termina l'istanza EC2** per non pagare.

## Note su sicurezza/efficienza
- **Non-root**: il container gira come l'utente host (`--user $(id -u):$(id -g)`, `HOME=/tmp`)
  → nessun file root sui volumi montati, nessun problema di permessi.
- **Dati read-only**: `data/` è montato `:ro` (il codice non lo modifica).
- **Immagine snella**: dati/output/venv/git esclusi via `.dockerignore`; solo gli script sono nell'immagine.
- **Isolamento**: nessun conflitto col torch/CUDA di sistema; l'ambiente è quello pinnato nell'immagine.
- **`--shm-size=8g`**: memoria condivisa ampia per i DataLoader (innocuo con num_workers=0, utile se li aumenti).
