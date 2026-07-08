# Resourcing TFT su GPU cloud

Obiettivo: valutare il TFT **equamente**, togliendo il cap di memoria (`hidden ≤ 32`)
imposto dai 16 GB della macchina locale. Su GPU si esplora `hidden` fino a 256 e si
completa anche la cella `imputeformer × TFT` che in locale andava OOM.

## 0. Istanza
GPU singola NVIDIA da ≥ 24 GB VRAM (L4 / A10 / A100). CUDA 12.x, ~40 GB disco.

## 1. Staging dati (da locale, ~1.8 GB)
Dalla root del repo locale, verso la macchina cloud (stessa struttura di cartelle):
```bash
rsync -avz \
  data/frn50k_train.parquet data/frn50k_eval.parquet \
  <user>@<host>:~/FreshNetRetail/data/
rsync -avz data/completed_sales_622/ <user>@<host>:~/FreshNetRetail/data/completed_sales_622/
rsync -avz pipeline/results/stratification.parquet <user>@<host>:~/FreshNetRetail/pipeline/results/
# gli script:
rsync -avz pipeline/31_hpo_tft_gpu.py pipeline/25_tft_full_training_gpu.py \
  pipeline/run_cloud_tft.sh pipeline/requirements_tft_gpu.txt \
  <user>@<host>:~/FreshNetRetail/pipeline/
```

## 2. Ambiente
```bash
cd ~/FreshNetRetail
python -m venv venv && source venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r pipeline/requirements_tft_gpu.txt
```

## 3. Smoke test (2 min, verifica che tutto giri su GPU)
```bash
HPO_SMOKE=1 PY=python bash pipeline/run_cloud_tft.sh hpo
```
Deve stampare `CUDA available: True` e completare 2 trial.

## 4. Run completo
```bash
# opzionale: tuning budget via env
#   TFT_N_TRIALS=48 TFT_HIDDEN_CAP=256 TFT_MAX_EPOCHS=30 TFT_MAX_TRAIN=400000
#   TFT_PRECISION=16-mixed   # più veloce/meno VRAM (default 32-true, più sicuro)
nohup bash pipeline/run_cloud_tft.sh all > pipeline/cloud_tft.log 2>&1 &
tail -f pipeline/cloud_tft.log
```
Fasi: (1) HPO ampia → `hpo_tft_gpu_best.json`; (2) 14 celle → `{imp}__tft_gpu_test_per_series.parquet`.
Resumable: rilanciando, gli output già presenti vengono saltati e lo study Optuna riprende.

## 5. Risultati da riportare in locale
```bash
rsync -avz <user>@<host>:~/FreshNetRetail/pipeline/results/'*__tft_gpu_test_per_series.parquet' \
  pipeline/results/
rsync -avz <user>@<host>:~/FreshNetRetail/pipeline/results/hpo_tft_gpu_best.json \
  <user>@<host>:~/FreshNetRetail/pipeline/results/hpo_tft_gpu_trials.parquet \
  pipeline/results/
```
Poi in locale: integrazione nella matrice, ri-esecuzione Friedman/Nemenyi/CD, rigenerazione
figure (heatmap/CD/Pareto/crossover) e aggiornamento del testo del paper.

## Stima costo/tempo (indicativa, A10/L4)
- HPO ampia (~48 trial, hidden≤256, epoche piene, con pruning): ~3–6 h
- 14 celle (training + predict, best config): ~2–4 h
- Totale ~5–10 h GPU → ~$8–20 a ~$1.5/h. A100 dimezza i tempi.

## Note tecniche
- **Device auto-detect**: gli script usano GPU se `torch.cuda.is_available()`, altrimenti CPU.
- **Study nuovo** (`hpo_tft_gpu`): NON eredita i 30 trial "crippled" del run CPU (`hpo_tft.db`).
- **Non-distruttivo**: output con suffisso `_gpu`; nulla dei risultati esistenti viene sovrascritto.
- **Miglioria vs versione CPU**: la predict del test parte dal *best checkpoint* (val_loss minimo),
  non dall'ultima epoca.
- Se emergono NaN con `16-mixed`, usa il default `32-true`.
