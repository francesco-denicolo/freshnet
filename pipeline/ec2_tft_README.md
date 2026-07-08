# TFT resourcing su AWS EC2 (full 50K, niente subsample)

Obiettivo: rilanciare il TFT con capacità piena (GPU, `hidden` fino a 256, epoche piene)
su **tutte le 49.939 serie**, per integrarlo direttamente nella matrice a 155 celle
**senza alcun caveat di campionamento**. La macchina consigliata risolve il vero collo
di bottiglia — la RAM di sistema per costruire il `TimeSeriesDataSet` su 50K serie.

## 0. Istanza
- **`g4dn.2xlarge`** (8 vCPU, **32 GB RAM**, 1× NVIDIA T4 16 GB). ~$0.75/h on-demand, ~$0.23/h Spot.
- **AMI**: "Deep Learning Base/OSS AMI (Ubuntu 22.04)" o una Ubuntu 22.04 liscia.
- **Storage**: EBS gp3 da **50 GB**.
- **Security group**: SSH (22) aperto dal tuo IP.
- I nostri script sono **resumable** → lo Spot va bene (vedi §6).

## 1. Connessione
```bash
KEY=~/percorso/tua-chiave.pem            # chmod 400 la prima volta
HOST=ubuntu@<EC2_PUBLIC_IP>
ssh -i "$KEY" "$HOST"
# crea la struttura di cartelle sull'istanza:
mkdir -p ~/FreshNetRetail/data/completed_sales_622 ~/FreshNetRetail/pipeline/results
exit
```

## 2. Staging dati + script (da locale, ~1.1 GB)
Esegui dal **Mac**, nella root del repo (`/Users/utente/Desktop/FreshNetRetail`):
```bash
KEY=~/percorso/tua-chiave.pem
HOST=ubuntu@<EC2_PUBLIC_IP>
DEST=/home/ubuntu/FreshNetRetail
RS="rsync -avz --progress -e \"ssh -i $KEY\""

# dati base
eval $RS data/frn50k_train.parquet data/frn50k_eval.parquet $HOST:$DEST/data/
eval $RS pipeline/results/stratification.parquet $HOST:$DEST/pipeline/results/

# i 13 file imputer necessari
eval $RS data/completed_sales_622/{media_glob,media_cond,mediana_glob,mediana_cond,forward_fill,seasonal_naive,linear_interp,lgb,dlinear,saits,itransformer,timesnet,imputeformer}.parquet \
  $HOST:$DEST/data/completed_sales_622/

# script
eval $RS pipeline/31_hpo_tft_gpu.py pipeline/25_tft_full_training_gpu.py \
  pipeline/run_cloud_tft.sh pipeline/requirements_tft_gpu.txt $HOST:$DEST/pipeline/
```

## 3. Ambiente (venv deterministico, consigliato)
Sull'istanza:
```bash
cd ~/FreshNetRetail
python3 -m venv ~/tftenv && source ~/tftenv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r pipeline/requirements_tft_gpu.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# deve stampare: CUDA: True Tesla T4
```
(Se usi la Deep Learning AMI con conda, in alternativa: `source activate pytorch` e poi solo
`pip install pytorch-forecasting==1.4.0 lightning==2.6.0 optuna optuna-integration`. Il venv è
più robusto perché evita conflitti con il torch preinstallato.)

## 4. Run completo (full 50K, in tmux)
`tmux` tiene vivo il processo anche se cade la connessione SSH.
```bash
cd ~/FreshNetRetail
tmux new -s tft
source ~/tftenv/bin/activate
# SERIES_SUBSAMPLE non impostata => 0 => tutte le serie. HPO ampia (hidden fino a 256).
PY=python TFT_HIDDEN_CAP=256 bash pipeline/run_cloud_tft.sh all 2>&1 | tee tft.log
#  -> stacca la tmux con:  Ctrl-b  poi  d
#  -> riattacca con:       tmux attach -t tft
```
Fasi: (1) HPO → `hpo_tft_gpu_best.json`; (2) 14 celle → `{imp}__tft_gpu_test_per_series.parquet`.

## 5. Monitoraggio (in un'altra sessione SSH)
```bash
tail -f ~/FreshNetRetail/tft.log     # progressi
nvidia-smi                            # uso GPU
free -g                               # RAM (deve stare < 32 GB; picco atteso ~15-20)
```

## 6. Spot / interruzioni → resume
Tutto è **resumable**:
- Optuna riprende dallo studio `hpo_tft_gpu.db`;
- le celle già presenti (`*__tft_gpu_*.parquet`) vengono saltate.

Se usi **Spot**, così l'EBS sopravvive all'interruzione e puoi riprendere:
- Request type **persistent**, "Interruption behavior" = **stop** (non terminate).
- Alla ripartenza dell'istanza: riattacca/rilancia lo stesso comando della §4 → riprende da dove era.

Se usi **on-demand**: nessuna interruzione; ricordati solo di terminare a fine lavoro (§8).

## 7. Scarica i risultati + integra in locale
Dal **Mac**:
```bash
KEY=~/percorso/tua-chiave.pem ; HOST=ubuntu@<EC2_PUBLIC_IP> ; DEST=/home/ubuntu/FreshNetRetail
rsync -avz -e "ssh -i $KEY" "$HOST:$DEST/pipeline/results/*__tft_gpu_test_per_series.parquet" pipeline/results/
rsync -avz -e "ssh -i $KEY" "$HOST:$DEST/pipeline/results/hpo_tft_gpu_best.json" \
  "$HOST:$DEST/pipeline/results/hpo_tft_gpu_trials.parquet" pipeline/results/
# integrazione nella matrice + statistiche + figure:
freshnet/bin/python pipeline/integrate_tft_gpu.py
```

## 8. Teardown (IMPORTANTE per non pagare)
```bash
# on-demand: TERMINA l'istanza da console EC2 (o: aws ec2 terminate-instances --instance-ids i-xxxx)
# Spot persistent: cancella lo spot request E termina l'istanza.
```

## Stima costo/tempo
- Run ~5–10 h su g4dn.2xlarge → **~$4–8 on-demand**, **~$1.5–3 Spot**.
- EBS 50 GB per un giorno: trascurabile (~$0.15).

## Perché così è "pulito" per il paper
- **Niente subsample** (32 GB reggono le 50K) → il TFT si integra nella matrice a 49.939 serie
  come ogni altra cella, senza caveat di campionamento.
- **HPO piena** (hidden fino a 256) → risolve alla radice la critica "under-resourced".
- Cella `imputeformer × TFT` finalmente prodotta → grid completa (156 celle).
