"""Genera colab_tft_gpu.ipynb — notebook Colab (T4 free) per il resourcing TFT su GPU.
Run: freshnet/bin/python pipeline/make_colab_notebook.py"""
import json, os

def md(*lines):  return {"cell_type": "markdown", "metadata": {}, "source": [l if l.endswith("\n") else l+"\n" for l in lines]}
def code(*lines): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l if l.endswith("\n") else l+"\n" for l in lines]}

cells = []

cells.append(md(
"# TFT resourcing su GPU — Colab (T4 free)",
"",
"Rilancia il TFT **senza** il cap di memoria `hidden<=32` che lo handicappava sul Mac 16 GB.",
"Su T4 la VRAM è dedicata alla GPU, quindi `hidden` fino a 128 gira senza problemi.",
"",
"**Prima di iniziare:**",
"1. `Runtime > Change runtime type > T4 GPU`.",
"2. Su Google Drive crea la cartella `MyDrive/tft_gpu/` e caricaci (una volta sola, ~1.8 GB):",
"   - `data/frn50k_train.parquet`, `data/frn50k_eval.parquet`, `data/stratification.parquet`",
"   - `data/completed_sales_622/` con i 13 file imputer necessari (tutti tranne `no_imp`)",
"   - `scripts/31_hpo_tft_gpu.py`, `scripts/25_tft_full_training_gpu.py`",
"3. Esegui le celle in ordine. **È tutto resumable**: se la sessione si stacca, riapri e",
"   ri-esegui `Run all` — lo studio Optuna riprende dal DB e le celle già fatte vengono saltate.",
))

cells.append(md("## 1. Verifica GPU"))
cells.append(code(
"import torch, subprocess",
"print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)",
"assert torch.cuda.is_available(), 'Nessuna GPU: Runtime > Change runtime type > T4 GPU'",
"print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)",
))

cells.append(md("## 2. Dipendenze (pinned)"))
cells.append(code(
"# pytorch-forecasting porta con sé lightning; pin per compatibilità con gli script.",
"!pip -q install pytorch-forecasting==1.4.0 lightning==2.6.0 optuna optuna-integration pyarrow 2>&1 | tail -3",
"import importlib, pytorch_forecasting, lightning",
"print('pytorch-forecasting', pytorch_forecasting.__version__, '| lightning', lightning.__version__)",
))

cells.append(md("## 3. Monta Drive e prepara la struttura di lavoro",
"I dati vanno su disco locale veloce (`/content`); gli output piccoli si sincronizzano su Drive."))
cells.append(code(
"import os, shutil, glob",
"from google.colab import drive",
"drive.mount('/content/drive')",
"",
"DRIVE = '/content/drive/MyDrive/tft_gpu'          # <-- cartella che hai creato su Drive",
"ROOT  = '/content/FreshNetRetail'",
"assert os.path.isdir(DRIVE), f'Manca {DRIVE} su Drive (vedi istruzioni cella 1)'",
"os.makedirs(f'{ROOT}/data/completed_sales_622', exist_ok=True)",
"os.makedirs(f'{ROOT}/pipeline/results', exist_ok=True)",
"os.makedirs(f'{DRIVE}/results', exist_ok=True)     # target di sincronizzazione output",
"",
"# --- copia dati e script da Drive al disco locale ---",
"for fn in ['frn50k_train.parquet', 'frn50k_eval.parquet']:",
"    shutil.copy2(f'{DRIVE}/data/{fn}', f'{ROOT}/data/{fn}')",
"shutil.copy2(f'{DRIVE}/data/stratification.parquet', f'{ROOT}/pipeline/results/stratification.parquet')",
"for f in glob.glob(f'{DRIVE}/data/completed_sales_622/*.parquet'):",
"    shutil.copy2(f, f'{ROOT}/data/completed_sales_622/{os.path.basename(f)}')",
"for s in ['31_hpo_tft_gpu.py', '25_tft_full_training_gpu.py']:",
"    shutil.copy2(f'{DRIVE}/scripts/{s}', f'{ROOT}/pipeline/{s}')",
"print('imputer files:', len(glob.glob(f'{ROOT}/data/completed_sales_622/*.parquet')))",
"print('staging OK')",
))

cells.append(md("## 4. Budget T4 + ripristino output già prodotti (resume)",
"Budget ridotto per stare nei limiti di tempo del free tier, ma **molto** oltre il cap `hidden<=32`.",
"Se una sessione precedente ha già prodotto output su Drive, li riportiamo in locale così vengono saltati."))
cells.append(code(
"os.environ['SERIES_SUBSAMPLE'] = '8000'  # T4 free 12.7GB: sottocampione stratificato (0=tutte -> OOM)",
"os.environ['TFT_HIDDEN_CAP'] = '128'      # T4: hidden fino a 128 (vs 32 sul Mac)",
"os.environ['TFT_N_TRIALS']   = '30'       # HPO trials",
"os.environ['TFT_MAX_EPOCHS'] = '12'",
"os.environ['TFT_PATIENCE']   = '3'",
"os.environ['TFT_MAX_TRAIN']  = '200000'   # window subsample per epoca",
"os.environ['TFT_PRECISION']  = '32-true'  # 16-mixed NON funziona col TFT (overflow fp16 nel mask attention)",
"",
"# ripristina da Drive gli output già fatti (HPO db/json + parquet celle)",
"for f in glob.glob(f'{DRIVE}/results/*'):",
"    dst = f'{ROOT}/pipeline/results/{os.path.basename(f)}'",
"    if not os.path.exists(dst):",
"        shutil.copy2(f, dst)",
"print('ripristinati da Drive:', [os.path.basename(f) for f in glob.glob(f'{DRIVE}/results/*')])",
"",
"def sync_to_drive(patterns):",
"    for pat in patterns:",
"        for f in glob.glob(f'{ROOT}/pipeline/results/{pat}'):",
"            shutil.copy2(f, f'{DRIVE}/results/{os.path.basename(f)}')",
))

cells.append(md("## 5. HPO ampia (no cap hidden=32)",
"Puoi rilanciare questa cella dopo una disconnessione: Optuna riprende dal DB su Drive."))
cells.append(code(
"import subprocess, sys",
"r = subprocess.run([sys.executable, 'pipeline/31_hpo_tft_gpu.py'], cwd=ROOT)",
"# salva subito i risultati HPO su Drive (db, best.json, trials)",
"sync_to_drive(['hpo_tft_gpu*.json', 'hpo_tft_gpu*.parquet', 'hpo_tft_gpu*.db'])",
"print('HPO exit', r.returncode)",
"import json",
"print(json.load(open(f'{ROOT}/pipeline/results/hpo_tft_gpu_best.json')))",
))

cells.append(md("## 6. Training delle 14 celle (con sync per cella)",
"Ogni cella completata viene copiata su Drive **subito**, così una disconnessione non perde il lavoro fatto.",
"Le celle già presenti vengono saltate dallo script."))
cells.append(code(
"import subprocess, sys, time",
"IMPUTERS = ['no_imp','media_glob','media_cond','mediana_glob','mediana_cond',",
"            'forward_fill','seasonal_naive','linear_interp','lgb',",
"            'dlinear','saits','itransformer','timesnet','imputeformer']",
"for imp in IMPUTERS:",
"    out = f'{ROOT}/pipeline/results/{imp}__tft_gpu_test_per_series.parquet'",
"    if os.path.exists(out):",
"        print(f'[skip] {imp} già fatto'); continue",
"    print(f'\\n===== {imp} x TFT =====', time.strftime('%H:%M:%S'))",
"    subprocess.run([sys.executable, 'pipeline/25_tft_full_training_gpu.py', imp], cwd=ROOT)",
"    sync_to_drive([f'{imp}__tft_gpu_test_per_series.parquet'])   # sync immediato",
"print('\\nFATTO. Celle prodotte:', len(glob.glob(f'{ROOT}/pipeline/results/*__tft_gpu_test_per_series.parquet')))",
))

cells.append(md("## 7. Fine — scarica i risultati",
"Tutti gli output sono in `MyDrive/tft_gpu/results/`. Scaricali sul Mac e lancia in locale:",
"`freshnet/bin/python pipeline/integrate_tft_gpu.py` per integrarli nella matrice."))
cells.append(code(
"import glob, os",
"print('Su Drive (MyDrive/tft_gpu/results/):')",
"for f in sorted(glob.glob(f'{DRIVE}/results/*')):",
"    print('  ', os.path.basename(f), f'{os.path.getsize(f)/1e6:.1f} MB')",
))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join(os.path.dirname(__file__), 'colab_tft_gpu.ipynb')
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print('wrote', out, '|', len(cells), 'cells')
