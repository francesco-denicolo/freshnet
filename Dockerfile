# =============================================================================
# TFT GPU run — AWS g4dn.2xlarge (NVIDIA T4, CUDA 12.1)
# Immagine riproducibile e isolata per il resourcing del TFT sulle 50K serie.
# Dati e output NON sono dentro l'immagine: si montano a runtime come volumi.
#
# Build:  docker build -t tft-gpu:latest .
# Run:    bash pipeline/run_docker_tft.sh all      (vedi pipeline/docker_tft_README.md)
# =============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python + tini (PID 1 per una chiusura pulita su SIGTERM, es. interruzione Spot)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip tini \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Dipendenze prima del codice (layer caching) -----------------------------
# torch con build CUDA 12.1 (le wheel includono le librerie CUDA/cuDNN runtime),
# poi le dipendenze pinnate del progetto.
COPY pipeline/requirements_tft_gpu.txt /app/requirements_tft_gpu.txt
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r /app/requirements_tft_gpu.txt

# --- Codice (solo script; dati/output montati a runtime) ---------------------
COPY pipeline/ /app/pipeline/

# Utente non-root con uid 1000 (= utente 'ubuntu' su EC2 => i volumi montati
# restano scrivibili senza problemi di permessi).
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data /app/pipeline/results \
    && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser

# Default: HPO ampia (hidden fino a 256), niente subsample (50K piene), fp32.
# Override a runtime con -e (vedi run_docker_tft.sh).
ENV PY=python \
    TFT_HIDDEN_CAP=256 \
    TFT_PRECISION=32-true

ENTRYPOINT ["tini", "--"]
CMD ["bash", "pipeline/run_cloud_tft.sh", "all"]
