#!/bin/bash
# Overnight full-scale re-tune co-adaptation check (review risk #2).
# Tunes MLP-M5 on the itransformer basis (12-config search) over the FULL 50K
# series, re-trains all 13 imputer cells natively, and reports Kendall's W across
# imputers. Expected ~7-9 h. Result parquet: pipeline/results/retune_itransformer_W.parquet
# Monitor: tail -f pipeline/results/_log_retune_FULL50k.txt
cd /Users/utente/Desktop/FreshNetRetail
SUB=0 nohup freshnet/bin/python pipeline/retune_itransformer_W.py \
  > pipeline/results/_log_retune_FULL50k.txt 2>&1 &
echo "launched full-50K re-tune, PID $!"
echo "log: pipeline/results/_log_retune_FULL50k.txt"
