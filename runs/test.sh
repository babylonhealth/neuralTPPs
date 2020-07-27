#!/bin/zsh


python scripts/train.py \
  --remote-server-uri "http://mlflow-tracking.neural-tpps.svc.cluster.local" \
  --experiment-name test \
  --run-name gru-lnm \
  --train-size 2048 \
  --val-size 256 \
  --test-size 256 \
  --batch-size 256 \
  --data-dir /home/air/project/neural-tpps/data \
  --plots-dir /home/air/project/neural-tpps/plots/gru_lnm \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 501 \
  --patience 100 \
  --encoder gru \
  --encoder-encoding learnable_with_labels \
  --encoder-emb-dim 8 \
  --encoder-attn-activation softmax \
  --encoder-units-rnn 8 \
  --encoder-units-mlp 8 \
  --encoder-activation-final-mlp relu \
  --decoder log-normal-mixture \
  --decoder-encoding learnable_with_labels \
  --decoder-emb-dim 8 \
  --decoder-attn-activation softmax \
  --decoder-units-rnn 8 \
  --decoder-units-mlp 8 \
  --include-poisson False \
  --use-air True \
  --verbose \
  \
