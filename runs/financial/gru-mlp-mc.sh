#!/bin/zsh

MLFLOW_HOST=192.168.4.94
MLFLOW_PORT=1235
REMOTE_SERVER_URI="http://${MLFLOW_HOST}:${MLFLOW_PORT}"

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name financial-split-1 \
  --run-name gru-mlp-mc \
  --batch-size 2 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/financial/split_1 \
  --plots-dir ~/neural-tpps/plots/financial_1/gru_mlp_mc \
  --save-model-freq 100 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 1001 \
  --patience 100 \
  --encoder gru \
  --encoder-encoding learnable_with_labels \
  --encoder-time-encoding absolute \
  --encoder-emb-dim 32 \
  --encoder-units-rnn 32 \
  --encoder-activation-final-mlp relu \
  --decoder mlp-mc \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 32 \
  --decoder-units-mlp 64 32 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name financial-split-2 \
  --run-name gru-mlp-mc \
  --batch-size 2 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/financial/split_2 \
  --plots-dir ~/neural-tpps/plots/financial_2/gru_mlp_mc \
  --save-model-freq 100 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 1001 \
  --patience 100 \
  --encoder gru \
  --encoder-encoding learnable_with_labels \
  --encoder-time-encoding absolute \
  --encoder-emb-dim 32 \
  --encoder-units-rnn 32 \
  --encoder-activation-final-mlp relu \
  --decoder mlp-mc \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 32 \
  --decoder-units-mlp 64 32 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name financial-split-3 \
  --run-name gru-mlp-mc \
  --batch-size 2 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/financial/split_3 \
  --plots-dir ~/neural-tpps/plots/financial_3/gru_mlp_mc \
  --save-model-freq 100 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 1001 \
  --patience 100 \
  --encoder gru \
  --encoder-encoding learnable_with_labels \
  --encoder-time-encoding absolute \
  --encoder-emb-dim 32 \
  --encoder-units-rnn 32 \
  --encoder-activation-final-mlp relu \
  --decoder mlp-mc \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 32 \
  --decoder-units-mlp 64 32 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name financial-split-4 \
  --run-name gru-mlp-mc \
  --batch-size 2 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/financial/split_4 \
  --plots-dir ~/neural-tpps/plots/financial_4/gru_mlp_mc \
  --save-model-freq 100 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 1001 \
  --patience 100 \
  --encoder gru \
  --encoder-encoding learnable_with_labels \
  --encoder-time-encoding absolute \
  --encoder-emb-dim 32 \
  --encoder-units-rnn 32 \
  --encoder-activation-final-mlp relu \
  --decoder mlp-mc \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 32 \
  --decoder-units-mlp 64 32 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name financial-split-5 \
  --run-name gru-mlp-mc \
  --batch-size 2 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/financial/split_5 \
  --plots-dir ~/neural-tpps/plots/financial_5/gru_mlp_mc \
  --save-model-freq 100 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 1001 \
  --patience 100 \
  --encoder gru \
  --encoder-encoding learnable_with_labels \
  --encoder-time-encoding absolute \
  --encoder-emb-dim 32 \
  --encoder-units-rnn 32 \
  --encoder-activation-final-mlp relu \
  --decoder mlp-mc \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 32 \
  --decoder-units-mlp 64 32 \
  --include-poisson True \
  \
