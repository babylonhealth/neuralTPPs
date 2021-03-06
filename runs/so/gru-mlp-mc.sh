#!/bin/zsh

MLFLOW_HOST=192.168.4.94
MLFLOW_PORT=1235
REMOTE_SERVER_URI="http://${MLFLOW_HOST}:${MLFLOW_PORT}"

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name so-split-1 \
  --run-name gru-mlp-mc \
  --batch-size 32 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/so/split_1 \
  --plots-dir ~/neural-tpps/plots/so_1/gru_mlp_mc \
  --time-scale 1e-5 \
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
  --experiment-name so-split-2 \
  --run-name gru-mlp-mc \
  --batch-size 32 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/so/split_2 \
  --plots-dir ~/neural-tpps/plots/so_2/gru_mlp_mc \
  --time-scale 1e-5 \
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
  --experiment-name so-split-3 \
  --run-name gru-mlp-mc \
  --batch-size 32 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/so/split_3 \
  --plots-dir ~/neural-tpps/plots/so_3/gru_mlp_mc \
  --time-scale 1e-5 \
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
  --experiment-name so-split-4 \
  --run-name gru-mlp-mc \
  --batch-size 32 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/so/split_4 \
  --plots-dir ~/neural-tpps/plots/so_4/gru_mlp_mc \
  --time-scale 1e-5 \
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
  --experiment-name so-split-5 \
  --run-name gru-mlp-mc \
  --batch-size 32 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/so/split_5 \
  --plots-dir ~/neural-tpps/plots/so_5/gru_mlp_mc \
  --time-scale 1e-5 \
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
