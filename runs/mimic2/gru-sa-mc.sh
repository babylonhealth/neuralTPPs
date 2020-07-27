#!/bin/zsh

MLFLOW_HOST=192.168.4.94
MLFLOW_PORT=1235
REMOTE_SERVER_URI="http://${MLFLOW_HOST}:${MLFLOW_PORT}"


python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name mimic2-split-1 \
  --run-name gru-sa-mc-softmax \
  --batch-size 65 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/mimic2/split_1 \
  --plots-dir ~/neural-tpps/plots/mimic2_1/gru_sa_mc \
  --save-model-freq 100 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 1001 \
  --patience 100 \
  --encoder gru \
  --encoder-encoding learnable_with_labels \
  --encoder-time-encoding absolute \
  --encoder-emb-dim 8 \
  --encoder-activation-final-mlp relu \
  --encoder-units-rnn 8 \
  --decoder selfattention-mc \
  --decoder-encoding learnable \
  --decoder-time-encoding absolute \
  --decoder-emb-dim 8 \
  --decoder-attn-activation softmax \
  --decoder-units-rnn 8 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name mimic2-split-2 \
  --run-name gru-sa-mc-softmax \
  --batch-size 65 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/mimic2/split_2 \
  --plots-dir ~/neural-tpps/plots/mimic2_2/gru_sa_mc \
  --save-model-freq 100 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 1001 \
  --patience 100 \
  --encoder gru \
  --encoder-encoding learnable_with_labels \
  --encoder-time-encoding absolute \
  --encoder-emb-dim 8 \
  --encoder-activation-final-mlp relu \
  --encoder-units-rnn 8 \
  --decoder selfattention-mc \
  --decoder-encoding learnable \
  --decoder-time-encoding absolute \
  --decoder-emb-dim 8 \
  --decoder-attn-activation softmax \
  --decoder-units-rnn 8 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name mimic2-split-3 \
  --run-name gru-sa-mc-softmax \
  --batch-size 65 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/mimic2/split_3 \
  --plots-dir ~/neural-tpps/plots/mimic2_3/gru_sa_mc \
  --save-model-freq 100 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 1001 \
  --patience 100 \
  --encoder gru \
  --encoder-encoding learnable_with_labels \
  --encoder-time-encoding absolute \
  --encoder-emb-dim 8 \
  --encoder-activation-final-mlp relu \
  --encoder-units-rnn 8 \
  --decoder selfattention-mc \
  --decoder-encoding learnable \
  --decoder-time-encoding absolute \
  --decoder-emb-dim 8 \
  --decoder-attn-activation softmax \
  --decoder-units-rnn 8 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name mimic2-split-4 \
  --run-name gru-sa-mc-softmax \
  --batch-size 65 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/mimic2/split_4 \
  --plots-dir ~/neural-tpps/plots/mimic2_4/gru_sa_mc \
  --save-model-freq 100 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 1001 \
  --patience 100 \
  --encoder gru \
  --encoder-encoding learnable_with_labels \
  --encoder-time-encoding absolute \
  --encoder-emb-dim 8 \
  --encoder-activation-final-mlp relu \
  --encoder-units-rnn 8 \
  --decoder selfattention-mc \
  --decoder-encoding learnable \
  --decoder-time-encoding absolute \
  --decoder-emb-dim 8 \
  --decoder-attn-activation softmax \
  --decoder-units-rnn 8 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name mimic2-split-5 \
  --run-name gru-sa-mc-softmax \
  --batch-size 65 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/mimic2/split_5 \
  --plots-dir ~/neural-tpps/plots/mimic2_5/gru_sa_mc \
  --save-model-freq 100 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 1001 \
  --patience 100 \
  --encoder gru \
  --encoder-encoding learnable_with_labels \
  --encoder-time-encoding absolute \
  --encoder-emb-dim 8 \
  --encoder-activation-final-mlp relu \
  --encoder-units-rnn 8 \
  --decoder selfattention-mc \
  --decoder-encoding learnable \
  --decoder-time-encoding absolute \
  --decoder-emb-dim 8 \
  --decoder-attn-activation softmax \
  --decoder-units-rnn 8 \
  --include-poisson True \
  \
