#!/bin/zsh

MLFLOW_HOST=192.168.4.94
MLFLOW_PORT=1235
REMOTE_SERVER_URI="http://${MLFLOW_HOST}:${MLFLOW_PORT}"

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name ear-infection-split-1 \
  --run-name gru-rmtpp-poisson \
  --batch-size 512 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/synthea/ear_infection/split_1 \
  --plots-dir ~/neural-tpps/plots/ear_infection_1/gru_rmtpp_poisson \
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
  --decoder rmtpp \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 32 \
  --decoder-units-rnn 32 \
  --decoder-units-mlp 32 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name ear-infection-split-2 \
  --run-name gru-rmtpp-poisson \
  --batch-size 512 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/synthea/ear_infection/split_2 \
  --plots-dir ~/neural-tpps/plots/ear_infection_2/gru_rmtpp_poisson \
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
  --decoder rmtpp \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 32 \
  --decoder-units-rnn 32 \
  --decoder-units-mlp 32 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name ear-infection-split-3 \
  --run-name gru-rmtpp-poisson \
  --batch-size 512 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/synthea/ear_infection/split_3 \
  --plots-dir ~/neural-tpps/plots/ear_infection_3/gru_rmtpp_poisson \
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
  --decoder rmtpp \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 32 \
  --decoder-units-rnn 32 \
  --decoder-units-mlp 32 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name ear-infection-split-4 \
  --run-name gru-rmtpp-poisson \
  --batch-size 512 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/synthea/ear_infection/split_4 \
  --plots-dir ~/neural-tpps/plots/ear_infection_4/gru_rmtpp_poisson \
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
  --decoder rmtpp \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 32 \
  --decoder-units-rnn 32 \
  --decoder-units-mlp 32 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name ear-infection-split-5 \
  --run-name gru-rmtpp-poisson \
  --batch-size 512 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/synthea/ear_infection/split_5 \
  --plots-dir ~/neural-tpps/plots/ear_infection_5/gru_rmtpp_poisson \
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
  --decoder rmtpp \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 32 \
  --decoder-units-rnn 32 \
  --decoder-units-mlp 32 \
  --include-poisson True \
  \
