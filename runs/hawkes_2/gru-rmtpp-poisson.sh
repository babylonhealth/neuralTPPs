#!/bin/zsh

MLFLOW_HOST=192.168.4.94
MLFLOW_PORT=1235
REMOTE_SERVER_URI="http://${MLFLOW_HOST}:${MLFLOW_PORT}"

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name hawkes-2-split-1 \
  --run-name gru-rmtpp-poisson \
  --mu 0.1 0.05 \
  --alpha 0.2 0. 0. 0.4 \
  --beta 1.0 1.0 1.0 2.0 \
  --hawkes-seed 0 \
  --seed 0 \
  --train-size 16384 \
  --val-size 4096 \
  --test-size 4096 \
  --batch-size 512 \
  --data-dir ~/neural-tpps/data \
  --plots-dir ~/neural-tpps/plots/hawkes_1_1/gru_rmtpp_poisson \
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
  --encoder-units-rnn 8 \
  --encoder-activation-final-mlp relu \
  --decoder rmtpp \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 8 \
  --decoder-units-rnn 8 \
  --decoder-units-mlp 8 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name hawkes-2-split-2 \
  --run-name gru-rmtpp-poisson \
  --mu 0.1 0.05 \
  --alpha 0.2 0. 0. 0.4 \
  --beta 1.0 1.0 1.0 2.0 \
  --hawkes-seed 25000 \
  --seed 25000 \
  --train-size 16384 \
  --val-size 4096 \
  --test-size 4096 \
  --batch-size 512 \
  --data-dir ~/neural-tpps/data \
  --plots-dir ~/neural-tpps/plots/hawkes_1_2/gru_rmtpp_poisson \
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
  --encoder-units-rnn 8 \
  --encoder-activation-final-mlp relu \
  --decoder rmtpp \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 8 \
  --decoder-units-rnn 8 \
  --decoder-units-mlp 8 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name hawkes-2-split-3 \
  --run-name gru-rmtpp-poisson \
  --mu 0.1 0.05 \
  --alpha 0.2 0. 0. 0.4 \
  --beta 1.0 1.0 1.0 2.0 \
  --hawkes-seed 50000 \
  --seed 50000 \
  --train-size 16384 \
  --val-size 4096 \
  --test-size 4096 \
  --batch-size 512 \
  --data-dir ~/neural-tpps/data \
  --plots-dir ~/neural-tpps/plots/hawkes_1_3/gru_rmtpp_poisson \
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
  --encoder-units-rnn 8 \
  --encoder-activation-final-mlp relu \
  --decoder rmtpp \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 8 \
  --decoder-units-rnn 8 \
  --decoder-units-mlp 8 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name hawkes-2-split-4 \
  --run-name gru-rmtpp-poisson \
  --mu 0.1 0.05 \
  --alpha 0.2 0. 0. 0.4 \
  --beta 1.0 1.0 1.0 2.0 \
  --hawkes-seed 75000 \
  --seed 75000 \
  --train-size 16384 \
  --val-size 4096 \
  --test-size 4096 \
  --batch-size 512 \
  --data-dir ~/neural-tpps/data \
  --plots-dir ~/neural-tpps/plots/hawkes_1_4/gru_rmtpp_poisson \
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
  --encoder-units-rnn 8 \
  --encoder-activation-final-mlp relu \
  --decoder rmtpp \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 8 \
  --decoder-units-rnn 8 \
  --decoder-units-mlp 8 \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name hawkes-2-split-5 \
  --run-name gru-rmtpp-poisson \
  --mu 0.1 0.05 \
  --alpha 0.2 0. 0. 0.4 \
  --beta 1.0 1.0 1.0 2.0 \
  --hawkes-seed 100000 \
  --seed 100000 \
  --train-size 16384 \
  --val-size 4096 \
  --test-size 4096 \
  --batch-size 512 \
  --data-dir ~/neural-tpps/data \
  --plots-dir ~/neural-tpps/plots/hawkes_1_5/gru_rmtpp_poisson \
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
  --encoder-units-rnn 8 \
  --encoder-activation-final-mlp relu \
  --decoder rmtpp \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-emb-dim 8 \
  --decoder-units-rnn 8 \
  --decoder-units-mlp 8 \
  --include-poisson True \
  \
