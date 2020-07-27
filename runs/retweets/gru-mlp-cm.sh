#!/bin/zsh

MLFLOW_HOST=192.168.4.94
MLFLOW_PORT=1235
REMOTE_SERVER_URI="http://${MLFLOW_HOST}:${MLFLOW_PORT}"

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name retweets-split-1 \
  --run-name gru-mlp-cm \
  --batch-size 256 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/retweets/split_1 \
  --plots-dir ~/neural-tpps/plots/retweets_1/gru_mlp_cm \
  --time-scale 1e-3 \
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
  --decoder mlp-cm \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-embedding-constraint nonneg \
  --decoder-emb-dim 32 \
  --decoder-units-mlp 64 32 \
  --decoder-constraint-mlp nonneg \
  --decoder-activation-mlp gumbel \
  --decoder-activation-final-mlp softplus \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name retweets-split-2 \
  --run-name gru-mlp-cm \
  --batch-size 256 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/retweets/split_2 \
  --plots-dir ~/neural-tpps/plots/retweets_2/gru_mlp_cm \
  --time-scale 1e-3 \
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
  --decoder mlp-cm \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-embedding-constraint nonneg \
  --decoder-emb-dim 32 \
  --decoder-units-mlp 64 32 \
  --decoder-constraint-mlp nonneg \
  --decoder-activation-mlp gumbel \
  --decoder-activation-final-mlp softplus \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name retweets-split-3 \
  --run-name gru-mlp-cm \
  --batch-size 256 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/retweets/split_3 \
  --plots-dir ~/neural-tpps/plots/retweets_3/gru_mlp_cm \
  --time-scale 1e-3 \
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
  --decoder mlp-cm \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-embedding-constraint nonneg \
  --decoder-emb-dim 32 \
  --decoder-units-mlp 64 32 \
  --decoder-constraint-mlp nonneg \
  --decoder-activation-mlp gumbel \
  --decoder-activation-final-mlp softplus \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name retweets-split-4 \
  --run-name gru-mlp-cm \
  --batch-size 256 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/retweets/split_4 \
  --plots-dir ~/neural-tpps/plots/retweets_4/gru_mlp_cm \
  --time-scale 1e-3 \
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
  --decoder mlp-cm \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-embedding-constraint nonneg \
  --decoder-emb-dim 32 \
  --decoder-units-mlp 64 32 \
  --decoder-constraint-mlp nonneg \
  --decoder-activation-mlp gumbel \
  --decoder-activation-final-mlp softplus \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name retweets-split-5 \
  --run-name gru-mlp-cm \
  --batch-size 256 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/retweets/split_5 \
  --plots-dir ~/neural-tpps/plots/retweets_5/gru_mlp_cm \
  --time-scale 1e-3 \
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
  --decoder mlp-cm \
  --decoder-encoding learnable \
  --decoder-time-encoding relative \
  --decoder-embedding-constraint nonneg \
  --decoder-emb-dim 32 \
  --decoder-units-mlp 64 32 \
  --decoder-constraint-mlp nonneg \
  --decoder-activation-mlp gumbel \
  --decoder-activation-final-mlp softplus \
  --include-poisson True \
  \
