#!/bin/zsh

MLFLOW_HOST=192.168.4.94
MLFLOW_PORT=1235
REMOTE_SERVER_URI="http://${MLFLOW_HOST}:${MLFLOW_PORT}"


python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --experiment-name mimic2-split-1 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 65 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/mimic2/split_1 \
  --plots-dir ~/neural-tpps/plots/mimic2_1/gru_mlp_cm_tanh \
  --save-model-freq 250 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 501 \
  --patience 500 \
  --encoder gru \
  --encoder-encoding concatenate \
  --encoder-emb-dim 8 \
  --encoder-units-rnn 64 \
  --encoder-units-mlp 64 \
  --encoder-activation-final-mlp relu \
  --decoder mlp-cm \
  --decoder-encoding times_only \
  --decoder-embedding-constraint nonneg \
  --decoder-emb-dim 8 \
  --decoder-units-mlp 64 64 \
  --decoder-constraint-rnn nonneg \
  --decoder-constraint-mlp nonneg \
  --decoder-activation-rnn tanh \
  --decoder-activation-mlp tanh \
  --decoder-activation-final-mlp softplus \
  --decoder-do-zero-subtraction False \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri "http://mlflow-tracking.neural-tpps.svc.cluster.local" \
  --experiment-name mimic2-split-2 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 65 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/mimic2/split_2 \
  --plots-dir ~/neural-tpps/plots/mimic2_2/gru_mlp_cm_tanh \
  --save-model-freq 250 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 501 \
  --patience 500 \
  --encoder gru \
  --encoder-encoding concatenate \
  --encoder-emb-dim 8 \
  --encoder-units-rnn 64 \
  --encoder-units-mlp 64 \
  --encoder-activation-final-mlp relu \
  --decoder mlp-cm \
  --decoder-encoding times_only \
  --decoder-embedding-constraint nonneg \
  --decoder-emb-dim 8 \
  --decoder-units-mlp 64 64 \
  --decoder-constraint-rnn nonneg \
  --decoder-constraint-mlp nonneg \
  --decoder-activation-rnn tanh \
  --decoder-activation-mlp tanh \
  --decoder-activation-final-mlp softplus \
  --decoder-do-zero-subtraction False \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri "http://mlflow-tracking.neural-tpps.svc.cluster.local" \
  --experiment-name mimic2-split-3 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 65 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/mimic2/split_3 \
  --plots-dir ~/neural-tpps/plots/mimic2_3/gru_mlp_cm_tanh \
  --save-model-freq 250 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 501 \
  --patience 500 \
  --encoder gru \
  --encoder-encoding concatenate \
  --encoder-emb-dim 8 \
  --encoder-units-rnn 64 \
  --encoder-units-mlp 64 \
  --encoder-activation-final-mlp relu \
  --decoder mlp-cm \
  --decoder-encoding times_only \
  --decoder-embedding-constraint nonneg \
  --decoder-emb-dim 8 \
  --decoder-units-mlp 64 64 \
  --decoder-constraint-rnn nonneg \
  --decoder-constraint-mlp nonneg \
  --decoder-activation-rnn tanh \
  --decoder-activation-mlp tanh \
  --decoder-activation-final-mlp softplus \
  --decoder-do-zero-subtraction False \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri "http://mlflow-tracking.neural-tpps.svc.cluster.local" \
  --experiment-name mimic2-split-4 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 65 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/mimic2/split_4 \
  --plots-dir ~/neural-tpps/plots/mimic2_4/gru_mlp_cm_tanh \
  --save-model-freq 250 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 501 \
  --patience 500 \
  --encoder gru \
  --encoder-encoding concatenate \
  --encoder-emb-dim 8 \
  --encoder-units-rnn 64 \
  --encoder-units-mlp 64 \
  --encoder-activation-final-mlp relu \
  --decoder mlp-cm \
  --decoder-encoding times_only \
  --decoder-embedding-constraint nonneg \
  --decoder-emb-dim 8 \
  --decoder-units-mlp 64 64 \
  --decoder-constraint-rnn nonneg \
  --decoder-constraint-mlp nonneg \
  --decoder-activation-rnn tanh \
  --decoder-activation-mlp tanh \
  --decoder-activation-final-mlp softplus \
  --decoder-do-zero-subtraction False \
  --include-poisson True \
  \

python scripts/train.py \
  --remote-server-uri "http://mlflow-tracking.neural-tpps.svc.cluster.local" \
  --experiment-name mimic2-split-5 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 65 \
  --data-dir ~/neural-tpps/data \
  --load-from-dir ~/neural-tpps/data/baseline/mimic2/split_5 \
  --plots-dir ~/neural-tpps/plots/mimic2_5/gru_mlp_cm_tanh \
  --save-model-freq 250 \
  --lr-rate-init 1e-2 \
  --lr-poisson-rate-init 1e-2 \
  --lr-scheduler-warmup 10 \
  --train-epochs 501 \
  --patience 500 \
  --encoder gru \
  --encoder-encoding concatenate \
  --encoder-emb-dim 8 \
  --encoder-units-rnn 64 \
  --encoder-units-mlp 64 \
  --encoder-activation-final-mlp relu \
  --decoder mlp-cm \
  --decoder-encoding times_only \
  --decoder-embedding-constraint nonneg \
  --decoder-emb-dim 8 \
  --decoder-units-mlp 64 64 \
  --decoder-constraint-rnn nonneg \
  --decoder-constraint-mlp nonneg \
  --decoder-activation-rnn tanh \
  --decoder-activation-mlp tanh \
  --decoder-activation-final-mlp softplus \
  --decoder-do-zero-subtraction False \
  --include-poisson True \
  \
