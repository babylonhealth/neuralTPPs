#!/bin/zsh


python scripts/train.py \
  --remote-server-uri "http://mlflow-tracking.neural-tpps.svc.cluster.local" \
  --experiment-name so-split-1 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 32 \
  --data-dir /home/air/project/neural-tpps/data \
  --load-from-dir /home/air/project/neural-tpps/data/baseline/so/split_1 \
  --plots-dir /home/air/project/neural-tpps/plots/so_1/gru_mlp_cm_tanh \
  --time-scale 1e-5 \
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
  --experiment-name so-split-2 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 32 \
  --data-dir /home/air/project/neural-tpps/data \
  --load-from-dir /home/air/project/neural-tpps/data/baseline/so/split_2 \
  --plots-dir /home/air/project/neural-tpps/plots/so_2/gru_mlp_cm_tanh \
  --time-scale 1e-5 \
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
  --experiment-name so-split-3 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 32 \
  --data-dir /home/air/project/neural-tpps/data \
  --load-from-dir /home/air/project/neural-tpps/data/baseline/so/split_3 \
  --plots-dir /home/air/project/neural-tpps/plots/so_3/gru_mlp_cm_tanh \
  --time-scale 1e-5 \
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
  --experiment-name so-split-4 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 32 \
  --data-dir /home/air/project/neural-tpps/data \
  --load-from-dir /home/air/project/neural-tpps/data/baseline/so/split_4 \
  --plots-dir /home/air/project/neural-tpps/plots/so_4/gru_mlp_cm_tanh \
  --time-scale 1e-5 \
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
  --experiment-name so-split-5 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 32 \
  --data-dir /home/air/project/neural-tpps/data \
  --load-from-dir /home/air/project/neural-tpps/data/baseline/so/split_5 \
  --plots-dir /home/air/project/neural-tpps/plots/so_5/gru_mlp_cm_tanh \
  --time-scale 1e-5 \
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
