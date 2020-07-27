#!/bin/zsh


python scripts/train.py \
  --remote-server-uri "http://mlflow-tracking.neural-tpps.svc.cluster.local" \
  --experiment-name financial-split-1 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 2 \
  --data-dir /home/air/project/neural-tpps/data \
  --load-from-dir /home/air/project/neural-tpps/data/baseline/financial/split_1 \
  --plots-dir /home/air/project/neural-tpps/plots/financial_1/gru_mlp_cm_tanh \
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
  --experiment-name financial-split-2 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 2 \
  --data-dir /home/air/project/neural-tpps/data \
  --load-from-dir /home/air/project/neural-tpps/data/baseline/financial/split_2 \
  --plots-dir /home/air/project/neural-tpps/plots/financial_2/gru_mlp_cm_tanh \
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
  --experiment-name financial-split-3 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 2 \
  --data-dir /home/air/project/neural-tpps/data \
  --load-from-dir /home/air/project/neural-tpps/data/baseline/financial/split_3 \
  --plots-dir /home/air/project/neural-tpps/plots/financial_3/gru_mlp_cm_tanh \
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
  --experiment-name financial-split-4 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 2 \
  --data-dir /home/air/project/neural-tpps/data \
  --load-from-dir /home/air/project/neural-tpps/data/baseline/financial/split_4 \
  --plots-dir /home/air/project/neural-tpps/plots/financial_4/gru_mlp_cm_tanh \
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
  --experiment-name financial-split-5 \
  --run-name gru-mlp-cm-tanh \
  --batch-size 2 \
  --data-dir /home/air/project/neural-tpps/data \
  --load-from-dir /home/air/project/neural-tpps/data/baseline/financial/split_5 \
  --plots-dir /home/air/project/neural-tpps/plots/financial_5/gru_mlp_cm_tanh \
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
