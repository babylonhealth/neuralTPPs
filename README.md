# Neural Temporal Point Processes (Neural TPPs)

We present in this repository code to train, evaluate and visualise (multi-class) Temporal Point Processes (TPPs), as described in our paper:

Joseph Enguehard, Dan Busbridge, Adam Bozson, Claire Woodcock, Nils Y. Hammerla, [Neural Temporal Point Processes For Modelling Electronic Health Records](https://arxiv.org/abs/2007.13794)

## Contact
+ Joseph Enguehard [joseph.enguehard@babylonhealth.com](mailto:joseph.enguehard@babylonhealth.com)

## The aim of this work
We want to allow Machine Learning (ML) to use Electronic Health Records (EHRs) in such a way that we can:
+ *Forecast*, i.e. predict the future health interactions of a patient given their medical history,
+ *Impute/interpolate*, i.e. highlight and determine missing data in a patient EHR,
+ *Represent*, i.e. provide representations for EHRs so they can be searched/retrieved semantically/efficiently.

## Brief description of our models
Our Neural TPP models are integrated into an encoder-decoder architecture framework.
Our encoders can be found in `tpp/models/encoders`, and our decoders in `tpp/models/decoders`.
Our encoders inherit from the abstract class defined in `tpp/models/encoders/base/encoder.py`.
Our decoders inherit from the abstract class defined in `tpp/models/decoders/base/decoder.py`.
These encoders and decoders are combined into a single model in `tpp/model/enc_dec.py`, where the intensity function and the likelihood are computed.
Our pre-trained models can be found here: https://drive.google.com/drive/folders/1LTkErAAEy7ZjX9HuRDMBb8PBzsqtHotR?usp=sharing.

## Data
Our preprocessed data can be found here: https://drive.google.com/drive/folders/1KZXUofQ6_jsUzRK4Oh049TVuUe5_ULGS?usp=sharing.
We do not provide the data for Hawkes (dependent) and Hawkes (independent) as they are deterministically generated during training time.

## Cite
Please cite our paper if you use this code in your own work:
```
@misc{enguehard2020neural,
    title={Neural Temporal Point Processes For Modelling Electronic Health Records},
    author={Joseph Enguehard and Dan Busbridge and Adam Bozson and Claire Woodcock and Nils Y. Hammerla},
    year={2020},
    eprint={2007.13794},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Relevant reading
### Lecture notes
+ [Lectures on the Poisson Process](https://arxiv.org/pdf/1806.00221.pdf).
+ [Temporal Point Processes and the Conditional Intensity Function](http://www.math.kit.edu/stoch/~last/seite/lectures_on_the_poisson_process/media/lastpenrose2017.pdf).

### Papers
+ [Recurrent Marked Temporal Point Processes: Embedding Event History to Vector](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf) - RNN encoder with simple exponential decay in time of intensity. Analytic likelihood, label classes not conditioned on time.
+ [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328).
+ [Fully Neural Network based Model for General Temporal Point Processes](https://arxiv.org/abs/1905.09690).
+ [Intensity-Free Learning of Temporal Point Processes](https://arxiv.org/abs/1909.12127).

## Development
### Installation

In order to use this repository, you need to create an environment:

```shell script
conda env create
source activate env_tpp
pip install -e .
```
You'll also need to install tick.
Depending on your machine there are number of different ways this might work.
See the commented out lines in the `environment.yml` for some examples.
We are aiming to remove the tick dependency in the future.

### Running
For any of the following (including starting MLflow) you will need to set up the python environment correctly.
```
source activate env_tpp
```

### Running (MLflow turned off, e.g. for debugging)
```
python scripts/train.py \
  --no-mlflow --model {model_name} \
  --param1 {value1} ...
```
### Running (Local MLflow)
In order to run experiments using MLflow, you need to follow these steps:

Set and create directories for the backend store and artifact root.
```
MLFLOW_BACKEND_STORE=~/mlflow/tpp/backend-store                  # For example
MLFLOW_DEFAULT_ARTIFACT_ROOT=~/mlflow/tpp/default-artifact-root  # For example
mkdir -p $MLFLOW_BACKEND_STORE 
mkdir -p $MLFLOW_DEFAULT_ARTIFACT_ROOT
```
Choose a host location.
```
MLFLOW_HOST=0.0.0.0
MLFLOW_PORT=5000
```
Start MLflow
```
mlflow server \
  --backend-store-uri $MLFLOW_BACKEND_STORE \
  --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT \
  --host $MLFLOW_HOST \
  --port $MLFLOW_PORT
```
Run the experiment.
```
REMOTE_SERVER_URI="http://${MLFLOW_HOST}:${MLFLOW_PORT}"
python scripts/train.py \
  --remote-server-uri $REMOTE_SERVER_URI \
  --encoder {encoder name} \
  --encoder_param1 {value} \
  --decoder {decoder name} \
  --decoder_param1 {value} \
```

### General usage - training a new model
```
usage: scripts/train.py [-h] [--seed SEED] [--padding-id PADDING_ID] [--disable-cuda]
                [--verbose] [--mu N [N ...]] [--alpha N [N ...]]
                [--beta N [N ...]] [--marks MARKS] [--hawkes-seed HAWKES_SEED]
                [--window WINDOW] [--train-size TRAIN_SIZE] [--val-size VAL_SIZE] 
                [--test-size TEST_SIZE]
                [--include-poisson INCLUDE_POISSON] [--batch-size BATCH_SIZE]
                [--train-epochs TRAIN_EPOCHS] [--use-coefficients USE_COEFFICIENTS]
                [--multi-labels MULTI_LABELS] [--time-scale TIME_SCALE]
                [--lr-rate-init LR_RATE_INIT] [--lr-poisson-rate-init LR_POISSON_RATE_INIT]
                [--lr-scheduler {plateau,step,milestones,cos,findlr,noam,clr,calr}]
                [--lr-scheduler-patience LR_SCHEDULER_PATIENCE]
                [--lr-scheduler-step-size LR_SCHEDULER_STEP_SIZE]
                [--lr-scheduler-gamma LR_SCHEDULER_GAMMA]
                [--lr-scheduler-warmup LR_SCHEDULER_WARMUP]
                [--patience PATIENCE]
                [--loss-relative-tolerance LOSS_RELATIVE_TOLERANCE]
                [--mu-cheat MU_CHEAT]
                [--encoder {gru,identity,mlp-fixed,mlp-variable,stub,selfattention}]
                [--encoder-history-size ENCODER_HISTORY_SIZE]
                [--encoder-emb-dim ENCODER_EMB_DIM]
                [--encoder-encoding {times_only,marks_only,concatenate,temporal,
                    learnable,temporal_with_labels,learnable_with_labels}]
                [--encoder-time-encoding {absolute, relative}]
                [--encoder-temporal-scaling ENCODER_TEMPORAL_SCALING]
                [--encoder-embedding-constraint {None, nonneg, sigmoid, softplus}]
                [--encoder-units-mlp N [N ...]]
                [--encoder-activation-mlp ENCODER_ACTIVATION_MLP]
                [--encoder-dropout-mlp ENCODER_DROPOUT_MLP]
                [--encoder-constraint-mlp {None, nonneg, sigmoid, softplus}]
                [--encoder-activation-final-mlp ENCODER_ACTIVATION_FINAL_MLP]
                [--encoder-attn-activation {identity,softmax,sigmoid}]
                [--encoder-dropout-rnn ENCODER_DROPOUT_RNN]
                [--encoder-layers-rnn ENCODER_LAYERS_RNN]
                [--encoder-units-rnn ENCODER_UNITS_RNN]
                [--encoder-n-heads ENCODER_N_HEADS]
                [--encoder-constraint-rnn {None, nonneg, sigmoid, softplus}]
                [--decoder {conditional-poisson,conditional-poisson-cm,hawkes,log-normal-mixture,
                    mlp-cm,mlp-mc,poisson,rmtpp,rmtpp-cm,selfattention-cm,selfattention-mc}]
                [--decoder-mc-prop-est DECODER_MC_PROP_EST]
                [--decoder-model-log-cm DECODER_MODEL_LOG_CM]
                [--decoder-do-zero-subtraction DECODER_DO_ZERO_SUBTRACTION]
                [--decoder-emb-dim DECODER_EMB_DIM]
                [--decoder-encoding {times_only,marks_only,concatenate,temporal,
                    learnable,temporal_with_labels,learnable_with_labels}]
                [--decoder-time-encoding {absolute, relative}]
                [--decoder-temporal-scaling DECODER_TEMPORAL_SCALING]
                [--decoder-embedding-constraint {None, nonneg, sigmoid, softplus}]
                [--decoder-units-mlp N [N ...]]
                [--decoder-activation-mlp DECODER_ACTIVATION_MLP]
                [--decoder-mlp-dropout DECODER_MLP_DROPOUT]
                [--decoder-activation-mlp DECODER_ACTIVATION_FINAL_MLP]
                [--decoder-constraint-mlp {None, nonneg, sigmoid, softplus}]
                [--decoder-attn-activation {identity,softmax,sigmoid}]
                [--decoder-activation-rnn DECODER_ACTIVATION_RNN]
                [--decoder-dropout-rnn DECODER_DROPOUT_RNN]
                [--decoder-layers-rnn DECODER_LAYERS_RNN]
                [--decoder-rnn-units DECODER_RNN_UNITS]
                [--decoder-n-heads DECODER_N_HEADS]
                [--decoder-constraint-rnn {None, nonneg, sigmoid, softplus}] 
                [--decoder-n-mixture DECODER_N_MIXTURE] [--no-mlflow]
                [--experiment-name EXPERIMENT_NAME] [--run-name RUN_NAME]
                [--remote-server-uri REMOTE_SERVER_URI]
                [--logging-frequency LOGGING_FREQUENCY]
                [--save-model-freq SAVE_MODEL_FREQ]
                [--eval-metrics EVAL_METRICS]
                [--eval-metrics-per-class EVAL_METRICS_PER_CLASS]
                [--load-from-dir LOAD_FROM_DIR]
                [--data-dir DATA_DIR]
                [--plots-dir PLOTS_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           The random seed.
  --padding-id PADDING_ID
                        The value used in the temporal sequences to indicate a
                        non-event.
  --disable-cuda        Disable CUDA
  --verbose             If `True`, prints all the things.
  --mu N [N ...]        The baseline intensity for the data generator.
  --alpha N [N ...]     The event parameter for the data generator. This will
                        be reshaped into a matrix the size of [mu,mu].
  --beta N [N ...]      The decay parameter for the data generator. This will
                        be reshaped into a matrix the size of [mu,mu].
  --marks MARKS         Generate a process with this many marks. Defaults to
                        `None`. If this is set to an integer, it will override
                        `alpha`, `beta` and `mu` with randomly generated
                        values corresponding to the number of requested marks.
  --hawkes-seed HAWKES_SEED
                        The random seed for generating the `alpha, `beta` and
                        `mu` if `marks` is not `None`.
  --window WINDOW       The window of the simulated process.py. Also taken as
                        the window of any parametric Hawkes model if chosen.
  --train-size TRAIN_SIZE
                        The number of unique sequences in each of the train 
                        dataset.
  --val-size VAL_SIZE     The number of unique sequences in each of the 
                        validation dataset.
  --test-size TEST_SIZE   The number of unique sequences in each of the test 
                        dataset.
  --include-poisson INCLUDE_POISSON
                        Include base intensity (where appropriate).
  --batch-size BATCH_SIZE
                        The batch size to use for parametric model training
                        and evaluation.
  --train-epochs TRAIN_EPOCHS
                        The number of training epochs.
  --use-coefficients USE_COEFFICIENTS
                        If true, the modular process will be trained with 
                        coefficients
  --multi-labels MULTI_LABELS
                        Whether the likelihood is computed on multi-labels 
                        events or not
  --time-scale TIME_SCALE
                        Time scale used to prevent overflow
  --enc-dec-hidden-size ENC_DEC_HIDDEN_SIZE
                        Number of features output by the encoder
  --lr-rate-init LR_RATE_INIT
                        Initial learning rate for optimization
  --lr-poisson-rate-init LR_POISSON_RATE_INIT
                        Initial poisson learning rate for optimization
  --lr-scheduler {plateau,step,milestones,cos,findlr,noam,clr,calr}
                        method to adjust learning rate
  --lr-scheduler-patience LR_SCHEDULER_PATIENCE
                        lr scheduler plateau: Number of epochs with no
                        improvement after which learning rate will be reduced
  --lr-scheduler-step-size LR_SCHEDULER_STEP_SIZE
                        lr scheduler step: number of epochs of learning rate
                        decay.
  --lr-scheduler-gamma LR_SCHEDULER_GAMMA
                        learning rate is multiplied by the gamma to decrease
                        it
  --lr-scheduler-warmup LR_SCHEDULER_WARMUP
                        The number of epochs to linearly increase the learning
                        rate. (noam only)
  --patience PATIENCE   The patience for early stopping.
  --loss-relative-tolerance LOSS_RELATIVE_TOLERANCE
                        The relative factor that the loss needs to decrease by
                        in order to not contribute to patience. If `None`,
                        will not use numerical convergence to control early
                        stopping. Defaults to `None`.
  --mu-cheat MU_CHEAT   If True, the starting mu value will be the actual mu 
                        value. Defaults to False.
  --encoder {gru,identity,mlp-fixed,mlp-variable,stub,selfattention}
                        The type of encoder to use.
  --encoder-history-size ENCODER_HISTORY_SIZE
                        The (fixed) history length to use for fixed history
                        size parametric models.
  --encoder-emb-dim ENCODER_EMB_DIM
                        Size of the embeddings. This is the size of the
                        temporal encoding and/or the label embedding if either
                        is used.
  --encoder-encoding {times_only,marks_only,concatenate,temporal,learnable,
                        temporal_with_labels,learnable_with_labels}
                        Type of the event encoding.
  --encoder-time-encoding {absolute, relative}
  --encoder-temporal-scaling ENCODER_TEMPORAL_SCALING
                        Rescale of times when using temporal encoding
  --encoder-embedding-constraint {None, nonneg, sigmoid, softplus}
  --encoder-units-mlp N [N ...]
                        Size of hidden layers in the encoder MLP. This will
                        have the decoder input size appended to it during
                        model build.
  --encoder-activation-mlp ENCODER_ACTIVATION_MLP
                        Activation function of the MLP
  --encoder-dropout-mlp ENCODER_DROPOUT_MLP
                        Dropout rate of the MLP.
  --encoder-constraint-mlp {None, nonneg, sigmoid, softplus}
  --encoder-activation-final-mlp ENCODER_ACTIVATION_FINAL_MLP
                        Final activation function of the MLP.
  --encoder-attn-activation {identity,softmax,sigmoid}
                        Activation function of the attention coefficients
  --encoder-dropout-rnn ENCODER_DROPOUT_RNN
                        Dropout rate of the RNN.
  --encoder-layers-rnn ENCODER_LAYERS_RNN
                        Number of layers for RNN and self-attention encoder.
  --encoder-units-rnn ENCODER_UNITS_RNN
                        Hidden size for RNN and self attention encoder.
  --encoder-n-heads ENCODER_N_HEADS
                        Number of heads for the transformer
  --encoder-constraint-rnn {None, nonneg, sigmoid, softplus}
  --decoder {conditional-poisson,conditional-poisson-cm,hawkes,log-normal-mixture,
                mlp-cm,mlp-mc,poisson,rmtpp,rmtpp-cm,selfattention-cm,selfattention-mc}
                        The type of decoder to use.
  --decoder-mc-prop-est DECODER_MC_PROP_EST
                        Proportion of MC samples, compared to dataset size
  --decoder-model-log-cm DECODER_MODEL_LOG_CM
                        Whether the cumulative models the log integral or the 
                        integral
  --decoder-do-zero-subtraction DECODER_DO_ZERO_SUBTRACTION
                        For cumulative estimation. If `True` the class computes 
                        Lambda(tau) = f(tau) - f(0) in order to enforce 
                        Lambda(0) = 0. Defaults to `true`, where instead 
                        Lambda(tau) = f(tau).
  --decoder-emb-dim DECODER_EMB_DIM
                        Size of the embeddings. This is the size of the
                        temporal encoding and/or the label embedding if either
                        is used.
  --decoder-encoding {times_only,marks_only,concatenate,temporal,
                        learnable,temporal_with_labels,learnable_with_labels}
                        Type of the event decoding.
  --decoder-time-encoding {absolute, relative}
  --decoder-temporal-scaling DECODER_TEMPORAL_SCALING
                        Rescale of times when using temporal encoding
  --decoder-embedding-constraint {None, nonneg, sigmoid, softplus}
  --decoder-units-mlp N [N ...]
                        Size of hidden layers in the decoder MLP. This will
                        have the number of marks appended to it during model
                        build.
  --decoder-activation-mlp DECODER_ACTIVATION_MLP
                        Activation function of the MLP
  --decoder-mlp-dropout DECODER_MLP_DROPOUT
                        Dropout rate of the MLP
  --decoder-activation-mlp DECODER_ACTIVATION_FINAL_MLP
                        Final activation function of the MLP.
  --decoder-constraint-mlp {None, nonneg, sigmoid, softplus}
  --decoder-attn-activation {identity,softmax,sigmoid}
                        Activation function of the attention coefficients
  --decoder-activation-rnn DECODER_ACTIVATION_RNN
                        Activation for the rnn.
  --decoder-dropout-rnn DECODER_DROPOUT_RNN
                        Dropout rate of the RNN
  --decoder-layers-rnn DECODER_LAYERS_RNN
                        Number of layers for self attention decoder.
  --decoder-rnn-units DECODER_RNN_UNITS
                        Hidden size for self attention decoder.
  --decoder-n-heads DECODER_N_HEADS
                        Number of heads for the transformer
  --decoder-constraint-rnn {None, nonneg, sigmoid, softplus}
  --decoder-n-mixture DECODER_N_MIXTURE
                        Number of mixtures for the log normal mixture model
  --no-mlflow           Do not use MLflow (default=False)
  --experiment-name EXPERIMENT_NAME
                        Name of MLflow experiment
  --run-name RUN_NAME   Name of MLflow run
  --remote-server-uri REMOTE_SERVER_URI
                        Remote MLflow server URI
  --logging-frequency LOGGING_FREQUENCY
                        The frequency to log values to MLFlow.
  --save-model-freq SAVE_MODEL_FREQ
                        The best model is saved every nth epoch
  --eval-metrics EVAL_METRICS
                        The model is evaluated using several metrics
  --eval-metrics-per-class EVAL_METRICS_PER_CLASS
                        The model is evaluated using several metrics per class
  --load-from-dir LOAD_FROM_DIR
                        If not None, load data from a directory
  --data-dir DATA_DIR   Directory to save the preprocessed data
  --plots-dir PLOTS_DIR
                        Directory to save the plots
```

### General usage - evaluating a trained model

```
usage: scripts/evaluate.py [-h] [--model-dir MODEL_DIR] [--seed SEED]
                [--padding-id PADDING_ID] [--mu N [N ...]] [--alpha N [N ...]]
                [--beta N [N ...]] [--marks MARKS] [--window WINDOW]
                [--val-size VAL_SIZE] [--test-size TEST_SIZE]
                [--batch-size BATCH_SIZE]
                [--time-scale TIME_SCALE] [--multi-labels MULTI_LABELS]
                [--eval-metrics EVAL_METRICS]
                [--eval-metrics-per-class EVAL_METRICS_PER_CLASS]
                [--load-from-dir LOAD_FROM_DIR]
                [--data-dir DATA_DIR]
                [--plots-dir PLOTS_DIR]
required arguments:
  --model-dir MODEL_DIR
                        Directory of the saved model
optional arguments:
  -h, --help            Show this help message and exit
  --seed SEED           The random seed.
  --padding-id PADDING_ID
                        The value used in the temporal sequences to indicate a
                        non-event.
  --mu N [N ...]        The baseline intensity for the data generator.
  --alpha N [N ...]     The event parameter for the data generator. This will
                        be reshaped into a matrix the size of [mu,mu].
  --beta N [N ...]      The decay parameter for the data generator. This will
                        be reshaped into a matrix the size of [mu,mu].
  --marks MARKS         Generate a process with this many marks. Defaults to
                        `None`. If this is set to an integer, it will override
                        `alpha`, `beta` and `mu` with randomly generated
                        values corresponding to the number of requested marks.
  --window WINDOW       The window of the simulated process.py. Also taken as
                        the window of any parametric Hawkes model if chosen.
  --val-size VAL_SIZE     The number of unique sequences in each of the 
                        validation dataset.
  --test-size TEST_SIZE   The number of unique sequences in each of the test 
                        dataset.
  --batch-size BATCH_SIZE
                        The batch size to use for parametric model training
                        and evaluation.
  --time-scale TIME_SCALE
                        Time scale used to prevent overflow
  --multi-labels MULTI_LABELS
                        Whether the likelihood is computed on multi-labels 
                        events or not
   --eval-metrics EVAL_METRICS
                        The model is evaluated using several metrics
  --eval-metrics-per-class EVAL_METRICS_PER_CLASS
                        The model is evaluated using several metrics per class
  --load-from-dir LOAD_FROM_DIR
                        If not None, load data from a directory
  --data-dir DATA_DIR   Directory to save the preprocessed data
  --plots-dir PLOTS_DIR
                        Directory to save the plots
```

## Repo notes
This repository was originally forked from https://github.com/woshiyyya/ERPP-RMTPP
