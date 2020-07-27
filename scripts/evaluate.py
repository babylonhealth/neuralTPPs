import json
import numpy as np
import os
import torch as th

from argparse import ArgumentParser, Namespace
from distutils.util import strtobool
from pathlib import Path

from scripts.train import evaluate
from tpp.processes.multi_class_dataset import MultiClassDataset as Dataset
from tpp.utils.data import get_loader
from tpp.utils.plot import log_figures


def parse_args():
    parser = ArgumentParser(allow_abbrev=False)
    # Model dir
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory of the saved model")
    # Run configuration
    parser.add_argument("--seed", type=int, default=0, help="The random seed.")
    parser.add_argument("--padding-id", type=float, default=-1.,
                        help="The value used in the temporal sequences to "
                             "indicate a non-event.")
    # Simulator configuration
    parser.add_argument("--mu", type=float, default=[0.05, 0.05],
                        nargs="+", metavar='N',
                        help="The baseline intensity for the data generator.")
    parser.add_argument("--alpha", type=float, default=[0.1, 0.2, 0.2, 0.1],
                        nargs="+", metavar='N',
                        help="The event parameter for the data generator. "
                             "This will be reshaped into a matrix the size of "
                             "[mu,mu].")
    parser.add_argument("--beta", type=float, default=[1.0, 1.0, 1.0, 1.0],
                        nargs="+", metavar='N',
                        help="The decay parameter for the data generator. "
                             "This will be reshaped into a matrix the size of "
                             "[mu,mu].")
    parser.add_argument("--marks", type=int, default=None,
                        help="Generate a process with this many marks. "
                             "Defaults to `None`. If this is set to an "
                             "integer, it will override `alpha`, `beta` and "
                             "`mu` with randomly generated values "
                             "corresponding to the number of requested marks.")
    parser.add_argument("--window", type=int, default=100,
                        help="The window of the simulated process.py. Also "
                             "taken as the window of any parametric Hawkes "
                             "model if chosen.")
    parser.add_argument("--val-size", type=int, default=128,
                        help="The number of unique sequences in each of the "
                             "validation dataset.")
    parser.add_argument("--test-size", type=int, default=128,
                        help="The number of unique sequences in each of the "
                             "test dataset.")
    # Common model hyperparameters
    parser.add_argument("--batch-size", type=int, default=32,
                        help="The batch size to use for parametric model"
                             " training and evaluation.")
    parser.add_argument("--time-scale", type=float, default=1.,
                        help='Time scale used to prevent overflow')
    parser.add_argument("--multi-labels",
                        type=lambda x: bool(strtobool(x)), default=False,
                        help="Whether the likelihood is computed on "
                             "multi-labels events or not")
    # Metrics
    parser.add_argument("--eval-metrics",
                        type=lambda x: bool(strtobool(x)), default=False,
                        help="The model is evaluated using several metrics")
    parser.add_argument("--eval-metrics-per-class",
                        type=lambda x: bool(strtobool(x)), default=False,
                        help="The model is evaluated using several metrics "
                             "per class")
    # Dirs
    parser.add_argument("--load-from-dir", type=str, default=None,
                        help="If not None, load data from a directory")
    parser.add_argument("--plots-dir", type=str,
                        default="~/neural-tpps/plots",
                        help="Directory to save the plots")
    parser.add_argument("--data-dir", type=str, default="~/neural-tpps/data",
                        help="Directory to save the preprocessed data")

    args, _ = parser.parse_known_args()
    if args.marks is None:
        args.mu = np.array(args.mu, dtype=np.float32)
        args.alpha = np.array(args.alpha, dtype=np.float32).reshape(
            args.mu.shape * 2)
        args.beta = np.array(args.beta, dtype=np.float32).reshape(
            args.mu.shape * 2)
        args.marks = len(args.mu)
    else:
        np.random.seed(args.hawkes_seed)
        args.mu = np.random.uniform(
            low=0.01, high=0.2, size=[args.marks]).astype(dtype=np.float32)
        args.alpha = np.random.uniform(
            low=0.01, high=0.2, size=[args.marks] * 2).astype(dtype=np.float32)
        args.beta = np.random.uniform(
            low=1.01, high=1.3, size=[args.marks] * 2).astype(dtype=np.float32)
        args.mu /= float(args.marks)
        args.alpha /= float(args.marks)

    if args.load_from_dir is not None:
        args.data_dir = os.path.expanduser(args.data_dir)
        args.save_dir = os.path.join(args.data_dir, args.load_from_dir)
        with open(os.path.join(args.save_dir, 'args.json'), 'r') as fp:
            args_dict_json = json.load(fp)
        args_dict = vars(args)
        print("Warning: overriding some args from json:")
        shared_keys = set(args_dict_json).intersection(set(args_dict))
        for k in shared_keys:
            v1, v2 = args_dict[k], args_dict_json[k]
            is_equal = np.allclose(v1, v2) if isinstance(
                v1, np.ndarray) else v1 == v2
            if not is_equal:
                print(f"    {k}: {v1} -> {v2}")
        args_dict.update(args_dict_json)
        args = Namespace(**args_dict)
        args.mu = np.array(args.mu, dtype=np.float32)
        args.alpha = np.array(
            args.alpha, dtype=np.float32).reshape(
            args.mu.shape * 2)
        args.beta = np.array(
            args.beta, dtype=np.float32).reshape(
            args.mu.shape * 2)

    else:
        args.data_dir = os.path.expanduser(args.data_dir)
        args.save_dir = os.path.join(args.data_dir, "None")
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    args.device = th.device('cpu')
    args.verbose = False

    return args


def main(args: Namespace):
    model = th.load(args.model_dir, map_location=args.device)

    dataset = Dataset(
        args=args, size=args.test_size, seed=args.seed, name="test")
    loader = get_loader(dataset, args=args, shuffle=False)

    log_figures(
        model=model,
        test_loader=loader,
        args=args,
        save_on_mlflow=False)

    metrics = evaluate(model=model, args=args, loader=loader)
    print(metrics)


if __name__ == "__main__":
    parsed_args = parse_args()
    main(args=parsed_args)
