import json
import numpy as np
import os

from argparse import Namespace
from pathlib import Path
from tqdm import tqdm

from tpp.processes.multi_class_dataset import generate_points
from tpp.utils.cli import parse_args
from tpp.utils.run import make_deterministic
from tpp.utils.record import hawkes_seq_to_record


def main(args: Namespace):
    if args.verbose:
        print(args)

    # Create paths for plots
    data_dir = os.path.expanduser(args.data_dir)
    hawkes_data_dir = os.path.join(data_dir, "hawkes")
    Path(hawkes_data_dir).mkdir(parents=True, exist_ok=True)

    seeds = {
        "train": [args.train_size, args.seed],
        "val": [args.val_size, args.seed + args.train_size],
        "test": [args.test_size, args.seed + args.train_size + args.val_size]}

    for name, [size, seed] in seeds.items():
        range_size = range(size)
        if args.verbose:
            range_size = tqdm(range_size)

        times_marked = [
            generate_points(
                n_processes=args.marks,
                mu=args.mu.astype(np.float64),
                alpha=args.alpha.astype(np.float64),
                decay=args.beta.astype(np.float64),
                window=args.window,
                seed=seed + i
            ) for i in range_size]  # D x M x Li

        records = [hawkes_seq_to_record(r) for r in times_marked]

        with open(os.path.join(hawkes_data_dir, name + ".json"), "w") as h:
            h.write(
                '[' + ',\n'.join(json.dumps(i) for i in records) + ']\n')

    codes_to_names = {str(i): str(i) for i in range(args.marks)}

    args_dict = vars(args)
    args_dict['alpha'] = args_dict['alpha'].tolist()
    args_dict['beta'] = args_dict['beta'].tolist()
    args_dict['mu'] = args_dict['mu'].tolist()
    keys_to_keep = ["seed", "mu", "alpha", "beta", "marks", "hawkes_seed",
                    "window", "train_size", "val_size", "test_size"]
    args_dict = {k: args_dict[k] for k in keys_to_keep}
    with open(os.path.join(hawkes_data_dir, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp)
    with open(os.path.join(hawkes_data_dir, 'int_to_codes.json'), 'w') as fp:
        json.dump(codes_to_names, fp)
    with open(os.path.join(hawkes_data_dir, 'codes_to_int.json'), 'w') as fp:
        json.dump(codes_to_names, fp)
    with open(os.path.join(hawkes_data_dir, 'codes_to_names.json'), 'w') as fp:
        json.dump(codes_to_names, fp)
    with open(os.path.join(hawkes_data_dir, 'names_to_codes.json'), 'w') as fp:
        json.dump(codes_to_names, fp)


if __name__ == "__main__":
    parsed_args = parse_args()

    # check_repo(allow_uncommitted=not parsed_args.use_mlflow)
    make_deterministic(seed=parsed_args.seed)

    main(args=parsed_args)
