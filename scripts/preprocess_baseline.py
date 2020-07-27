import json
import numpy as np
import os
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path

pd.set_option("max.rows", 10)
pd.set_option("max.columns", 20)
pd.set_option("display.width", 200)


def parse_args():
    parser = ArgumentParser(allow_abbrev=False)
    # Run configuration
    parser.add_argument("--seed", type=int, default=0, help="The random seed.")
    parser.add_argument("--ratio-train", type=float, default=0.6,
                        help="Ratio between train and total size.")
    parser.add_argument("--ratio-val", type=float, default=0.8,
                        help="Ratio between train+val and total size.")
    parser.add_argument("--marks", type=int, default=23,
                        help="Number of marks in the dataset")
    parser.add_argument("--baseline-path", type=str,
                        default='~/NeuralPointProcess/data/real/so',
                        help="The directory where baseline is saved")
    parser.add_argument("--save-path", type=str,
                        default='~/neural-tpps/data/baseline/so',
                        help="Path where the preprocessed data is saved.")
    args, _ = parser.parse_known_args()
    return args


def main(
        seed: int,
        ratio_train: float,
        ratio_val: float,
        marks: int,
        baseline_path: str,
        save_path: str):
    baseline_path = os.path.expanduser(baseline_path)
    save_path = os.path.expanduser(save_path)

    with open(os.path.join(baseline_path, "event.txt"), "r") as f:
        labels = f.read().splitlines()
    with open(os.path.join(baseline_path, "time.txt"), "r") as f:
        times = f.read().splitlines()
    assert len(labels) == len(times)
    for i in range(len(labels)):
        labels[i] = labels[i].split(" ")
        times[i] = times[i].split(" ")
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if len(times[i][j]) == 0:
                labels[i].pop(j)
                times[i].pop(j)

    codes_to_int = {str(i): str(i) for i in range(marks)}

    encounters = list()
    for i in range(len(labels)):
        encounters.append(list())
        for j in range(len(labels[i])):
            encounters[-1].append({
                "time": float(times[i][j]) - float(times[i][0]),
                "labels": [int(codes_to_int[str(int(labels[i][j])-1)])]})

    np.random.seed(seed=seed)
    np.random.shuffle(encounters)
    n_patients = len(encounters)
    train_idx = int(ratio_train * n_patients)
    val_idx = int(ratio_val * n_patients)

    data_train = encounters[:train_idx]
    data_valid = encounters[train_idx:val_idx]
    data_test = encounters[val_idx:]

    # Save data
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_dict = {
        "train": data_train,
        "val": data_valid,
        "test": data_test}

    for key, value in save_dict.items():
        with open(os.path.join(save_path, key + ".json"), "w") as h:
            h.write(
                '[' + ',\n'.join(json.dumps(i) for i in value) + ']\n')

    args_dict = {
        "seed": seed,
        "window": None,
        "train_size": len(data_train),
        "val_size": len(data_valid),
        "test_size": len(data_test),
        "ratio_train": ratio_train,
        "ratio_val": ratio_val,
        "marks": len(codes_to_int)}
    with open(os.path.join(save_path, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp)

    with open(os.path.join(save_path, 'codes_to_int.json'), 'w') as fp:
        json.dump(codes_to_int, fp)
    with open(os.path.join(save_path, 'int_to_codes.json'), 'w') as fp:
        json.dump(codes_to_int, fp)
    with open(os.path.join(save_path, 'codes_to_names.json'), 'w') as fp:
        json.dump(codes_to_int, fp)
    with open(os.path.join(save_path, 'names_to_codes.json'), 'w') as fp:
        json.dump(codes_to_int, fp)


if __name__ == '__main__':
    parsed_args = parse_args()
    main(
        seed=parsed_args.seed,
        ratio_train=parsed_args.ratio_train,
        ratio_val=parsed_args.ratio_val,
        marks=parsed_args.marks,
        baseline_path=parsed_args.baseline_path,
        save_path=parsed_args.save_path)
