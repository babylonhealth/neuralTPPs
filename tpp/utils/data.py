import os

from torch.utils.data import DataLoader
from argparse import Namespace
from typing import Optional

from tpp.processes.multi_class_dataset import MultiClassDataset as Dataset


def get_loader(
        dataset: Dataset,
        args: Namespace,
        shuffle: Optional[bool] = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=Dataset.to_features)


def load_data(args: Namespace) -> dict:
    size_seeds = {
        "train": [args.train_size, args.seed],
        "val": [args.val_size, args.seed + args.train_size],
        "test": [args.test_size, args.seed + args.train_size + args.val_size]}

    if args.verbose:
        print("Generating datasets...")
    datasets = {
        k: Dataset(args=args, size=sz, seed=ss, name=k
                   ) for k, (sz, ss) in size_seeds.items()}

    if args.verbose:
        max_seq_lens = {
            k: max(v.lengths) for k, v in datasets.items()}
        print("Done! Maximum sequence lengths:")
        for k, v in max_seq_lens.items():
            print("{}: {}".format(k, v))

    return datasets
