import json
import os

import numpy as np
import torch as th

from tick.hawkes import SimuHawkes, HawkesKernelExp
from tqdm import tqdm
from typing import List

from tpp.utils.marked_times import objects_from_events
from tpp.utils.marked_times import pad
from tpp.utils.record import hawkes_seq_to_record


class MultiClassDataset:
    """
    MultiClassDataset: Unmarked Multi-class Hawkes dataset.
    The intensity function for this process is:
    lambda_i(t|tjk) = mu(i) + sum[
        alpha(i, j) * sum[exp(t - tjk) for tjk < t] for j in range(n_nodes)
    ]
    where tjk are timestamps of all events of node j
    Args
        alpha: excitation components of the Hawkes processes
        decay: decay components of the Hawes processes
        device: device where the generated data is loaded
        mu: base components of the Hawkes process
        n_processes: number of generated Hawkes processes
        padding_id: id of the padded value used to generate the dataset
        seed: seed of the process
        size: size of the dataset
        window: window size of the Hawkes processes
    """
    def __init__(self, args, size, seed, name):
        self.alpha = args.alpha.astype(np.float64)
        self.data_dir = args.data_dir
        self.decay = args.beta.astype(np.float64)
        self.device = args.device
        self.mu = args.mu.astype(np.float64)
        self.n_processes = args.marks
        self.name = name
        self.padding_id = args.padding_id
        self.load_from_dir = args.load_from_dir
        self.seed = seed
        self.size = size
        self.window = args.window
        self.times_dtype = th.float32
        self.labels_dtype = th.float32
        self.verbose = args.verbose

        if args.load_from_dir is None:
            assert len(self.alpha.shape) == 2
            assert self.alpha.shape[0] == self.alpha.shape[1]
            assert self.alpha.shape[0] == self.n_processes

            assert len(self.decay.shape) == 2
            assert self.decay.shape[0] == self.decay.shape[1]
            assert self.decay.shape[0] == self.n_processes

        self.raw_objects = self._build_sequences()
        self.times = self.raw_objects["times"]
        self.labels = self.raw_objects["labels"]

        self.lengths = [len(x) for x in self.times]

        self.max_length = max(self.lengths)
        self.lengths = th.Tensor(self.lengths).long().to(self.device)

        self.build_dict_names(args)

    def _build_sequences(self):
        events = self.load_data()

        if "events" in events[0]:
            records = events
            events = [r["events"] for r in records]
            events = [e for e in events if len(e) > 0]

        # times, labels
        raw_objects = objects_from_events(
            events=events,
            marks=self.n_processes,
            labels_dtype=self.labels_dtype,
            verbose=self.verbose,
            device=self.device)

        not_empty = [len(x) > 0 for x in raw_objects["times"]]

        def keep_not_empty(x):
            return [y for y, nonempty in zip(x, not_empty) if nonempty]

        return {k: keep_not_empty(v) for k, v in raw_objects.items()}

    def __getitem__(self, item):
        raw_objects = {k: v[item] for k, v in self.raw_objects.items()}
        seq_len = self.lengths[item]
        result = {
            "raw": raw_objects, "seq_len": seq_len,
            "padding_id": self.padding_id}
        return result

    def __len__(self):
        return len(self.times)

    @staticmethod
    def to_features(batch):
        """
        Casts times and events to PyTorch tensors
        """
        times = [b["raw"]["times"] for b in batch]
        labels = [b["raw"]["labels"] for b in batch]
        padding_id = batch[0]["padding_id"]

        assert padding_id not in th.cat(times)
        padded_times = pad(x=times, value=padding_id)  # [B,L]
        # Pad with zero, not with padding_id so that the embeddings don't fail.
        padded_labels = pad(x=labels, value=0)  # [B,L]

        features = {"times": padded_times, "labels": padded_labels,
                    "seq_lens": th.stack([b["seq_len"] for b in batch])}
        return features

    def load_data(self) -> List:
        if self.load_from_dir is not None:
            data_dir = os.path.join(self.data_dir, self.load_from_dir)
            data_path = os.path.join(data_dir, self.name + ".json")
            with open(data_path, "r") as h:
                records = json.load(h)
        else:
            range_size = range(self.size)
            if self.verbose:
                range_size = tqdm(range_size)

            times_marked = [
                generate_points(
                    n_processes=self.n_processes,
                    mu=self.mu,
                    alpha=self.alpha,
                    decay=self.decay,
                    window=self.window,
                    seed=self.seed + i) for i in range_size]  # D x M x Li

            records = [hawkes_seq_to_record(seq) for seq in times_marked]

        return records

    def build_dict_names(self, args):
        if self.load_from_dir is None:
            codes_to_names = names_to_codes = {}
            for i in range(self.n_processes):
                codes_to_names[str(i)] = str(i)
                names_to_codes[str(i)] = str(i)
            with open(os.path.join(args.save_dir, 'codes_to_int.json'), 'w'
                      ) as fp:
                json.dump(codes_to_names, fp)
            with open(os.path.join(args.save_dir, 'int_to_codes.json'), 'w'
                      ) as fp:
                json.dump(names_to_codes, fp)
            with open(os.path.join(args.save_dir, 'int_to_codes_to_plot.json'
                                   ), 'w'
                      ) as fp:
                json.dump(names_to_codes, fp)
            with open(os.path.join(args.save_dir, 'codes_to_names.json'), 'w'
                      ) as fp:
                json.dump(codes_to_names, fp)
            with open(os.path.join(args.save_dir, 'names_to_codes.json'), 'w'
                      ) as fp:
                json.dump(names_to_codes, fp)
            with open(os.path.join(args.save_dir, 'int_to_codes.json'), 'w'
                      ) as fp:
                json.dump(names_to_codes, fp)
            with open(os.path.join(args.save_dir, 'codes_to_int.json'), 'w'
                      ) as fp:
                json.dump(names_to_codes, fp)
            with open(os.path.join(args.save_dir, 'int_to_codes_to_plot.json'
                                   ), 'w') as fp:
                json.dump(names_to_codes, fp)


def generate_points(
        n_processes,
        mu,
        alpha,
        decay,
        window,
        seed,
        dt=0.01):
    """
    Generates points of an marked Hawkes processes using the tick library
    """
    hawkes = SimuHawkes(
        n_nodes=n_processes,
        end_time=window,
        verbose=False,
        seed=seed)
    for i in range(n_processes):
        for j in range(n_processes):
            hawkes.set_kernel(
                i=i, j=j,
                kernel=HawkesKernelExp(
                    intensity=alpha[i][j] / decay[i][j], decay=decay[i][j]))
        hawkes.set_baseline(i, mu[i])

    hawkes.track_intensity(dt)
    hawkes.simulate()
    return hawkes.timestamps
