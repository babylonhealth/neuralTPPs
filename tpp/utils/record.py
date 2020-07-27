import numpy as np
import torch as th

from typing import Dict, List, Tuple


def hawkes_seq_to_record(seq: List[np.ndarray]):
    times = np.concatenate(seq)
    labels = np.concatenate([[i] * len(x) for i, x in enumerate(seq)])
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    labels = labels[sort_idx]
    record = [
        {"time": float(t),
         "labels": (int(l),)} for t, l in zip(times, labels)]
    return record
