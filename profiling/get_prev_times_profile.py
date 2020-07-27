import time
import torch as th
import numpy as np

import matplotlib.pyplot as plt

from tpp.utils.events import get_events, get_window
from tpp.utils.history import get_prev_times
from tpp.utils import history_bst


def get_test_events_query(
        batch_size=16, seq_len=16, n_queries=16, device=th.device('cpu'),
        dtype=th.float32):
    marks = 1
    padding_id = -1.

    times = np.random.uniform(
        low=0.01, high=1., size=[batch_size, seq_len]).astype(np.float32)
    query = np.random.uniform(
        low=0.01, high=1., size=[batch_size, n_queries]).astype(np.float32)
    mask = times != padding_id
    times, query = th.from_numpy(times), th.from_numpy(query)
    times, query = times.type(dtype), query.type(dtype)
    mask = th.from_numpy(mask).type(times.dtype)
    times, query, mask = times.to(device), query.to(device), mask.to(device)

    window_start, window_end = get_window(times=times, window=1.)
    events = get_events(
        times=times,
        mask=mask,
        window_start=window_start,
        window_end=window_end)

    (prev_times, _), is_event, _ = get_prev_times(
        query=query, events=events, allow_window=True)

    alpha = th.from_numpy(np.array([[0.1]], dtype=np.float32))
    beta = th.from_numpy(np.array([[1.0]], dtype=np.float32))
    mu = th.from_numpy(np.array([0.05], dtype=np.float32))

    return marks, query, events, prev_times, is_event, alpha, beta, mu


def get_times(batch_size=1, seq_len=16, n_queries=16, n_iters=100):
    (marks, query, events, prev_times,
     is_event, alpha, beta, mu) = get_test_events_query(
        batch_size=batch_size, seq_len=seq_len, n_queries=n_queries)

    t1 = time.time()
    for _ in range(n_iters):
        get_prev_times(query=events.times, events=events)
    t1 = time.time() - t1
    t1 = t1 / n_iters

    t2 = time.time()
    for _ in range(n_iters):
        history_bst.get_prev_times(query=events.times, events=events)
    t2 = time.time() - t2
    t2 = t2 / n_iters

    if th.cuda.is_available():
        (marks, query, events, prev_times,
         is_event, alpha, beta, mu) = get_test_events_query(
            batch_size=batch_size, seq_len=seq_len, n_queries=n_queries,
            device=th.device("cuda"))

        t3 = time.time()
        for _ in range(n_iters):
            history_bst.get_prev_times(query=events.times, events=events)
        t3 = time.time() - t3
        t3 = t3 / n_iters
    else:
        t3 = None
    return t1, t2, t3


def main():
    from importlib import reload
    reload(history_bst)
    seq_lens = np.arange(3, 6)
    seq_lens = np.power(2, seq_lens)

    batch_sizes = np.arange(0, 9)
    batch_sizes = np.power(2, batch_sizes)

    fig, ax = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=[7, 7])
    for b, a in zip(batch_sizes, ax.flatten()):
        times = {
            seq_len: get_times(
                batch_size=b, seq_len=seq_len, n_iters=20)
            for seq_len in seq_lens}

        normal_time, bst_time, bst_cuda_time = zip(*times.values())

        a.plot(seq_lens, normal_time, label="normal")
        a.plot(seq_lens, bst_time, label="bst")
        if th.cuda.is_available():
            a.plot(seq_lens, bst_cuda_time, label="bst_cuda_32")
        a.legend()
        a.set_xlabel("Sequence length")
        a.set_ylabel("Query time")
        a.set_yscale("log")
        a.set_title("batch size {}".format(b))

    fig.show()


if __name__ == "__main__":
    main()
