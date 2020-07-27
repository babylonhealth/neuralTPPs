import time
import matplotlib.pyplot as plt
import numpy as np
import torch as th

# from tpp.processes.hawkes.r_terms import get_r_terms as naive
from tpp.processes.hawkes.r_terms_recursive import get_r_terms as recursive
from tpp.processes.hawkes.r_terms_recursive_v import get_r_terms as recursive_v
from tpp.utils.test import get_test_events_query


def get_times(batch_size=1, marks=1, seq_len=16, n_queries=16, n_iters=100):
    events, _ = get_test_events_query(
        marks=marks, batch_size=batch_size, max_seq_len=seq_len,
        queries=n_queries)
    beta = th.rand([marks, marks], dtype=th.float32).to(th.device("cpu"))

    # t1 = time.time()
    # for _ in range(n_iters):
    #     naive(events=events, beta=beta)
    # t1 = time.time() - t1
    # t1 = t1 / n_iters

    t2 = time.time()
    for _ in range(n_iters):
        recursive(events=events, beta=beta)
    t2 = time.time() - t2
    t2 = t2 / n_iters

    t3 = time.time()
    for _ in range(n_iters):
        recursive_v(events=events, beta=beta)
    t3 = time.time() - t3
    t3 = t3 / n_iters

    if th.cuda.is_available():
        events, query = get_test_events_query(
            marks=marks, batch_size=batch_size, max_seq_len=seq_len,
            queries=n_queries, device=th.device("cuda"))
        beta = beta.to(th.device("cuda"))
        t4 = time.time()
        for _ in range(n_iters):
            recursive_v(events=events, beta=beta)
        t4 = time.time() - t4
        t4 = t4 / n_iters
    else:
        t4 = None
    return None, t2, t3, t4


def main():
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
                batch_size=int(b), seq_len=int(seq_len), n_iters=10)
            for seq_len in seq_lens}

        naive, recursive, recursive_v, recursive_v_cuda = zip(*times.values())

        # a.plot(seq_lens, naive, label="naive")
        a.plot(seq_lens, recursive, label="recursive")
        a.plot(seq_lens, recursive_v, label="recursive_v")
        if th.cuda.is_available():
            a.plot(seq_lens, recursive_v_cuda, label="recursive_v_cuda")
        a.legend()
        a.set_xlabel("Sequence length")
        a.set_ylabel("Query time")
        a.set_yscale("log")
        a.set_title("batch size {}".format(b))

    fig.show()


if __name__ == "__main__":
    main()
