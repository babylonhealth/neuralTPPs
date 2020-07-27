import torch as th

from tpp.utils.events import get_events, get_window
from tpp.utils.marked_times import pad


def get_test_events_query(
        marks=2,
        batch_size=16,
        max_seq_len=16,
        queries=4,
        padding_id=-1.,
        device=th.device('cpu'), dtype=th.float32):
    seq_lens = th.randint(low=1, high=max_seq_len, size=[batch_size])
    times = [th.rand(size=[seq_len]) for seq_len in seq_lens]
    labels = [th.randint(low=0, high=marks, size=[seq_len])
              for seq_len in seq_lens]
    sort_idx = [th.argsort(x) for x in times]
    times = [x[idx] for x, idx in zip(times, sort_idx)]
    labels = [x[idx] for x, idx in zip(labels, sort_idx)]

    times = pad(times, value=padding_id).type(dtype)
    labels = pad(labels, value=0)

    times, labels = times.to(device), labels.to(device)

    mask = (times != padding_id).type(times.dtype).to(times.device)
    window_start, window_end = get_window(times=times, window=1.)
    events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end, marks=marks)

    query = th.rand(size=[batch_size, queries])
    query = th.sort(query, dim=-1).values

    query = query.to(device)

    return events, query
