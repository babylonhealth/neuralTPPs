import torch as th

from tpp.utils.test import get_test_events_query


def test_to_flat_idxs():
    events, query = get_test_events_query()

    times_marked = events.get_times(marked=True)
    times = events.get_times()
    masks = events.get_mask(marked=True)
    to_flat_idxs = events.to_flat_idxs

    for time_marked, time, mask, to_flat_idx in zip(
            times_marked, times, masks, to_flat_idxs):
        for tm, m, idx in zip(time_marked, mask, to_flat_idx):
            assert th.all(tm * m == th.take(time, idx) * m)


if __name__ == "__main__":
    test_to_flat_idxs()
