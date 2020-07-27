import torch as th

from tpp.utils.test import get_test_events_query
from tpp.utils.history_bst import get_prev_times
from tpp.utils.history_marked_bst import get_prev_times_marked


def test_get_previous_times_marked():
    th.random.manual_seed(0)
    events, query = get_test_events_query(
        batch_size=1, max_seq_len=12, queries=7)
    result = get_prev_times_marked(query=query, events=events)

    a = 0


if __name__ == "__main__":
    test_get_previous_times_marked()
