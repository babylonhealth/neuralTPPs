import torch as th

from tests.processes.test_hawkes_fast import get_test_setup
from tpp.utils.history import get_prev_times
from tpp.utils.history_bst import get_prev_times as get_prev_times_bst


def test_get_previous_times():
    (marks, query, events,
     prev_times, is_event,
     alpha, beta, mu) = get_test_setup()

    for allow_window in [False, True]:
        (prev_times, prev_times_idxs), is_event, mask = get_prev_times(
            query=query, events=events, allow_window=allow_window)
        ((prev_times_bst, prev_times_idxs_bst),
         is_event_bst, mask_bst) = get_prev_times_bst(
            query=query, events=events, allow_window=allow_window)

        assert th.all(mask == mask_bst)

        prev_times_masked = prev_times * mask
        prev_times_masked_bst = prev_times_bst * mask_bst
        assert th.all(prev_times_masked == prev_times_masked_bst)

        prev_times_idx_masked = prev_times * mask.type(prev_times.dtype)
        prev_times_idx_masked_bst = prev_times_bst * mask_bst.type(
            prev_times_bst.dtype)
        assert th.all(prev_times_idx_masked == prev_times_idx_masked_bst)

        is_event_masked = is_event * mask.type(is_event.dtype)
        is_event_masked_bst = is_event_bst * mask.type(is_event_bst.dtype)
        assert th.all(is_event_masked == is_event_masked_bst)


if __name__ == "__main__":
    test_get_previous_times()
