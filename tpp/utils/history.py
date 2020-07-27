import torch as th

from typing import Optional, Tuple

from tpp.utils import batch as bu
from tpp.utils.events import Events
from tpp.utils.utils import smallest_positive
from tpp.utils.index import take_2_by_2
from tpp.utils.history_bst import get_prev_times as get_prev_times_bst


def _get_rank(x: th.Tensor) -> int:
    return len(x.shape)


def expand_to_rank(x: th.Tensor, rank: int, dim: int = -1) -> th.Tensor:
    """Expand a tensor to a desired rank.

    Args:
        x: The tensor to expand.
        rank: The target rank.
        dim: The dim to expand along. Defaults to `-1`.

    Returns:
        A tensor expanded to the given rank.

    """
    x_rank = _get_rank(x)
    if x_rank > rank:
        raise ValueError(
            "Rank of `x` ({}) greater than desired rank ({})".format(
                x_rank, rank))
    for _ in range(rank - x_rank):
        x = x.unsqueeze(dim)
    return x


def build_histories(
        query: th.Tensor,
        events: Events,
        history_size: Optional[int] = 1) -> Tuple[th.Tensor, th.Tensor]:
    """Get the set of times corresponding to the 'history' of a query time of
    fixed size.

    Args:
        query: [B,T] The times to create histories for.
        events: [B,L] Times and labels of events to create histories from.
        history_size: The size of each history. Defaults to 1.

    Returns:
        history (th.Tensor): [B,T,H] The history for each query time.
        mask (th.Tensor): [B,T] The mask corresponding to whether a
            particular query can be used or not based on the required size of
            history.

    """
    batch_size, max_queries = query.shape
    batch_size_s, max_seq_len = events.times.shape

    if batch_size_s != batch_size:
        raise ValueError(
            "The batch size for `query_times` "
            "({}) does not match the batch size for `sequences` "
            "({}).".format(batch_size, batch_size_s))

    if history_size > max_seq_len:
        raise ValueError(
            "The chosen value for `history_size` "
            "({}) is greater than the size of the largest sequence "
            "({}).".format(history_size, max_seq_len))

    ((prev_times, prev_times_idxs),
     is_event, pos_delta_mask) = get_prev_times_bst(
        query=query, events=events)             # ([B,T], [B,T]), [B,T], [B,T]

    relative_history_idxs = th.arange(
        start=1 - history_size, end=1, device=events.times.device)

    batch_idxs_shift = th.arange(
        start=0, end=batch_size_s, device=events.times.device) * max_seq_len
    batch_idxs_shift = batch_idxs_shift.reshape([batch_size, 1, 1])

    history_seq_idxs = prev_times_idxs.reshape([batch_size, max_queries, 1])
    history_seq_idxs = history_seq_idxs + relative_history_idxs
    batch_history_seq_idxs = history_seq_idxs + batch_idxs_shift
    batch_history_seq_idxs = batch_history_seq_idxs.long()  # [B,T,H]

    history = th.take(events.times, batch_history_seq_idxs)

    history_idxs_positive = th.prod(history_seq_idxs >= 0, dim=-1)  # [B,T]
    history_idxs_positive = history_idxs_positive.type(pos_delta_mask.dtype)
    history_mask = pos_delta_mask * history_idxs_positive
    history_mask = history_mask.type(history.dtype)  # [B,T]

    return history, history_mask


def get_prev_times(
        query: th.Tensor,
        events: Events,
        allow_window: Optional[bool] = False
) -> Tuple[Tuple[th.Tensor, th.Tensor], th.Tensor, th.Tensor]:
    """For each query, get the event time that directly precedes it. If no
    events precedes it (but the window start does), return the window start.
    Otherwise, mask the value.

    Args:
        query: [B,T] Sequences of query times to evaluate the intensity
            function.
        events: [B,L] Times and labels of events.
        allow_window: If `True`, a previous time can be the window boundary.
            Defaults to `False`.

    Returns:
        `times` is a tuple of tensor of values [B,T] and indices,  [B,T] of the
            largest time value in the sequence that is strictly smaller than
            the query time value, or the window. the index only event indexes
            into the events. If the window is returned, it should be dealt with
            explicitly at encoding/decoding time.

        `is_event` is a tensor [B,T] that indicates whether the time
            corresponds to an event or not (a 1 indicates an event and a 0
            indicates a window boundary).

        `mask` is a tensor [B,T] that indicates whether the time difference to
            those times what positive or not.

    """
    event_times = events.get_times(prepend_window=allow_window)     # [B,L+1]
    batch_size, max_seq_len = event_times.shape

    time_diffs = bu.batchwise_difference(query, event_times)        # [B,T,L+1]

    event_mask = events.get_mask(prepend_window=allow_window)       # [B,L+1]
    event_mask = event_mask.reshape([batch_size, 1, max_seq_len])   # [B,1,L+1]
    time_diffs = time_diffs * event_mask                            # [B,T,L+1]
    smallest_time_diffs, mask = smallest_positive(time_diffs, dim=-1)   # [B,T]
    prev_times_idxs = smallest_time_diffs.indices                       # [B,T]
    prev_times = take_2_by_2(event_times, index=prev_times_idxs)        # [B,T]

    if allow_window:
        # If the first event shares a time with the window boundary, that the
        # index returned is the index of the event, rather than the window
        # boundary.
        idx_is_window = (prev_times_idxs == 0).type(
            prev_times_idxs.dtype)                                      # [B,T]
        do_idx_shift = events.first_event_on_window.type(
            idx_is_window.dtype)                                        # [B]
        idx_shift = idx_is_window * do_idx_shift.reshape(-1, 1)
        prev_times_idxs = prev_times_idxs + idx_shift

        # Check the indexes in case one of the window indexes became an event.
        is_event = (prev_times_idxs != 0).type(mask.dtype)             # [B,T]
    else:
        is_event = th.ones_like(prev_times_idxs)                       # [B,T]

    query_above_window = query > events.window_start.reshape(-1, 1)
    query_below_window = query <= events.window_end.reshape(-1, 1)
    query_within_window = query_above_window & query_below_window
    query_within_window = query_within_window.type(mask.dtype)
    mask = mask * query_within_window

    return (prev_times, prev_times_idxs), is_event, mask  # [B,T]
