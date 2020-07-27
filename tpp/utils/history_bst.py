import torch as th

from typing import Optional, Tuple

from tpp.utils.events import Events
from tpp.utils.index import take_2_by_2
from tpp.utils.searchsorted import searchsorted


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
    event_times = events.get_times(prepend_window=allow_window)
    event_mask = events.get_mask(prepend_window=allow_window)
    prev_times_idxs = searchsorted(
        a=event_times, v=query, mask=event_mask)
    prev_times_idxs = prev_times_idxs - 1
    prev_times = take_2_by_2(event_times, index=prev_times_idxs)        # [B,T]
    mask = (prev_times_idxs >= 0).type(event_times.dtype)               # [B,T]

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

    query_within_window = events.within_window(query=query)            # [B,T]
    mask = mask * query_within_window                                  # [B,T]

    return (prev_times, prev_times_idxs), is_event, mask               # [B,T]
