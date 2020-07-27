import torch as th

from typing import Optional, Tuple

from tpp.utils.events import Events
from tpp.utils.index import take_2_by_2
from tpp.utils.searchsorted import searchsorted_marked


def get_prev_times_marked(
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
        `times` is a tuple of tensor of values [B,T,M] and indices,
            [B,T,M] of the largest time value in the sequence that is strictly
            smaller than the query time value, or the window. the index only
            event indexes into the events. If the window is returned, it
            should be dealt with explicitly at encoding/decoding time.

        `is_event` is a tensor [B,T,M] that indicates whether the time
            corresponds to an event or not (a 1 indicates an event and a 0
            indicates a window boundary).

        `mask` is a tensor [B,TM] that indicates whether the time difference to
            those times what positive or not.

    """
    b, t = query.shape

    event_times = events.get_times(prepend_window=allow_window)      # [B,L]
    event_times_marked = events.get_times(
        marked=True, prepend_window=allow_window)                    # [B,M,LM]
    event_mask_marked = events.get_mask(
        marked=True, prepend_window=allow_window)                    # [B,M,LM]
    marks = events.marks

    prev_times_idxs_marked = searchsorted_marked(
        a=event_times_marked, v=query, mask=event_mask_marked)        # [B,M,T]
    prev_times_idxs_marked = prev_times_idxs_marked - 1               # [B,M,T]

    prev_times = take_2_by_2(
        event_times_marked.reshape(b * marks, -1),
        prev_times_idxs_marked.reshape(b * marks, -1)).reshape(
        b, marks, -1)                                                 # [B,M,T]

    # We want to index into the original unmarked times object, so we need to
    # use the index map
    # lm = event_mask_marked.shape[-1]
    # to_flat_idxs = events.get_to_flat_idxs(prepend_window=allow_window)
    # to_flat_idxs = to_flat_idxs.reshape(-1, lm)                      # [B*M,LM]
    # prev_times_idxs = prev_times_idxs_marked.reshape(-1, t)          # [B*M,T]
    # prev_times_idxs = take_2_by_2(
    #     to_flat_idxs, index=prev_times_idxs)                         # [B*M,T]
    #
    # prev_times_idxs_flat = prev_times_idxs.reshape(b, -1)  # [B,M*T]
    # prev_times = take_2_by_2(
    #     event_times, index=prev_times_idxs_flat)  # [B,M*T]
    # prev_times_idxs = prev_times_idxs.reshape(prev_times_idxs_marked.shape)
    # prev_times = prev_times.reshape(prev_times_idxs_marked.shape)
    # Mask based on original indexes being out of range. The new ones won't
    # be -1 as they'll pick the -1 element of the map.
    mask = (prev_times_idxs_marked >= 0).type(
        event_times_marked.dtype)                                     # [B,M,T]

    if allow_window:
        # If the first event shares a time with the window boundary, that the
        # index returned is the index of the event, rather than the window
        # boundary.
        idx_is_window = (prev_times_idxs_marked == 0).type(
            prev_times_idxs_marked.dtype)                             # [B,M,T]
        do_idx_shift = events.first_event_on_window.type(
            idx_is_window.dtype)                                      # [B]
        idx_shift = idx_is_window * do_idx_shift.reshape(-1, 1, 1)    # [B,M,T]
        prev_times_idxs = prev_times_idxs_marked + idx_shift

        # Check the indexes in case one of the window indexes became an event.
        is_event = (prev_times_idxs != 0).type(mask.dtype)            # [B,M,T]
    else:
        is_event = th.ones_like(
            prev_times_idxs_marked,
            device=event_times.device, dtype=event_times.dtype)       # [B,M,T]

    query_within_window = events.within_window(query=query)           # [B,T]
    mask = mask * query_within_window.unsqueeze(dim=1)                # [B,M,T]

    return (prev_times, prev_times_idxs), is_event, mask              # [B,M,T]
