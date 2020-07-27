import torch as th

from typing import NamedTuple, Optional, Tuple


class Events(NamedTuple):
    """All event information.

    Props:
        times: [B,L] The times of the events.
        times_first: [B] The time of the first event.
        times_first_idx: [B] The index (into [L]) of the first event.
        times_final: [B] The time of the final event.
        times_final_idx: [B] The index (into [L]) of the final event.
        mask: [B,L] The mask indicating which times are defined.
        labels: [B,L] The labels for each time.
        window_start: [B] The start of the observation window.
        window_end: [B] The end of the observation window.
        first_event_on_window: [B] Boolean indicating if the first event
            lies precisely on the window.
        final_event_on_window: [B] Boolean indicating if the final event
            lies precisely on the window.

    """
    times: th.Tensor                           # [B,L]
    times_first: th.Tensor                     # [B]
    times_first_idx: th.LongTensor             # [B]
    times_final: th.Tensor                     # [B]
    times_final_idx: th.LongTensor             # [B]
    mask: th.Tensor                            # [B,L]
    labels: th.LongTensor                      # [B,L]
    marks: int
    window_start: th.Tensor                    # [B]
    window_end: th.Tensor                      # [B]
    first_event_on_window: th.Tensor           # [B]
    final_event_on_window: th.Tensor           # [B]

    def batch_size(self):
        return self.times.shape[0]

    def get_times(
            self,
            prepend_window: Optional[bool] = False,
            postpend_window: Optional[bool] = False):
        batch_size = self.batch_size()
        times = [self.times]
        w_reshape, w_repeats = (batch_size, 1), (1, 1)

        window_start = self.window_start.reshape(w_reshape).repeat(w_repeats)
        window_end = self.window_end.reshape(w_reshape).repeat(w_repeats)
        # Consider that each mark has a starting window event, but they're all
        # really the same event.
        if prepend_window:
            times = [window_start] + times
        # Consider that each mark has an ending window event, but they're all
        # really the same event.
        if postpend_window:
            times = times + [window_end]

        return th.cat(times, dim=-1)

    def get_mask(
            self,
            prepend_window: Optional[bool] = False,
            postpend_window: Optional[bool] = False):
        batch_size = self.batch_size()
        mask = [self.mask]
        o_reshape, o_repeats = (batch_size, 1), (1, 1)

        ones = th.ones_like(
            self.window_start,
            dtype=self.mask.dtype,
            device=self.window_start.device)
        ones = [ones.reshape(o_reshape).repeat(o_repeats)]

        if prepend_window:
            mask = ones + mask
        if postpend_window:
            mask = mask + ones
        return th.cat(mask, dim=-1)

    def within_window(self, query: th.Tensor):
        after_window = query > self.window_start.unsqueeze(dim=-1)  # [B,T]
        before_window = query <= self.window_end.unsqueeze(dim=-1)  # [B,T]
        result = after_window & before_window                       # [B,T]
        result = result.type(query.dtype)                           # [B,T]
        return result


def get_events(
        times: th.Tensor,
        mask: th.Tensor,
        labels: Optional[th.LongTensor] = None,
        window_start: Optional[th.Tensor] = None,
        window_end: Optional[th.Tensor] = None
) -> Events:
    """

    Args:
        times: [B,L] The times of the events.
        mask: [B,L] The mask indicating which times are defined.
        labels: [B,L] The labels for each time. If `None`, the labels will be
            all 0's.
        window_start: [B] The start of the observation window. If `None`, this
            will be taken as the first observed event.
        window_end: The end of the observation window. If `None`, this
            will be taken as the final observed event.

    Returns:
        events: The events named tuple, containing:
            times: [B,L] The times of the events.
            times_first: [B] The time of the first event.
            times_first_idx: [B] The index (into [L]) of the first event.
            times_final: [B] The time of the final event.
            times_final_idx: [B] The index (into [L]) of the final event.
            mask: [B,L] The mask indicating which times are defined.
            labels: [B,L] The labels for each time.
            window_start: [B] The start of the observation window.
            window_end: [B] The end of the observation window.
            first_event_on_window: [B] Boolean indicating if the first event
                lies precisely on the window.
            final_event_on_window: [B] Boolean indicating if the final event
                lies precisely on the window.

    """
    masked_times = mask * times                                         # [B,L]
    times_final, times_final_idx = th.max(masked_times, dim=-1)           # [B]

    inverted_mask = 1. - mask                                           # [B,L]
    masked_time_shift = times_final.unsqueeze(dim=-1) * inverted_mask   # [B,L]
    masked_times_shifted = masked_times + masked_time_shift             # [B,L]
    times_first, times_first_idx = th.min(masked_times_shifted, dim=-1)  # [B]

    if window_start is None:
        window_start = times_first
    if window_end is None:
        window_end = times_final

    first_event_on_window = times_first == window_start
    final_event_on_window = times_final == window_end

    if labels is None:
        labels = th.ones(
            size=times.shape,
            dtype=times.dtype,
            device=times.device).unsqueeze(dim=-1)

    assert len(labels.shape) == 3
    assert times.shape == labels.shape[:-1]

    marks = labels.shape[-1]

    events = Events(
        times=times,
        times_first=times_first, times_first_idx=times_first_idx,
        times_final=times_final, times_final_idx=times_final_idx,
        mask=mask,
        labels=labels,
        marks=marks,
        window_start=window_start, window_end=window_end,
        first_event_on_window=first_event_on_window,
        final_event_on_window=final_event_on_window)

    return events


def get_window(
        times: th.Tensor, window: float) -> Tuple[th.Tensor, th.Tensor]:
    batch_size = times.shape[0]
    window_start = th.zeros([batch_size]).type(times.dtype).to(times.device)
    if window is None:
        window_end = None
    else:
        window_end = (th.ones([batch_size]) * window).type(
            times.dtype).to(times.device)
    return window_start, window_end
