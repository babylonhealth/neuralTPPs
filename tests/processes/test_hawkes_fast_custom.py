import torch as th

from tpp.utils.history_bst import get_prev_times

from tpp.processes.hawkes_fast import decoder_fast
from tpp.processes.hawkes_slow import decoder_slow
from tpp.utils.events import get_events, get_window


def get_fast_slow_results():
    padding_id = -1.
    times = th.Tensor([[1, 2, -1., -1.]]).type(th.float32)
    labels = th.Tensor([[1, 0, 0, 1]]).type(th.long)
    mask = (times != padding_id).type(times.dtype).to(times.device)
    marks = 2
    query = th.Tensor([[2.5, 7.]]).type(th.float32)

    window_start, window_end = get_window(times=times, window=4.)

    beta = th.Tensor([2, 1, 1, 3]).reshape(marks, marks).float()
    alpha = th.Tensor([1, 2, 1, 1]).reshape(marks, marks).float()

    mu = th.zeros(size=[marks], dtype=th.float32) + 3.00001

    events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end, marks=2)

    (prev_times, _), is_event, _ = get_prev_times(
        query=query, events=events, allow_window=True)

    results_fast = decoder_fast(
        events=events,
        query=query,
        prev_times=prev_times,
        is_event=is_event,
        alpha=alpha,
        beta=beta,
        mu=mu,
        marks=marks)

    results_slow = decoder_slow(
        events=events,
        query=query,
        prev_times=prev_times,
        is_event=is_event,
        alpha=alpha,
        beta=beta,
        mu=mu,
        marks=marks)

    return results_fast, results_slow


def test_fast_intensity():
    results_fast, results_slow = get_fast_slow_results()
    log_int_f, int_itg_f, mask_f, _ = results_fast  # [B,T,M], [B,T]
    log_int_s, int_itg_s, mask_s, _ = results_slow  # [B,T,M], [B,T]

    assert th.all(mask_f == mask_s), "Masks do not match"

    masked_intensity_fast = log_int_f * mask_f.unsqueeze(dim=-1)
    masked_intensity_slow = log_int_s * mask_s.unsqueeze(dim=-1)

    assert th.allclose(
        masked_intensity_fast,
        masked_intensity_slow,
        atol=1.e-5), "Intensities do not match."

    masked_intensity_itg_fast = int_itg_f * mask_f.unsqueeze(dim=-1)
    masked_intensity_itg_slow = int_itg_s * mask_s.unsqueeze(dim=-1)

    assert th.allclose(
        masked_intensity_itg_fast,
        masked_intensity_itg_slow,
        atol=1.e-5), "Intensity integrals do not match."


if __name__ == "__main__":
    test_fast_intensity()
