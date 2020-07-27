import torch as th

from tpp.utils.history_bst import get_prev_times
from tpp.utils.index import unravel_index
from tpp.utils.test import get_test_events_query
from tpp.processes.hawkes_fast import decoder_fast
from tpp.processes.hawkes_slow import decoder_slow


def get_fast_slow_results(
        queries=1, marks=2, max_seq_len=10, batch_size=1,
        dtype=th.float32):
    events, query = get_test_events_query(
        marks=marks, batch_size=batch_size, max_seq_len=max_seq_len,
        queries=queries, dtype=dtype)
    alpha = th.rand([marks, marks], dtype=dtype)
    beta = th.rand([marks, marks], dtype=dtype)
    mu = th.rand([marks], dtype=dtype)

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

    a = 0

    assert th.allclose(
        masked_intensity_itg_fast,
        masked_intensity_itg_slow,
        atol=1.e-3), (
        "Intensity integrals do not match. "
        "Max abs difference: {} occurs at {}".format(
            th.max(th.abs(masked_intensity_itg_fast -
                          masked_intensity_itg_slow)),
        unravel_index(
            th.argmax(
                th.abs(masked_intensity_itg_fast -
                       masked_intensity_itg_slow)),
            shape=masked_intensity_itg_fast.shape)))


if __name__ == "__main__":
    # for i in range(10000):
    #     print(i)
    #     th.manual_seed(i)
    #     test_fast_intensity()
    th.manual_seed(9943)
    test_fast_intensity()
