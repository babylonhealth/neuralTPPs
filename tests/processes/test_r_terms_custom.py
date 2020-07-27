import torch as th

from tpp.processes.hawkes.r_terms import get_r_terms as get_r_terms_n
from tpp.processes.hawkes.r_terms_recursive import get_r_terms as get_r_terms_r
from tpp.processes.hawkes.r_terms_recursive_v import get_r_terms as get_r_terms_v
from tpp.utils.events import get_events, get_window


def test_setup():
    padding_id = -1.
    times = th.Tensor([[1, 2, 3]]).type(th.float32)
    labels = th.Tensor([[1, 0, 0]]).type(th.long)
    mask = (times != padding_id).type(times.dtype).to(times.device)

    window_start, window_end = get_window(times=times, window=4.)
    events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end, marks=2)

    query = th.Tensor([[2.5]]).type(th.float32)
    return events, query


def test_r_terms():
    th.manual_seed(3)
    events, query = test_setup()
    beta = th.Tensor([[1., 1.], [1., 1.]]).type(th.float32)

    r_terms_naive = get_r_terms_n(events=events, beta=beta)[0]
    r_terms_recursive = get_r_terms_r(events=events, beta=beta)[0]
    r_terms_recursive_v = get_r_terms_v(events=events, beta=beta)[0]

    assert th.allclose(
        r_terms_naive, r_terms_recursive), (
        "The r term computational approaches do not match.")

    assert th.allclose(
        r_terms_naive, r_terms_recursive_v), (
        "The r term vector computational approaches do not match.")


if __name__ == "__main__":
    test_r_terms()
