import torch as th

from tpp.processes.hawkes.r_terms import get_r_terms as get_r_terms_n
from tpp.processes.hawkes.r_terms_recursive import get_r_terms as get_r_terms_r
from tpp.processes.hawkes.r_terms_recursive_v import get_r_terms as get_r_terms_v
from tpp.utils.test import get_test_events_query


def test_r_terms(device=th.device("cpu")):
    th.manual_seed(3)
    marks, batch_size = 12, 23
    max_seq_len = 138
    queries = 13
    events, _ = get_test_events_query(
        marks=marks, batch_size=batch_size, max_seq_len=max_seq_len,
        queries=queries, device=device)
    beta = th.rand([marks, marks], dtype=th.float32).to(device)

    r_terms_naive = get_r_terms_n(events=events, beta=beta)
    r_terms_recursive = get_r_terms_r(events=events, beta=beta)
    r_terms_recursive_v = get_r_terms_v(events=events, beta=beta)

    assert th.allclose(
        r_terms_naive, r_terms_recursive), (
        "The r term computational approaches do not match.")

    assert th.allclose(
        r_terms_naive, r_terms_recursive_v), (
        "The r term vector computational approaches do not match.")


if __name__ == "__main__":
    test_r_terms()
    if th.cuda.is_available():
        test_r_terms(device=th.device("cuda"))
