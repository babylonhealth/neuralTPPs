import torch as th

from tpp.processes.hawkes.r_terms_recursive_v import get_r_terms
from tpp.utils.test import get_test_events_query


def run_test():
    marks = 3
    events, query = get_test_events_query(marks=marks)
    beta = th.rand([marks, marks])

    get_r_terms(events=events, beta=beta)


if __name__ == '__main__':
    run_test()
