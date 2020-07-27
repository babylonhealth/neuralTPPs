import numpy as np
import torch as th

from tpp.utils.searchsorted import searchsorted


def get_test_data(rows=7, data_cols=9, query_cols=11, padding_id=-1.):
    x_padded = th.rand(rows, data_cols).float()
    x_padded = th.sort(x_padded, dim=-1).values
    query = th.rand(rows, query_cols).float()
    lens = th.randint(low=1, high=data_cols, size=[rows]).long()
    for xi, li in zip(x_padded, lens):
        xi[li:] = padding_id
    mask = (x_padded != padding_id).type(x_padded.dtype)
    x = [xi[mi.bool()] for xi, mi in zip(x_padded, mask)]
    return x, x_padded, mask, query


def simple_approach(x, query):
    return [np.searchsorted(a=xi, v=qi) for xi, qi in zip(x, query)]


def test_searchsorted():
    x, x_padded, mask, query = get_test_data()
    simple_answer = simple_approach(x=x, query=query)
    simple_answer = th.stack(simple_answer, dim=0)
    batched_answer = searchsorted(a=x_padded, v=query, mask=mask)
    assert th.all(simple_answer == batched_answer)


if __name__ == "__main__":
    test_searchsorted()
