import torch as th

from tpp.utils.stability import subtract_exp


def test_stability():
    a, b, c = th.tensor(8.1), th.tensor(0.0), th.tensor(0.0)
    naive_subtraction_1 = th.exp(a) - th.exp(b)
    safe_subtraction_1 = subtract_exp(a, b)

    naive_subtraction_2 = th.exp(b) - th.exp(a)
    safe_subtraction_2 = subtract_exp(b, a)

    naive_subtraction_3 = th.exp(c) - th.exp(c)
    safe_subtraction_3 = subtract_exp(c, c)

    assert th.isclose(naive_subtraction_1, safe_subtraction_1)
    assert th.isclose(naive_subtraction_2, safe_subtraction_2)
    assert th.isclose(naive_subtraction_3, safe_subtraction_3)


if __name__ == "__main__":
    test_stability()
