import torch as th

from typing import Optional


def epsilon(eps=1e-30, dtype=th.float32, device=None):
    return th.tensor(eps, dtype=dtype, device=device)


def epsilon_like(x, eps=1e-3):
    return th.zeros_like(x) + th.tensor(eps, dtype=x.dtype, device=x.device)


def log_sub_exp(
        a: th.Tensor,
        b: th.Tensor,
        regularize: Optional[bool] = True) -> th.Tensor:
    """Compute log(exp(a) - exp(b)) safely."""
    if th.any(a < b):
        raise ValueError(
            "All elements of exponent `a` ({}) must be at least as large "
            "as `b` ({}).".format(a, b))
    max_a = th.max(a)
    a, b = a - max_a, b - max_a
    arg = th.exp(a) - th.exp(b)
    if regularize:
        arg = arg + epsilon(dtype=arg.dtype, device=arg.device)
    return th.log(arg) + max_a


def subtract_exp(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    """Compute th.exp(a) - th.exp(b) safely."""
    # Make sure we dont' have b > a
    b_gt_a = (b > a).type(a.dtype)

    # Anywhere it's not true, replace a with b and b with a
    a1 = a + b_gt_a * (b - a)
    b1 = b + b_gt_a * (a - b)

    log_subtraction = log_sub_exp(a=a1, b=b1)
    result = th.exp(log_subtraction)

    # Swap the signs around where a and b were swapped
    result = result * th.pow(-1, b_gt_a)
    return result


def check_tensor(
        t: th.Tensor,
        positive: Optional[bool] = False,
        strict: Optional[bool] = False):
    """Check if a tensor is valid """
    assert th.isnan(t).sum() == 0
    assert th.isinf(t).sum() == 0
    if positive:
        if strict:
            assert (t <= 0.).sum() == 0
        else:
            assert (t < 0.).sum() == 0
