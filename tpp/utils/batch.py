import torch as th


def _batchwise_fn(x, y, f):
    """For each value of `x` and `y`, compute `f(x, y)` batch-wise.

    Args:
        x (th.Tensor): [B1, B2, ... , BN, X] The first tensor.
        y (th.Tensor): [B1, B2, ... , BN, Y] The second tensor.
        f (function): The function to apply.

    Returns:
        (th.Tensor): [B1, B2, ... , BN, X, Y] A tensor containing the result of
            the function application.

    """
    if x.shape[:-1] != y.shape[:-1]:
        raise ValueError(
            "Shape of `x` ({}) incompatible with shape of y ({})".format(
                x.shape, y.shape))
    x = x.unsqueeze(-1)  # [B1, B2, ... , BN, X, 1]
    y = y.unsqueeze(-2)  # [B1, B2, ... , BN, 1, Y]
    result = f(x, y)     # [B1, B2, ... , BN, X, Y]
    return result


def _product(x, y):
    return x * y


def batchwise_product(x, y):
    """For each value of `x` and `y`, compute `x * y` batch-wise.

    Args:
        x (th.Tensor): [B1, B2, ... , BN, X] The first tensor.
        y (th.Tensor): [B1, B2, ... , BN, Y] The second tensor.

    Returns:
        (th.Tensor): [B1, B2, ... , BN, X, Y] A tensor containing the result of
            x * y.

    """
    return _batchwise_fn(x, y, f=_product)


def _difference(x, y):
    return x - y


def batchwise_difference(x, y):
    """For each value of `x` and `y`, compute `x - y` batch-wise.

    Args:
        x (th.Tensor): [B1, B2, ... , BN, X] The first tensor.
        y (th.Tensor): [B1, B2, ... , BN, Y] The second tensor.

    Returns:
        (th.Tensor): [B1, B2, ... , BN, X, Y] A tensor containing the result of
            x - y.

    """
    return _batchwise_fn(x, y, f=_difference)
