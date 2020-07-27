import torch as th


def take_3_by_2(x: th.Tensor, index: th.LongTensor) -> th.Tensor:
    """Index into a rank 3 tensor with a rank 2 tensor. Specifically, replace
    each index I with the corresponding indexed D-dimensional vector, where I
    specifies the location in L, batch-wise.

    Args:
        x: [B,L,D] The D-dimensional vectors to be indexed.
        index: [B,I] The indexes.

    Returns:
        [B,I,D] The indexed tensor.

    """
    b, l, d = x.shape

    batch_idx_shift = th.arange(start=0, end=b, device=x.device) * l * d
    batch_idx_shift = batch_idx_shift.reshape([b, 1, 1])

    rep_idxs = th.arange(start=0, end=d, device=x.device).reshape([1, 1, d])

    idxs_shift = batch_idx_shift + rep_idxs                      # [B,1,D]

    idxs_shifted = index.unsqueeze(dim=-1) * d                   # [B,I,1]
    idxs_shifted = idxs_shifted + idxs_shift                     # [B,I,D]

    return th.take(x, index=idxs_shifted)                        # [B,I,D]


def take_3_by_1(x: th.Tensor, index: th.LongTensor) -> th.Tensor:
    return take_3_by_2(x, index=index.unsqueeze(dim=1)).squeeze(dim=1)


def take_2_by_1(x: th.Tensor, index: th.LongTensor) -> th.Tensor:
    """Index into a rank 2 tensor with a rank 1 tensor. Specifically, replace
    each index B with the corresponding indexed D-dimensional vector, where I
    specifies the location in L, batch-wise.

    Args:
        x: [B,D] The D-dimensional vectors to be indexed.
        index: [B] The indexes.

    Returns:
        [B] The indexed tensor.

    """
    b, d = x.shape
    batch_idx_shift = th.arange(start=0, end=b, device=x.device) * d      # [B]
    idxs_shifted = index + batch_idx_shift                                # [B]
    return th.take(x, index=idxs_shifted)                                 # [B]


def take_2_by_2(x: th.Tensor, index: th.LongTensor) -> th.Tensor:
    """Index into a rank 2 tensor with a rank 2 tensor. Specifically, replace
    each index B with the corresponding indexed D-dimensional vector, where I
    specifies the location in L, batch-wise.

    Args:
        x: [B,D] The D-dimensional vectors to be indexed.
        index: [B,I] The indexes.

    Returns:
        [B,I] The indexed tensor.

    """
    b, d = x.shape
    batch_idx_shift = th.arange(start=0, end=b, device=x.device) * d    # [B]
    idxs_shifted = index + batch_idx_shift.reshape([b, 1])              # [B,I]
    return th.take(x, index=idxs_shifted)                               # [B,I]


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))
