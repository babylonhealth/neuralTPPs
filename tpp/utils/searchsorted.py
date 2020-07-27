import torch as th

from torchsearchsorted import searchsorted as ss

from typing import Optional


def searchsorted(
        a: th.Tensor, v: th.Tensor, mask: Optional[th.Tensor] = None):
    """
        Args:
            a: [B,L] The row sorted array to tree sort batch-wise.
            v: [B,T] The queries for the tree sort.
            mask: [B,L] A mask for `a`. Default is no mask.
        Returns:
            [B,T] Indices such that a[i, j-1] < v[i] <= a[i, j].
    """
    if mask is not None:
        # Calculate mask shift such that a is still ordered, and that the index
        # of anything above non-padded values of a will be in the index one
        # above the final non-padded value of a
        mask_shift = 1 - mask
        mask_shift = th.cumsum(mask_shift, dim=-1)
        min_a, max_v = th.min(a), th.max(v)
        shift_value = max_v - min_a + 1
        mask_shift = mask_shift * shift_value
        a = a + mask_shift
    idxs = ss(a=a, v=v)
    return idxs


def searchsorted_marked(
        a: th.Tensor,
        v: th.Tensor,
        mask: Optional[th.Tensor] = None):
    """
        Args:
            a: [B,M,L] The row sorted array to tree sort batch-wise.
            v: [B,T] The queries for the tree sort.
            mask: [B,M,L] A mask for `a`. Default is no mask.
        Returns:
            [B,M,T] Indices such that a[i, j-1] < v[i] <= a[i, j].
    """
    (b, marks), t = a.shape[:-1], v.shape[-1]

    result = th.zeros(size=(b, marks, t), dtype=th.long, device=a.device)

    for m in range(marks):
        a_m = a[:, m, :]
        if mask is not None:
            m_m = mask[:, m, :]
        else:
            m_m = None
        result[:, m, :] = searchsorted(a=a_m, v=v, mask=m_m)

    return result
