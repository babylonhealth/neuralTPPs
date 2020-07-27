import torch as th

from torch.nn.functional import one_hot
from tqdm import tqdm
from typing import Dict, List, Optional

from tpp.utils.sequence import pad_sequence


def get_unmasked_tensor(
        x: th.Tensor, mask: th.Tensor) -> List[th.Tensor]:
    """
        Args:
            x: [B,L] The tensor to subset by the mask.
            mask: [B,L] The mask.

        Returns:
            B x [Li] tensors of non-masked values, each in principle of
            different lengths.

    """
    return [y[m.bool()] for y, m in zip(x, mask)]


def times_marked_from_times_labels(
        times: List[th.Tensor],
        labels: List[th.Tensor],
        marks: int) -> List[List[th.Tensor]]:
    """

    Args:
        times: B x [Li] tensors of non-masked times, each in principle of
            different lengths.
        labels: B x [Li] tensors of non-masked labels, each in principle of
            different lengths.
        marks: The number of marks.

    Returns:
        B x [M x [Li]] Tensors of non-masked times for each mark, each in
            principle of different lengths.

    """
    return [
        [times_i[labels_i == mark] for mark in range(marks)]
        for times_i, labels_i in zip(times, labels)]


def objects_from_events(
        events: List[List[th.Tensor]],
        marks: int,
        times_dtype: Optional[th.dtype] = th.float32,
        labels_dtype: Optional[th.dtype] = th.float32,
        verbose: Optional[bool] = False,
        device=th.device("cpu"),
) -> Dict[str, List[th.Tensor]]:
    """

    Args:
        events: D x [Dict(time=12.3, labels = tuple(1, 3))].
        times_dtype: Time datatype (default=th.float32).
        labels_dtype: Label datatype (default=th.float32). This is a flaat
            default because it needs to be used in the NLL, embeddings, ...
             with multiplications of other floating point object.
        marks: The number of classes.
        device: The device to put the objects on. Defaults to cpu.

    Returns:
        A dictionary of:
            times: B x [L] tensors of non-masked times, each in principle of
                different lengths.
            labels: B x [L,M] tensors of non-masked labels, each in principle
                of different lengths.

    """
    times_unsorted = [
        th.Tensor([x["time"] for x in r]).to(device).type(times_dtype)
        for r in events]                                             # [D,L]

    if verbose:
        events = tqdm(events, total=len(events))
    labels_unsorted = [
        th.stack([
            one_hot(
                th.Tensor(x["labels"]).to(device).long(),
                num_classes=marks).sum(0).type(labels_dtype)
            for x in r]) for r in events]    # [D,L,M]

    to_sorted_idxs = [th.argsort(x) for x in times_unsorted]          # [D,L]

    times = [
        x[idxs] for x, idxs in zip(times_unsorted, to_sorted_idxs)]   # [D,L]
    labels = [
        x[idxs] for x, idxs in zip(labels_unsorted, to_sorted_idxs)]  # [D,L,M]

    return {"times": times, "labels": labels}


def pad(x: List[th.Tensor], value, pad_len: Optional[int] = None):
    """

    Args:
        x: B x [Li] The tensors to stack together post-padding.
        value: The value to pad each tensor by to bring them all to the same
            length.
        pad_len: Length to pad all sequences to. If `None`,
            uses the longest sequence length. Default: `None`.

    Returns:
        Tensor of size [B, `pad_len`].

    """
    return pad_sequence(
        sequences=x, batch_first=True, padding_value=value, pad_len=pad_len)
