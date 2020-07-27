import torch as th

from tpp.utils.events import Events


def get_r_terms(events: Events, beta: th.Tensor) -> th.Tensor:
    """
    R_{m,n}(i)=sum_{j: t_j^n < t_i^m} exp(- beta_{m,n} (t_i^m - t_j^n))

    Returns:
        [B,Li,M,N] The R term for each event. Note, these are only defined when
        there are actually events. See `events.mask` for this information.

    """
    (batch_size, seq_len), marks = events.times.shape, beta.shape[0]

    r_terms = th.zeros(
        [batch_size, seq_len, marks, marks],
        dtype=events.times.dtype, device=events.times.device)
    for b in range(batch_size):
        times, labels, mask = (events.times[b], events.labels[b],
                               events.mask[b])  # [L]
        for i, (ti, mi) in enumerate(zip(times, mask)):
            for tin, label, mn in zip(times, labels, mask):
                if tin < ti:
                    delta_t = ti - tin
                    beta_m = beta[:, label]   # [M]
                    arg = - beta_m * delta_t  # [M]
                    exp = th.exp(arg)         # [M]
                    exp = exp * mi * mn
                    r_terms[b, i, :, label] = r_terms[b, i, :, label] + exp

    return r_terms
