import torch as th

from typing import Optional

from tpp.utils.events import Events


def get_r_terms(
        events: Events,
        beta: th.Tensor,
        r_terms: Optional[th.Tensor] = None) -> th.Tensor:
    """
    R_{m,n}(i)=sum_{j: t_j^n < t_i^m} exp(- beta_{m,n} (t_i^m - t_j^n))

    Computed using the recursive definition
    R_{m,n}(i) = term_1 + term_2
    R_{m,n}(1) = 0

    term_1
    exp(- beta_{m,n} (t_i^m - t_{i-1}^m)) * R_{m,n}(i - 1)

    term_2
    sum_{j: t_{i-1}^m <= t_j^n <t_i^m} exp(- beta_{m,n} (t_i^m - t_j^n))

    Returns:
        [B,Li,M,N] The R term for each event. Note, these are only defined when
        there are actually events. See `events.mask` for this information.

    """
    marks, batch_size = events.marks, events.batch_size()

    if r_terms is None:
        r_terms = th.zeros(
            [batch_size, 1, marks, marks],
            dtype=events.times.dtype, device=events.times.device)

    i, seq_len = r_terms.shape[1], events.times.shape[-1]

    if i == seq_len:
        return r_terms

    times, labels, mask = events.times, events.labels, events.mask  # [B,L]

    ti, tim1, lim1 = times[:, i], times[:, i-1], labels[:, i-1]     # [B]
    mi, mim1 = mask[:, i], mask[:, i-1]                             # [B]
    rim1 = r_terms[:, i-1]  # [B,M,N]

    delta_t_i = (ti - tim1).unsqueeze(dim=-1).unsqueeze(dim=-1)  # [B,1,1]
    arg = - beta.unsqueeze(dim=0) * delta_t_i                    # [B,M,N]
    exp = th.exp(arg)                                            # [B,M,N]
    exp = exp * mi.unsqueeze(dim=-1).unsqueeze(dim=-1)           # [B,M,N]
    exp = exp * mim1.unsqueeze(dim=-1).unsqueeze(dim=-1)         # [B,M,N]
    ones = th.zeros(
        [batch_size, marks, marks],
        dtype=exp.dtype,
        device=exp.device)                                       # [B,M,N]
    # TODO: Vectorise this
    for b in range(batch_size):
        ones[b, :, lim1[b]] = 1.
    ri = exp * (rim1 + ones)                                     # [B,M,N]
    r_terms = th.cat([r_terms, ri.unsqueeze(dim=1)], dim=1)      # [B,I,M,N]
    return get_r_terms(events=events, beta=beta, r_terms=r_terms)
