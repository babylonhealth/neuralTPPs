import time
import torch as th

from tpp.utils.events import Events


def get_r_terms(events: Events, beta: th.Tensor) -> th.Tensor:
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
        [B,L,M,N] The R term for each event. Note, these are only defined when
        there are actually events. See `events.mask` for this information.

    """
    marks, batch_size = events.marks, events.batch_size()

    times, mask = events.times, events.mask    # [B,L]
    labels = th.argmax(events.labels, dim=-1)  # [B,L]
    seq_len = times.shape[-1]

    ti, tim1, lim1 = times[:, 1:], times[:, :-1], labels[:, :-1]  # [B,L-1]
    mi = mask[:, 1:] * mask[:, :-1]                               # [B,L-1]

    delta_t_i = ti - tim1                                         # [B,L-1]
    delta_t_i = delta_t_i.unsqueeze(-1).unsqueeze(-1)             # [B,L-1,1,1]
    arg = - beta.unsqueeze(0).unsqueeze(0) * delta_t_i            # [B,L-1,M,N]
    arg = arg * mi.unsqueeze(-1).unsqueeze(-1)                    # [B,L-1,M,N]
    exp = th.exp(arg)                                             # [B,L-1,M,N]
    exp = exp * mi.unsqueeze(-1).unsqueeze(-1)                    # [B,L-1,M,N]

    mark_range = th.arange(end=marks, device=lim1.device, dtype=lim1.dtype)
    ones = (lim1.unsqueeze(dim=-1) ==
            mark_range.unsqueeze(dim=0).unsqueeze(dim=0))    # [B,L-1,M]
    ones = ones.unsqueeze(dim=2).repeat(1, 1, marks, 1)  # [B,L-1,M,N]
    ones = ones.type(exp.dtype)

    r_terms = [th.zeros(
        batch_size, marks, marks,
        dtype=exp.dtype, device=exp.device)]
    for i in range(1, seq_len):
        r_term_i = r_terms[i-1] + ones[:, i-1]
        r_term_i = exp[:, i-1] * r_term_i
        r_terms.append(r_term_i)
    r_terms = th.stack(r_terms, dim=1)

    return r_terms
