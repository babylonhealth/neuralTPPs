import torch as th

from typing import Dict, Optional, Tuple

import tpp.utils.batch as bu

from tpp.utils.events import Events


def decoder_slow(
        events: Events,
        query: th.Tensor,
        prev_times: th.Tensor,
        is_event: th.Tensor,
        alpha: th.Tensor,
        beta: th.Tensor,
        mu: th.Tensor,
        marks: Optional[int] = 1
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
    """Compute the intensities for each query time given event
    representations.

    Args:
        events: [B,L] Times and labels of events.
        query: [B,T] Times to evaluate the intensity function.
        prev_times: [B,T] Times of events directly preceding queries.
        is_event: [B,T] A mask indicating whether the time given by
            `prev_times_idxs` corresponds to an event or not (a 1 indicates
            an event and a 0 indicates a window boundary).
        alpha: [M,M] The alpha matrix for the Hawkes process.
        beta: [M,M] The beta matrix for the Hawkes process.
        mu: [M,M] The mu vector for the Hawkes process.
        marks: The number of marks for the process (default=1).

    Returns:
        log_intensity: [B,T,M] The intensities for each query time for
            each mark (class).
        intensity_integrals: [B,T,M] The integral of the intensity from
            the most recent event to the query time for each mark.
        intensities_mask: [B,T] Which intensities are valid for further
            computation based on e.g. sufficient history available.

    """
    # TODO: Work mainly in log space, rather than exponentiating
    (b, t), seq_len, m = query.shape, events.times.shape[-1], marks

    mark_vocab = th.arange(marks, device=events.labels.device)  # [M]
    mark_vocab_r = mark_vocab.reshape(1, 1, m)  # [1,1,M]

    # The argmax step is probably redundant due to the mark index that follows
    events_labels = th.argmax(events.labels, dim=-1).long()  # [B,L]
    mark_index = events_labels.unsqueeze(dim=-1) == mark_vocab_r  # [B,L,M]
    mark_index = mark_index.type(events.times.dtype)  # [B,L,M]

    # Alpha, beta and gamma=alpha/beta coefficients for each point in
    # history affecting marks of all types
    al_event = th.matmul(mark_index, alpha.transpose(1, 0))  # [B,L,M]
    be_event = th.matmul(mark_index, beta.transpose(1, 0))  # [B,L,M]

    # Compute query time dependent terms
    be_event_r = be_event.reshape(b, 1, seq_len, m)  # [B,1,L,M]

    # Double masked so that we don't end up with infinities after the
    # exponential that we then mask
    def get_masked_exp(times, allow_equal_times=False):
        arg = bu.batchwise_difference(times, events.times)  # [B,T,L]
        if allow_equal_times:
            mask = (arg >= 0).type(arg.dtype)  # [B,T,L]
        else:
            mask = (arg > 0).type(arg.dtype)  # [B,T,L]
        mask = mask * is_event.unsqueeze(dim=-1)  # [B,T,L]
        mask = mask * events.mask.unsqueeze(dim=1)  # [B,T,L]
        arg = arg.reshape(b, t, seq_len, 1)  # [B,T,L,1]
        arg = - be_event_r * arg  # [B,T,L,M]
        arg = mask.unsqueeze(dim=-1) * arg  # [B,T,L,M]
        exp = th.exp(arg)  # [B,T,L,M]
        exp = mask.unsqueeze(dim=-1) * exp  # [B,T,L,M]
        return exp

    # exp_1 = exp( - beta_mn (t - tn) )         tn < t            [B,T,L,M]
    exp_1_masked = get_masked_exp(times=query)  # [B,T,L,M]

    # term_1 = sum_n al_mn sum_{tn < t} exp( - beta_mn (t - tn) )   [B,T,M]
    # term_2 = sum_n ga_mn sum_{tn < t} exp( - beta_mn (t - tn) )   [B,T,M]
    # ga_mn = al_mn / be_mn
    al_event_r = al_event.reshape(b, 1, seq_len, m)  # [B,1,L,M]
    term_1 = exp_1_masked * al_event_r  # [B,T,L,M]
    term_2 = term_1 / be_event_r  # [B,T,L,M]
    term_1 = th.sum(term_1, dim=2)  # [B,T,M]
    term_2 = th.sum(term_2, dim=2)  # [B,T,M]

    # exp_2 = exp( - beta_mn (ti - tn) )       tn <= ti           [B,T,L,M]
    exp_2_masked = get_masked_exp(
        times=prev_times, allow_equal_times=True)  # [B,T,L,M]

    # term_3 = sum_n ga_mn sum_{tn < ti} exp( - beta_mn (ti - tn))  [B,T,M]
    term_3 = exp_2_masked * al_event_r / be_event_r  # [B,T,L,M]
    term_3 = th.sum(term_3, dim=2)  # [B,T,M]

    # term_4 = mu (t - t_i)                    ti < t
    delta_t = query - prev_times  # [B,T]
    mu_r = mu.reshape(1, 1, marks)  # [1,1,M]
    term_4 = delta_t.unsqueeze(dim=-1) * mu_r  # [B,T,M]

    intensity = mu_r + term_1  # [B,T,M]
    log_intensity = th.log(intensity)  # [B,T,M]

    intensity_integral = term_4 + term_3 - term_2  # [B,T,M]

    intensities_mask = events.within_window(query)  # [B,T]

    return log_intensity, intensity_integral, intensities_mask, dict()
