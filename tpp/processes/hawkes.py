import torch as th

from torch.nn.functional import relu


def intensity_at_t(mu, alpha, sequences_padded, mask, t):
    """Finds the hawkes intensity:
    mu + alpha * sum( np.exp(-(t-s)) for s in points if s<=t )

    Args:
        mu: float
        alpha: float
        sequences_padded: 2d numpy array
        mask: 2d numpy array
        t: float

    Returns:
        1d numpy array

    """
    history_mask = (sequences_padded < t).type(sequences_padded.dtype)
    combined_mask = history_mask * mask

    intensities = sequences_padded - t
    intensities = _zero_positives(intensities)
    intensities = alpha * th.exp(intensities)
    intensities = combined_mask * intensities
    intensities = th.sum(intensities, dim=-1)
    intensities += mu

    return intensities


def _zero_positives(x):
    return - relu(-x)


def intensity_at_times(mu, alpha, sequences_padded, times, sequence_mask=None):
    """Finds the hawkes intensity:
    mu + alpha * sum( np.exp(-(t-s)) for s in points if s<=t )

    Args:
        mu: float
        alpha: float
        sequences_padded: 2d numpy array B x L
        sequence_mask: 2d numpy array B x L
        times: 2d numpy array B x T

    Returns:
        2d numpy array B x T

    """
    if sequence_mask is None:
        sequence_mask = th.ones_like(
            sequences_padded, device=sequences_padded.device)
    sequences_padded = sequences_padded.unsqueeze(dim=1)  # B x 1 x L
    sequence_mask = sequence_mask.unsqueeze(dim=1)  # B x 1 x L
    times = times.unsqueeze(dim=-1)  # B x T x 1
    history_mask = (sequences_padded < times).type(times.dtype)  # B x T x L
    history_mask = history_mask.float()
    combined_mask = history_mask * sequence_mask  # B x T x L

    intensities = sequences_padded - times     # B x T x L
    # We only care about when the quantity above is negative
    intensities = _zero_positives(intensities)
    intensities = alpha * th.exp(intensities)  # B x T x L
    intensities = combined_mask * intensities  # B x T x L
    intensities = th.sum(intensities, dim=-1)  # B x T
    intensities += mu                          # B x T

    return intensities


def neg_log_likelihood(mu, alpha, sequences_padded, sequence_mask, window):
    """Find the nll:
    
    Args:
        mu: float
        alpha: float
        sequences_padded: 2d numpy array B x L
        sequence_mask: 2d numpy array B x L
        window: float

    Returns:
        1d numpy array B

    """
    if sequence_mask is None:
        sequence_mask = th.ones_like(
            sequences_padded, device=sequences_padded.device)
    intensities = intensity_at_times(
        mu=mu, alpha=alpha, 
        sequences_padded=sequences_padded, 
        sequence_mask=sequence_mask, 
        times=sequences_padded
    )  # B x L
    intensities = th.log(intensities)
    intensities *= sequence_mask
    intensities = th.sum(intensities, dim=-1)    # B

    seq_lens = th.sum(sequence_mask, dim=-1)     # B
    
    exp_term = sequences_padded - window         # B x L
    exp_term = th.exp(exp_term)
    exp_term *= sequence_mask
    exp_term = alpha * th.sum(exp_term, dim=-1)  # B

    return - intensities + window * mu + alpha * seq_lens - exp_term  # B


def intensity_old(mu, alpha, points, t):
    """Finds the hawkes intensity:
    mu + alpha * sum( np.exp(-(t-s)) for s in points if s<=t )
    """
    p = points[points < t]
    p = th.exp(p - t) * alpha
    return mu + th.sum(p)


def neg_log_likelihood_old(mu, alpha, points, window):
    intensities = sum(
        [th.log(intensity_old(mu, alpha, points, point)) for
         point in points])

    n_points = len(points)

    neg_log_l = (
            - intensities
            + window * mu
            + alpha * n_points
            - alpha * sum(th.exp(points - window))
    )

    return neg_log_l
