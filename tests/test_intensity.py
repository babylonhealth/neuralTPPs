import numpy as np
import torch as th

from tpp.processes.hawkes import intensity_old as intensity_old
from tpp.processes.hawkes import intensity_at_t as intensity_new
from tpp.processes.hawkes import intensity_at_times

from tpp.utils.keras_preprocessing.sequence import pad_sequences


def test_intensity():
    n_seq = 10
    my_alpha = 0.7
    my_mu = 0.1
    pad_id = -1.
    my_t = 0.5

    my_sizes = [np.random.randint(low=1, high=10) for _ in range(n_seq)]
    my_points = [th.sort(th.rand(size=[s])).values for s in my_sizes]

    intensity_1 = [
        intensity_old(mu=my_mu, alpha=my_alpha, points=p, t=my_t) for p in
        my_points]
    intensity_1 = th.stack(intensity_1, dim=0)

    my_points_padded = pad_sequences(
        my_points, padding="post", dtype=np.float32, value=pad_id)
    my_points_padded = th.from_numpy(my_points_padded)
    my_mask = (my_points_padded != pad_id).type(my_points_padded.dtype)

    intensity_2 = intensity_new(
        mu=my_mu, alpha=my_alpha,
        sequences_padded=my_points_padded, mask=my_mask, t=my_t)

    assert np.allclose(intensity_1, intensity_2)
    
    
def test_intensity_general():
    n_seq = 10
    my_alpha = 0.7
    my_mu = 0.1
    pad_id = -1.

    my_seq_sizes = [np.random.randint(low=1, high=10) for _ in range(n_seq)]
    my_time_sizes = [np.random.randint(low=1, high=5) for _ in range(n_seq)]

    my_seqs = [th.sort(th.rand(size=[s])).values for s in my_seq_sizes]
    my_times = [th.sort(th.rand(size=[s])).values for s in my_time_sizes]

    my_seqs_padded = pad_sequences(
        my_seqs, padding="post", dtype=np.float32, value=pad_id)
    my_seqs_padded = th.from_numpy(my_seqs_padded)

    my_times_padded = pad_sequences(
        my_times, padding="post", dtype=np.float32, value=pad_id)
    my_times_padded = th.from_numpy(my_times_padded)

    my_seq_mask = (my_seqs_padded != pad_id).type(
        my_seqs_padded.dtype)  # B x L

    intensity_1 = [[intensity_new(
        mu=my_mu, alpha=my_alpha,
        sequences_padded=my_seqs_padded, mask=my_seq_mask, t=t)[b]
                    for t in times] for b, times in enumerate(my_times_padded)]
    intensity_1 = np.array(intensity_1)
    
    intensity_2 = intensity_at_times(
        mu=my_mu, alpha=my_alpha,
        sequences_padded=my_seqs_padded, sequence_mask=my_seq_mask, 
        times=my_times_padded)

    assert np.allclose(intensity_1, intensity_2)


if __name__ == "__main__":
    test_intensity()
    test_intensity_general()
