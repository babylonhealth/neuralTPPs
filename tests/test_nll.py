import numpy as np
import torch as th

from tpp.processes.hawkes import neg_log_likelihood_old as nll_old
from tpp.processes.hawkes import neg_log_likelihood as nll_new

from tpp.utils.keras_preprocessing.sequence import pad_sequences


def test_nll():
    n_seq = 10
    my_alpha = 0.7
    my_mu = 0.1
    pad_id = -1.
    my_window = 100

    my_sizes = [np.random.randint(low=1, high=10) for _ in range(n_seq)]
    my_points = [th.sort(th.rand(size=[s])).values for s in my_sizes]
    
    nll_1 = [nll_old(mu=my_mu, alpha=my_alpha, points=p, window=my_window)
             for p in my_points]
    nll_1 = th.stack(nll_1, dim=0)

    my_points_padded = pad_sequences(
        my_points, padding="post", dtype=np.float32, value=pad_id)
    my_points_padded = th.from_numpy(my_points_padded)
    my_mask = (my_points_padded != pad_id).type(my_points_padded.dtype)

    nll_2 = nll_new(
        mu=my_mu, alpha=my_alpha,
        sequences_padded=my_points_padded, sequence_mask=my_mask,
        window=my_window)

    assert np.allclose(nll_1, nll_2)


if __name__ == "__main__":
    test_nll()
