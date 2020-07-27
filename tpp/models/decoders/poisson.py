import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple

from tpp.models.decoders.base.decoder import Decoder
from tpp.utils.events import Events
from tpp.utils.nnplus import non_neg_param


class PoissonDecoder(Decoder):
    """A parametric Hawkes Process decoder.

    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(self, marks: Optional[int] = 1, **kwargs):
        super(PoissonDecoder, self).__init__(name="poisson", marks=marks)
        self.mu = nn.Parameter(th.Tensor(self.marks))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.mu)

    def forward(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.LongTensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        batch_size, n_queries = query.shape
        self.mu.data = non_neg_param(self.mu.data)
        mu = self.mu.reshape([1, 1, self.marks])              # [1,1,M]
        mu = mu.repeat([batch_size, n_queries, 1])            # [B,T,M]

        delta_t = query - prev_times                                  # [B,T]
        delta_t = delta_t.unsqueeze(dim=-1)                           # [B,T,1]
        intensity_integrals = mu * delta_t                # [B,T,M]

        intensities_mask = events.within_window(query)

        return th.log(mu), intensity_integrals, intensities_mask, dict()
