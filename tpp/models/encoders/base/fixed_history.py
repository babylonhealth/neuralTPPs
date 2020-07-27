import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple

from tpp.utils.events import Events
from tpp.models.encoders.base.encoder import Encoder
from tpp.utils.history import build_histories


class FixedHistoryEncoder(Encoder):
    """A parametric encoder process with a fixed history size representation.

    Args
        name: The name of the encoder class.
        net: The network used to encode the history.
        history_size: The size of each history.
        output_size: The output size (dimensionality) of the representations
            formed by the encoder.
        marks: The distinct number of marks (classes) for the process.
            Defaults to 1.
    """
    def __init__(
            self,
            name: str,
            net: nn.Module,
            output_size: int,
            history_size: Optional[int] = 2,
            marks: Optional[int] = 1,
            **kwargs):
        super(FixedHistoryEncoder, self).__init__(
            name=name, output_size=output_size, marks=marks, **kwargs)
        self.net = net
        self.history_size = history_size

    def get_history_representations(
            self, events: Events) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the history vectors.

        Args:
            events: [B,L] Times and labels of events.

        Returns:
            histories: [B,L+1,H] Histories of each event.
            histories_mask: [B,L+1] Mask indicating which histories
                are well-defined.

        """
        histories = events.times.unsqueeze(dim=-1)  # [B,L,1]
        histories_mask = events.mask  # [B,L]
        batch_size, _ = histories_mask.shape

        if self.history_size > 1:
            h_prev, h_prev_mask = build_histories(
                query=events.times, events=events,
                history_size=self.history_size - 1)          # [B,L,H-1], [B,L]
            histories = th.cat([h_prev, histories], dim=-1)  # [B,L,H]
            histories_mask = histories_mask * h_prev_mask    # [B,L]

        # Add on a masked history for the window start representation
        window_history = th.zeros(
            [batch_size, 1, self.history_size],
            dtype=histories.dtype,
            device=histories.device)
        histories = th.cat([window_history, histories], dim=1)  # [B,L+1,H]
        window_mask = th.zeros(
            [batch_size, 1],
            dtype=histories_mask.dtype,
            device=histories.device)
        histories_mask = th.cat(
            [window_mask, histories_mask], dim=1)               # [B,L+1]

        return histories, histories_mask                   # [B,L+1,H], [B,L+1]

    def forward(self, events: Events) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Compute the (query time independent) event representations.

        Args:
            events: [B,L] Times and labels of events.

        Returns:
            representations: [B,L+1,M+1] Representations of each event.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined.

        """
        histories, histories_mask = self.get_history_representations(
            events=events)                               # [B,L+1,H] [B,L+1]
        representations = self.net(histories)            # [B,L+1,M+1]
        return (representations,
                histories_mask, dict())  # [B,L+1,M+1], [B,L+1], Dict
