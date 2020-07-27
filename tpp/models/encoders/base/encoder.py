import abc

import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple

from tpp.utils.events import Events


class Encoder(nn.Module, abc.ABC):
    """An encoder for a TPP.

    Args:
        name: The name of the encoder class.
        output_size: The output size (dimensionality) of the representations
            formed by the encoder.
        marks: The distinct number of marks (classes) for the process.
            Defaults to 1.

    """
    def __init__(
            self,
            name: str,
            output_size: int,
            marks: Optional[int] = 1,
            **kwargs):
        super(Encoder, self).__init__()
        self.name = name
        self.marks = marks
        self.output_size = output_size

    @abc.abstractmethod
    def forward(self, events: Events) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Compute the (query time independent) event representations.

        Args:
            events: [B,L] Times and labels of events.

        Returns:
            representations: [B,L+1,D] Representations of each event,
                including the window start.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined.
            artifacts: A dictionary of whatever else you might want to return.

        """
        pass
