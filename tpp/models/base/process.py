import abc

import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple

from tpp.utils.events import Events


class Process(nn.Module):
    def __init__(self, name: str, marks: Optional[int] = 1, **kwargs):
        """A parametric process.

    Args:
        name: The name of the process.
        marks: The distinct number of marks (classes) for the process.
            Defaults to 1.
    """
        super(Process, self).__init__()
        self.name = name
        self.marks = marks

    @abc.abstractmethod
    def intensity(
            self, query: th.Tensor, events: Events
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the intensities at query times given events.

        Args:
            query: [B,T] Sequences of query times to evaluate the intensity
                function.
            events: [B,L] Times and labels of events.

        Returns:
            intensities: [B,T,M] The intensities for each query time for each
                mark (class).
            intensity_mask: [B,T,M] Which intensities are valid for further
                computation based on e.g. sufficient history available.

        """
        pass

    @abc.abstractmethod
    def neg_log_likelihood(
            self, events: Events) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Compute the negative log likelihood of events.

        Args:
            events: [B,L] Times and labels of events.
        Returns:
            nll: [B] The negative log likelihoods for each sequence.
            nll_mask: [B] Which neg_log_likelihoods are valid for further
                computation based on e.g. at least one element in sequence has
                a contribution.
            artifacts: Other useful quantities.

        """
        pass
