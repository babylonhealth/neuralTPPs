import abc

import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple

from tpp.utils.events import Events


class Decoder(nn.Module, abc.ABC):
    """An decoder for a TPP.

    Args:
        name: The name of the decoder class.
        input_size: The dimensionality of the input required from the encoder.
            Defaults to `None`. This is mainly just for tracking/debugging
            ease.
        marks: The distinct number of marks (classes) for the process.
            Defaults to 1.
    """
    def __init__(
            self,
            name: str,
            input_size: Optional[int] = None,
            marks: Optional[int] = 1,
            **kwargs):
        super(Decoder, self).__init__()
        self.name = name
        self.input_size = input_size
        self.marks = marks
        if self.input_size is not None and self.input_size <= 0:
            raise ValueError("Representation dimensionality of decoder is 0.")

    @abc.abstractmethod
    def forward(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            is_event: th.Tensor,
            pos_delta_mask: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        """Compute the intensities for each query time given event
        representations.

        Args:
            events: [B,L] Times and labels of events.
            query: [B,T] Times to evaluate the intensity function.
            prev_times: [B,T] Times of events directly preceding queries.
            prev_times_idxs: [B,T] Indexes of times of events directly
                preceding queries. These indexes are of window-prepended
                events.
            pos_delta_mask: [B,T] A mask indicating if the time difference
                `query - prev_times` is strictly positive.
            is_event: [B,T] A mask indicating whether the time given by
                `prev_times_idxs` corresponds to an event or not (a 1 indicates
                an event and a 0 indicates a window boundary).
            representations: [B,L+1,D] Representations of window start and
                each event.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined. If `None`, there is no mask. Defaults to
                `None`.
            artifacts: A dictionary of whatever else you might want to return.

        Returns:
            log_intensity: [B,T,M] The intensities for each query time for
                each mark (class).
            intensity_integrals: [B,T,M] The integral of the intensity from
                the most recent event to the query time for each mark.
            intensities_mask: [B,T] Which intensities are valid for further
                computation based on e.g. sufficient history available.
            artifacts: A dictionary of whatever else you might want to return.

        """
        pass
