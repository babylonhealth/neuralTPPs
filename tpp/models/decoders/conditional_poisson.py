import torch as th

from typing import Dict, Optional, Tuple, List

from tpp.models.decoders.base.decoder import Decoder

from tpp.pytorch.models import MLP

from tpp.utils.events import Events
from tpp.utils.index import take_2_by_2, take_3_by_2
from tpp.utils.stability import epsilon, check_tensor


class ConditionalPoissonDecoder(Decoder):
    """A parametric Hawkes Process decoder.

    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            units_mlp: List[int],
            activation_mlp: Optional[str] = "relu",
            dropout_mlp: Optional[float] = 0.,
            constraint_mlp: Optional[str] = None,
            activation_final_mlp: Optional[str] = "parametric_softplus",
            marks: Optional[int] = 1,
            **kwargs):
        input_size = units_mlp[0]
        if len(units_mlp) < 2:
            raise ValueError("Units of length at least 2 need to be specified")
        super(ConditionalPoissonDecoder, self).__init__(
            name="conditional-poisson",
            input_size=input_size,
            marks=marks)
        self.mlp = MLP(
            units=units_mlp[1:],
            activations=activation_mlp,
            constraint=constraint_mlp,
            dropout_rates=dropout_mlp,
            # units_mlp in this class also provides the input dimensionality
            # of the mlp
            input_shape=input_size,
            activation_final=activation_final_mlp)

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
        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)        # [B,T,units_mlp[0]]

        outputs = self.mlp(history_representations)          # [B,T,M]
        outputs = outputs + epsilon(dtype=outputs.dtype, device=outputs.device)

        delta_t = query - prev_times                       # [B,T]
        delta_t = delta_t.unsqueeze(dim=-1)                # [B,T,1]
        intensity_integrals = outputs * delta_t            # [B,T,M]

        intensity_mask = pos_delta_mask  # [B,T]
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)  # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        check_tensor(outputs, positive=True, strict=True)
        check_tensor(intensity_integrals * intensity_mask.unsqueeze(-1),
                     positive=True)
        return th.log(outputs), intensity_integrals, intensity_mask, artifacts
