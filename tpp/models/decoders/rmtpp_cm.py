import torch as th
import torch.nn as nn

from typing import Dict, Optional, Tuple

from tpp.models.decoders.base.cumulative import CumulativeDecoder
from tpp.utils.events import Events
from tpp.utils.index import take_3_by_2, take_2_by_2
from tpp.utils.stability import epsilon


class RMTPPCmDecoder(CumulativeDecoder):
    """Analytic decoder process, uses a closed form for the intensity
    integeral. Has a closed form for the intensity but we compute using the
    gradient. This is just a check.
    See https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf.
    Args:
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            multi_labels: Optional[bool] = False,
            model_log_cm: Optional[bool] = False,
            do_zero_subtraction: Optional[bool] = True,
            marks: Optional[int] = 1,
            **kwargs):
        super(RMTPPCmDecoder, self).__init__(
            name="rmtpp-cm",
            do_zero_subtraction=do_zero_subtraction,
            model_log_cm=model_log_cm,
            input_size=marks+1,
            marks=marks)
        self.multi_labels = multi_labels
        self.w = nn.Parameter(th.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.w, b=0.001)

    def cum_intensity(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None,
            update_running_stats: Optional[bool] = True
    ) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Compute the cumulative log intensity and a mask
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
            representations: [B,L+1,D] Representations of each event.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined. If `None`, there is no mask. Defaults to
                `None`.
            artifacts: A dictionary of whatever else you might want to return.
            update_running_stats: whether running stats are updated or not.

        Returns:
            intensity_integral: [B,T,M] The cumulative intensities for each
                query time for each mark (class).
            intensities_mask: [B,T]   Which intensities are valid for further
                computation based on e.g. sufficient history available.
            artifacts: Some measures.
        """
        query_representations = take_3_by_2(
            representations, index=prev_times_idxs)                   # [B,T,D]

        v_h_t = query_representations[:, :, 0]                        # [B,T]
        v_h_m = query_representations[:, :, 1:]                       # [B,T,M]

        w_delta_t = self.w * (query - prev_times)                     # [B,T]

        if self.multi_labels:
            p_m = th.sigmoid(v_h_m)  # [B,T,M]
        else:
            p_m = th.softmax(v_h_m, dim=-1)  # [B,T,M]
        regulariser = epsilon(dtype=p_m.dtype, device=p_m.device)
        p_m = p_m + regulariser

        intensity_mask = pos_delta_mask                                 # [B,T]
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)            # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        exp_1, exp_2 = w_delta_t, v_h_t                                 # [B,T]

        # Avoid exponentiating to get masked infinity - seems to induce an
        # error in gradient calculationg if we use the numerically stable one.
        exp_1, exp_2 = exp_1 * intensity_mask, exp_2 * intensity_mask   # [B,T]

        if self.model_log_cm:
            base_intensity_itg = -th.log(self.w) + exp_2 + th.log(
                th.exp(exp_1) - 1. + 1e-30)
        else:
            base_intensity_itg = th.exp(exp_1 + exp_2) - th.exp(exp_2)
            base_intensity_itg = base_intensity_itg / self.w            # [B,T]
            base_intensity_itg = th.relu(base_intensity_itg)

        marked_intensity_itg = base_intensity_itg.unsqueeze(dim=-1)   # [B,T,1]
        if self.model_log_cm:
            marked_intensity_itg = marked_intensity_itg + p_m         # [B,T,M]
        else:
            marked_intensity_itg = marked_intensity_itg * p_m         # [B,T,M]

        artifacts_decoder = {
            "base_intensity_integral": base_intensity_itg,
            "mark_probability": p_m}
        if artifacts is None:
            artifacts = {'decoder': artifacts_decoder}
        else:
            artifacts['decoder'] = artifacts_decoder

        return (marked_intensity_itg,
                intensity_mask, artifacts)               # [B,T,M], [B,T], Dict
