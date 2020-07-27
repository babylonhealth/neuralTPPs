import abc

import torch as th

from typing import Optional, Tuple, Dict

from tpp.models.decoders.base.variable_history import VariableHistoryDecoder
from tpp.models.base.process import Events
from tpp.utils.stability import check_tensor


class MCDecoder(VariableHistoryDecoder, abc.ABC):
    """Decoder based on Monte Carlo method. Here, the intensity is specified,
    but its cumulative function is determined by a Monte Carlo estimation.

    Args:
        name: The name of the decoder class.
        mc_prop_est: Proportion of numbers of samples for the MC method,
            compared to the size of the input. Defaults to 1.
        input_size: The dimensionality of the input required from the encoder.
            Defaults to `None`. This is mainly just for tracking/debugging
            ease.
        emb_dim: Size of the embeddings. Defaults to 1.
        temporal_scaling: Scaling parameter for temporal encoding
        encoding: Way to encode the queries: either times_only, marks_only,
                  concatenate or temporal_encoding. Defaults to times_only
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(self,
                 name: str,
                 mc_prop_est: Optional[float] = 1.,
                 input_size: Optional[int] = None,
                 emb_dim: Optional[int] = 1,
                 temporal_scaling: Optional[float] = 1.,
                 encoding: Optional[str] = "times_only",
                 time_encoding: Optional[str] = "relative",
                 marks: Optional[int] = 1,
                 **kwargs):
        super(MCDecoder, self).__init__(
            name=name,
            input_size=input_size,
            emb_dim=emb_dim,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.mc_prop_est = mc_prop_est

    @abc.abstractmethod
    def log_intensity(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None
    ) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Compute the log_intensity and a mask

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

        Returns:
            log_intensity: [B,T,M] The intensities for each query time for
                each mark (class).
            intensities_mask: [B,T]   Which intensities are valid for further
                computation based on e.g. sufficient history available.

        """
        pass

    def forward(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
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
            representations: [B,L+1,D] Representations of each event.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined. If `None`, there is no mask. Defaults to
                `None`.
            artifacts: A dictionary of whatever else you might want to return.

        Returns:
            log_intensity: [B,T,M] The intensities for each query time for
                each mark (class).
            intensity_integrals: [B,T,M] The integral of the intensity from
                the most recent event to the query time for each mark.
            intensities_mask: [B,T]   Which intensities are valid for further
                computation based on e.g. sufficient history available.

        """
        marked_log_intensity, intensity_mask, artifacts = self.log_intensity(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs,
            pos_delta_mask=pos_delta_mask,
            is_event=is_event,
            representations=representations,
            representations_mask=representations_mask,
            artifacts=artifacts)  # [B,T,M], [B,T], dict

        # Create Monte Carlo samples and sort them
        n_est = int(self.mc_prop_est)
        mc_times_samples = th.rand(
            query.shape[0], query.shape[1], n_est, device=query.device) * \
            (query - prev_times).unsqueeze(-1) + prev_times.unsqueeze(-1)
        mc_times_samples = th.sort(mc_times_samples, dim=-1).values
        mc_times_samples = mc_times_samples.reshape(
            mc_times_samples.shape[0], -1)  # [B, TxN]

        mc_marked_log_intensity, _, _ = self.log_intensity(
            events=events,
            query=mc_times_samples,
            prev_times=th.repeat_interleave(prev_times, n_est, dim=-1),
            prev_times_idxs=th.repeat_interleave(
                prev_times_idxs, n_est, dim=-1),
            pos_delta_mask=th.repeat_interleave(pos_delta_mask, n_est, dim=-1),
            is_event=th.repeat_interleave(is_event, n_est, dim=-1),
            representations=representations,
            representations_mask=representations_mask)  # [B,TxN,M]

        mc_marked_log_intensity = mc_marked_log_intensity.reshape(
            query.shape[0], query.shape[1], n_est, self.marks)  # [B,T,N,M]
        mc_marked_log_intensity = mc_marked_log_intensity * \
            intensity_mask.unsqueeze(-1).unsqueeze(-1)  # [B,T,N,M]
        marked_intensity_mc = th.exp(mc_marked_log_intensity)
        intensity_integrals = (query - prev_times).unsqueeze(-1) * \
            marked_intensity_mc.sum(-2) / float(n_est)  # [B,T,M]

        check_tensor(marked_log_intensity)
        check_tensor(intensity_integrals * intensity_mask.unsqueeze(-1),
                     positive=True)
        return (marked_log_intensity,
                intensity_integrals,
                intensity_mask,
                artifacts)  # [B,T,M], [B,T,M], [B,T], Dict
