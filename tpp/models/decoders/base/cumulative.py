import abc

import torch as th

from typing import Optional, Tuple, Dict

from tpp.models.base.process import Events
from tpp.models.decoders.base.variable_history import VariableHistoryDecoder

from tpp.pytorch.layers.log import Log

from tpp.utils.stability import epsilon, check_tensor, subtract_exp


class CumulativeDecoder(VariableHistoryDecoder, abc.ABC):
    """Decoder based on Cumulative intensity method. Here, the cumulative
       intensity is specified, but its derivative is directly computed

    Args:
        name: The name of the decoder class.
        do_zero_subtraction: If `True` the class computes
            Lambda(tau) = Lambda'(tau) - Lambda'(0)
            in order to enforce Lambda(0) = 0. Defaults to `True`.
        input_size: The dimensionality of the input required from the encoder.
            Defaults to `None`. This is mainly just for tracking/debugging
            ease.
        emb_dim: Size of the embeddings. Defaults to 1.
        encoding: Way to encode the queries: either times_only, marks_only,
                  concatenate or temporal_encoding. Defaults to times_only
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(self,
                 name: str,
                 do_zero_subtraction: Optional[bool] = True,
                 model_log_cm: Optional[bool] = False,
                 input_size: Optional[int] = None,
                 emb_dim: Optional[int] = 1,
                 encoding: Optional[str] = "times_only",
                 time_encoding: Optional[str] = "relative",
                 marks: Optional[int] = 1,
                 **kwargs):
        super(CumulativeDecoder, self).__init__(
            name=name,
            input_size=input_size,
            emb_dim=emb_dim,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.do_zero_subtraction = do_zero_subtraction
        self.model_log_cm = model_log_cm

    @abc.abstractmethod
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
            update_running_stats: whether running stats are updated or not
                                     (optional).

        Returns:
            intensity_integral: [B,T,M] The cumulative intensities for each
                query time for each mark (class).
            intensities_mask: [B,T]   Which intensities are valid for further
                computation based on e.g. sufficient history available.
            artifacts: Some measures
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
            artifacts: Optional[bool] = None
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
            artifacts: Some measures

        """
        # Add grads for query to compute derivative
        query.requires_grad = True

        intensity_integrals_q, intensity_mask_q, artifacts = \
            self.cum_intensity(
                events=events,
                query=query,
                prev_times=prev_times,
                prev_times_idxs=prev_times_idxs,
                pos_delta_mask=pos_delta_mask,
                is_event=is_event,
                representations=representations,
                representations_mask=representations_mask,
                artifacts=artifacts,
                update_running_stats=False)

        # Remove masked values and add epsilon for stability
        intensity_integrals_q = \
            intensity_integrals_q * intensity_mask_q.unsqueeze(-1)

        # Optional zero substraction
        if self.do_zero_subtraction:
            (intensity_integrals_z, intensity_mask_z,
             artifacts_zero) = self.cum_intensity(
                events=events,
                query=prev_times,
                prev_times=prev_times,
                prev_times_idxs=prev_times_idxs,
                pos_delta_mask=pos_delta_mask,
                is_event=is_event,
                representations=representations,
                representations_mask=representations_mask,
                artifacts=artifacts)

            intensity_integrals_z = \
                intensity_integrals_z * intensity_mask_z.unsqueeze(-1)
            intensity_integrals_q = th.clamp(
                intensity_integrals_q - intensity_integrals_z, min=0.
            ) + intensity_integrals_z
            intensity_integrals_q = intensity_integrals_q + epsilon(
                eps=1e-3,
                dtype=intensity_integrals_q.dtype,
                device=intensity_integrals_q.device) * query.unsqueeze(-1)
            if self.model_log_cm:
                intensity_integrals = subtract_exp(
                    intensity_integrals_q, intensity_integrals_z)
            else:
                intensity_integrals = \
                    intensity_integrals_q - intensity_integrals_z
            intensity_mask = intensity_mask_q * intensity_mask_z

        else:
            intensity_integrals_q = intensity_integrals_q + epsilon(
                eps=1e-3,
                dtype=intensity_integrals_q.dtype,
                device=intensity_integrals_q.device) * query.unsqueeze(-1)
            intensity_mask = intensity_mask_q
            if self.model_log_cm:
                intensity_integrals = th.exp(intensity_integrals_q)
            else:
                intensity_integrals = intensity_integrals_q

        check_tensor(intensity_integrals * intensity_mask.unsqueeze(-1),
                     positive=True)

        # Compute derivative of the integral
        grad_outputs = th.zeros_like(intensity_integrals_q, requires_grad=True)
        grad_inputs = th.autograd.grad(
            outputs=intensity_integrals_q,
            inputs=query,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True)[0]
        marked_intensity = th.autograd.grad(
            outputs=grad_inputs,
            inputs=grad_outputs,
            grad_outputs=th.ones_like(grad_inputs),
            retain_graph=True,
            create_graph=True)[0]
        query.requires_grad = False

        check_tensor(marked_intensity, positive=True, strict=True)
        log = Log.apply
        if self.model_log_cm:
            marked_log_intensity = \
                log(marked_intensity) + intensity_integrals_q
        else:
            marked_log_intensity = log(marked_intensity)

        artifacts_decoder = {
            "intensity_integrals": intensity_integrals,
            "marked_intensity": marked_intensity,
            "marked_log_intensity": marked_log_intensity,
            "intensity_mask": intensity_mask}
        if artifacts is None:
            artifacts = {'decoder': artifacts_decoder}
        else:
            if 'decoder' in artifacts:
                if 'attention_weights' in artifacts['decoder']:
                    artifacts_decoder['attention_weights'] = \
                        artifacts['decoder']['attention_weights']
            artifacts['decoder'] = artifacts_decoder

        return (marked_log_intensity,
                intensity_integrals,
                intensity_mask,
                artifacts)  # [B,T,M], [B,T,M], [B,T], Dict
