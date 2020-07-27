import torch as th

from typing import List, Optional, Tuple, Dict

from tpp.models.decoders.base.cumulative import CumulativeDecoder
from tpp.models.base.process import Events

from tpp.pytorch.models import MLP

from tpp.utils.encoding import encoding_size
from tpp.utils.index import take_3_by_2


class MLPCmDecoder(CumulativeDecoder):
    """A mlp decoder based on the cumulative approach.

    Args:
        units_mlp: List of hidden layers sizes, including the output size.
        activation_mlp: Activation functions. Either a list or a string.
        constraint_mlp: Constraint of the network. Either none, nonneg or
            softplus.
        dropout_mlp: Dropout rates, either a list or a float.
        activation_final_mlp: Last activation of the MLP.

        mc_prop_est: Proportion of numbers of samples for the MC method,
                     compared to the size of the input. (Default=1.).
        do_zero_subtraction: If `True` the class computes
            Lambda(tau) = Lambda'(tau) - Lambda'(0)
            in order to enforce Lambda(0) = 0. Defaults to `True`.
        emb_dim: Size of the embeddings (default=2).
        encoding: Way to encode the events: either times_only, or temporal.
            Defaults to times_only.
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            # MLP
            units_mlp: List[int],
            activation_mlp: Optional[str] = "relu",
            dropout_mlp: Optional[float] = 0.,
            constraint_mlp: Optional[str] = "nonneg",
            activation_final_mlp: Optional[str] = "parametric_softplus",
            # Other params
            model_log_cm: Optional[bool] = False,
            do_zero_subtraction: Optional[bool] = True,
            emb_dim: Optional[int] = 2,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            **kwargs):

        if constraint_mlp is None:
            print("Warning! MLP decoder is unconstrained. Setting to `nonneg`")
            constraint_mlp = "nonneg"

        enc_size = encoding_size(encoding=encoding, emb_dim=emb_dim)
        input_size = units_mlp[0] - enc_size
        super(MLPCmDecoder, self).__init__(
            name="mlp-cm",
            do_zero_subtraction=do_zero_subtraction,
            model_log_cm=model_log_cm,
            input_size=input_size,
            emb_dim=emb_dim,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.mlp = MLP(
            units=units_mlp[1:],
            activations=activation_mlp,
            constraint=constraint_mlp,
            dropout_rates=dropout_mlp,
            # units_mlp in this class also provides the input dimensionality
            # of the mlp
            input_shape=units_mlp[0],
            activation_final=activation_final_mlp)

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
        (query_representations,
         intensity_mask) = self.get_query_representations(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs,
            pos_delta_mask=pos_delta_mask,
            is_event=is_event,
            representations=representations,
            representations_mask=representations_mask)  # [B,T,enc_size], [B,T]

        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)                   # [B,T,D]

        hidden = th.cat(
            [query_representations, history_representations],
            dim=-1)                                        # [B,T,units_mlp[0]]

        intensity_itg = self.mlp(hidden)                    # [B,T,output_size]

        return intensity_itg, intensity_mask, artifacts
