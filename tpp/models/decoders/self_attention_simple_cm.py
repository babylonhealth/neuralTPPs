import torch as th

from typing import List, Optional, Tuple, Dict

from tpp.models.decoders.base.cumulative import CumulativeDecoder
from tpp.models.base.process import Events

from tpp.pytorch.models import MLP

from tpp.utils.encoding import encoding_size


class SelfAttentionCmDecoder(CumulativeDecoder):
    """A self attention decoder based on the new cumulative sigmoid attention.

    Args:
        units_mlp: List of hidden layers sizes, including the output size.
        activation_mlp: Activation functions. Either a list or a string.
        constraint_mlp: Constraint of the network. Either none, nonneg or
            softplus.
        dropout_mlp: Dropout rates, either a list or a float.
        activation_final_mlp: Last activation of the MLP.

        units_rnn: Hidden size of the Transformer.
        layers_rnn: Number of layers in the Transformer.
        num_heads: Number of heads in the Transformer.
        activation_rnn: The non-linearity to use for the Transformer.
        dropout_rnn: Rate of dropout in the Transformer.

        emb_dim: Size of the embeddings (default=2).
        temporal_scaling: Scaling parameter for temporal encoding
        encoding: Way to encode the events: either times_only, marks_only,
                  concatenate or temporal_encoding. Defaults to times_only.
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            # MLP
            units_mlp: List[int],
            activation_mlp: Optional[str] = "relu",
            dropout_mlp: Optional[float] = 0.,
            constraint_mlp: Optional[str] = None,
            activation_final_mlp: Optional[str] = "parametric_softplus",
            # Transformer
            units_rnn: Optional[int] = 16,
            layers_rnn: Optional[int] = 1,
            num_heads: Optional[int] = 1,
            activation_rnn: Optional[str] = "relu",
            dropout_rnn: Optional[float] = 0.,
            attn_activation: Optional[str] = "softmax",
            # Other params
            mc_prop_est: Optional[float] = 1.,
            emb_dim: Optional[int] = 4,
            temporal_scaling: Optional[float] = 1.,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            **kwargs):
        super(SelfAttentionCmDecoder, self).__init__(
            name="selfattention-cm",
            input_size=encoding_size(encoding=encoding, emb_dim=emb_dim),
            mc_prop_est=mc_prop_est,
            emb_dim=emb_dim,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.mlp = MLP(
            units=units_mlp,
            activations=activation_mlp,
            constraint=constraint_mlp,
            dropout_rates=dropout_mlp,
            input_shape=self.encoding_size,
            activation_final=activation_final_mlp)

        self.mlp1 = MLP(
            units=[1],
            activations=None,
            constraint="nonneg",
            dropout_rates=None,
            input_shape=1,
            activation_final=None,
            use_bias=False)

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
            update_running_stats: whether running stats are updated or not.

        Returns:
            log_intensity: [B,T,M] The intensities for each query time for
                each mark (class).
            intensities_mask: [B,T]   Which intensities are valid for further
                computation based on e.g. sufficient history available.

        """
        _, events_length, _ = representations.size()
        batch_size, query_length = query.size()
        event_times = events.get_times(prepend_window=True)

        query_representations, intensity_mask = self.get_query_representations(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs,
            pos_delta_mask=pos_delta_mask,
            is_event=is_event,
            representations=representations,
            representations_mask=representations_mask)  # [B,T,D], [B,T]

        memory_mask = th.arange(
            events_length, device=representations.device).repeat(
            batch_size, query_length).reshape(
            batch_size, query_length, events_length)

        items_to_zero = memory_mask <= prev_times_idxs.unsqueeze(-1)
        # items_to_zero[:, :, 0] = events.first_event_on_window
        # Make sure there is at least one zero in the row
        missing_zeros = items_to_zero.sum(-1) == 0
        items_to_zero = items_to_zero | missing_zeros.unsqueeze(dim=-1)

        items_to_neg_inf = ~items_to_zero

        memory_mask = memory_mask.float()
        memory_mask = memory_mask.masked_fill(items_to_zero, float(0.))
        memory_mask = memory_mask.masked_fill(items_to_neg_inf, float('-inf'))

        assert list(memory_mask.size()) == [
            batch_size, query_length, events_length]

        delta_times = query.unsqueeze(1) - event_times.unsqueeze(-1)
        delta_times = delta_times.unsqueeze(-1)                   # [B,T,L+1,D]
        delta_times = self.mlp1(delta_times)
        delta_times = delta_times.squeeze(-1)                       # [B,T,L+1]
        attention_coefficients = th.exp(-delta_times)               # [B,T,L+1]




        # [B,T,D] -> [T,B,D] and [B,L+1,D] -> [L+1,B,D]
        tgt = query_representations.transpose(0, 1)                 # [T,B,D]
        tgt = th.zeros_like(tgt)

        memory = representations.transpose(0, 1)                    # [L+1,B,D]
        hidden, attn_weights = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            memory_mask=memory_mask,
            tgt_times=tgt_times,
            memory_times=memory_times,
            prev_times=prev_times_x)               # [T,B,hidden_size], [B,T,L]

        # [T,B,hidden_size] -> [B,T,hidden_size]
        hidden = hidden.transpose(0, 1)

        outputs = self.mlp(hidden)  # [B,L,output_size]

        if artifacts is None:
            artifacts = {'decoder': {"attention_weights": attn_weights}}
        else:
            artifacts['decoder'] = {"attention_weights": attn_weights}

        return outputs, intensity_mask, artifacts
