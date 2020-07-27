import torch as th
import torch.nn.functional as F

from typing import List, Optional, Tuple, Dict

from tpp.models.encoders.base.variable_history import VariableHistoryEncoder

from tpp.pytorch.models import MLP

from tpp.utils.events import Events
from tpp.utils.transformer_utils import TransformerEncoderNetwork
from tpp.utils.transformer_utils import TransformerEncoderLayer
from tpp.utils.transformer_utils import generate_square_subsequent_mask


class SelfAttentionEncoder(VariableHistoryEncoder):
    """Self-attention network, based on a variable history encoder.

    Args:
        units_mlp: List of hidden layers sizes, including the output size.
        activation_mlp: Activation functions. Either a list or a string.
        constraint_mlp: Constraint of the network. Either none, nonneg or
            softplus.
        dropout_mlp: Dropout rates, either a list or a float.
        activation_final_mlp: Last activation of the MLP.

        units_rnn: Hidden size of the Transformer.
        layers_rnn: Number of layers in the Transformer.
        n_heads: Number of heads in the Transformer.
        activation_rnn: The non-linearity to use for the Transformer.
        dropout_rnn: Rate of dropout in the Transformer.

        allow_window_attention: If True, attention allows attendence to the
            window. False otherwise. Defaults to False,
        emb_dim: Size of the embeddings (default=2).
        embedding_constraint: Constraint on the weights. Either `None`,
            'nonneg' or 'softplus'. Defaults to `None`.
        temporal_scaling: Scaling parameter for temporal encoding
        encoding: Way to encode the events: either times_only, marks_only,
                  concatenate or temporal_encoding. Defaults to times_only
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
            activation_final_mlp: Optional[str] = None,
            # Transformer
            units_rnn: Optional[int] = 16,
            layers_rnn: Optional[int] = 1,
            n_heads: Optional[int] = 1,
            activation_rnn: Optional[str] = "relu",
            dropout_rnn: Optional[float] = 0.,
            attn_activation: Optional[str] = "softmax",
            # Other
            allow_window_attention: Optional[bool] = False,
            emb_dim: Optional[int] = 2,
            embedding_constraint: Optional[str] = None,
            temporal_scaling: Optional[float] = 1.,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            **kwargs):
        super(SelfAttentionEncoder, self).__init__(
            name="selfattention",
            output_size=units_mlp[-1],
            emb_dim=emb_dim,
            embedding_constraint=embedding_constraint,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        self.src_mask = None
        self.allow_window_attention = allow_window_attention
        encoder_layer = TransformerEncoderLayer(
            d_model=self.encoding_size,
            nhead=n_heads,
            dim_feedforward=units_rnn,
            dropout=dropout_rnn,
            activation=activation_rnn,
            attn_activation=attn_activation)
        self.transformer_encoder = TransformerEncoderNetwork(
            encoder_layer=encoder_layer,
            num_layers=layers_rnn)
        self.mlp = MLP(
            units=units_mlp,
            activations=activation_mlp,
            constraint=constraint_mlp,
            dropout_rates=dropout_mlp,
            input_shape=self.encoding_size,
            activation_final=activation_final_mlp)

    def forward(self, events: Events) -> Tuple[th.Tensor, th.Tensor, Dict]:
        """Compute the (query time independent) event representations.

        Args:
            events: [B,L] Times and labels of events.

        Returns:
            representations: [B,L+1,M+1] Representations of each event.
            representations_mask: [B,L+1] Mask indicating which representations
                are well-defined.

        """
        histories, histories_mask = self.get_events_representations(
            events=events)  # [B,L+1,D] [B,L+1]

        # Compute src_mask
        if (self.src_mask is None
                or self.src_mask.size(0) != histories.size()[1]):
            src_mask = generate_square_subsequent_mask(
                sz=histories.size()[1],
                device=histories.device)
            self.src_mask = src_mask

        if not self.allow_window_attention:
            self.src_mask[1:, 0] = float('-inf')

        # [B,L,D] -> [L,B,D]
        histories = histories.transpose(0, 1)
        hidden, attn_weights = self.transformer_encoder(
            src=histories,
            mask=self.src_mask
        )  # [L,B,hidden_size], [B,L,L]

        # [L,B,hidden_size] -> [B,L,hidden_size]
        hidden = hidden.transpose(0, 1)

        hidden = F.normalize(hidden, dim=-1, p=2)
        outputs = self.mlp(hidden)  # [B,L,output_size]

        artifacts = {'encoder': {"attention_weights": attn_weights}}

        return outputs, histories_mask, artifacts
