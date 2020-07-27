from torch import nn
from typing import Optional, List

from tpp.models.encoders.base.recurrent import RecurrentEncoder
from tpp.utils.encoding import encoding_size


class GRUEncoder(RecurrentEncoder):
    """GRU network, based on a variable recurrent encoder.

    Args:
        units_rnn: Hidden size of the GRU.
        layers_rnn: Number of layers in the GRU.
        units_mlp: List of hidden layers sizes for MLP.
        activations: MLP activation functions. Either a list or a string.
        dropout: Dropout rates (shared by MLP and GRU).
        activation_final_mlp: Activation of final layer of MLP.
        emb_dim: Size of the embeddings. Defaults to 1.
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
            # RNN args
            units_rnn: int,
            layers_rnn: int,
            dropout_rnn: float,
            # MLP args
            units_mlp: List[int],
            activation_mlp: Optional[str] = "relu",
            dropout_mlp: Optional[float] = 0.,
            constraint_mlp: Optional[str] = None,
            activation_final_mlp: Optional[str] = None,
            # Other args
            emb_dim: Optional[int] = 1,
            embedding_constraint: Optional[str] = None,
            temporal_scaling: Optional[float] = 1.,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            **kwargs):
        gru = nn.GRU(
            # This will become self.encoding_size later whoops.
            input_size=encoding_size(encoding=encoding, emb_dim=emb_dim),
            hidden_size=units_rnn,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_rnn,
            num_layers=layers_rnn)
        super(GRUEncoder, self).__init__(
            name="gru",
            rnn=gru,
            units_mlp=units_mlp,
            activation=activation_mlp,
            dropout_mlp=dropout_mlp,
            constraint=constraint_mlp,
            activation_final_mlp=activation_final_mlp,
            emb_dim=emb_dim,
            embedding_constraint=embedding_constraint,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
