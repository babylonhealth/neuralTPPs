import torch as th

from typing import Dict, List, Optional, Tuple

from tpp.models.encoders.base.variable_history import VariableHistoryEncoder
from tpp.pytorch.models import MLP
from tpp.utils.events import Events


class MLPVariableEncoder(VariableHistoryEncoder):
    """Variable MLP encoder, i.e. r(t) = MLP(rep(l, t))

    Args:
        units_mlp: List of hidden layers sizes for MLP.
        activations: MLP activation functions. Either a list or a string.
        emb_dim: Size of the embeddings. Defaults to 1.
        embedding_constraint: Constraint on the weights. Either `None`,
            'nonneg' or 'softplus'. Defaults to `None`.
        temporal_scaling: Scaling parameter for temporal encoding
        padding_id: Id of the padding. Defaults to -1.
        encoding: Way to encode the events: either times_only, marks_only,
                  concatenate or temporal_encoding. Defaults to times_only
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
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
        super(MLPVariableEncoder, self).__init__(
            name="mlp-variable",
            output_size=units_mlp[-1],
            emb_dim=emb_dim,
            embedding_constraint=embedding_constraint,
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
            events=events)                                # [B,L+1,enc] [B,L+1]
        representations = self.mlp(histories)             # [B,L+1,D]
        return (representations, histories_mask,
                dict())  # [B,L+1,D], [B,L+1], Dict
