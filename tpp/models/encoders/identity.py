import torch as th

from typing import Dict, Optional, Tuple

from tpp.models.encoders.base.variable_history import VariableHistoryEncoder
from tpp.utils.encoding import encoding_size
from tpp.utils.events import Events


class IdentityEncoder(VariableHistoryEncoder):
    """Variable encoder that passes the representations straight to the
        decoder, i.e. r(t) = rep(l, t).

    Args:
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
            # Other args
            emb_dim: Optional[int] = 1,
            embedding_constraint: Optional[str] = None,
            temporal_scaling: Optional[float] = 1.,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            **kwargs):
        super(IdentityEncoder, self).__init__(
            name="identity",
            output_size=encoding_size(encoding=encoding, emb_dim=emb_dim),
            emb_dim=emb_dim,
            embedding_constraint=embedding_constraint,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)

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
        return (histories, histories_mask,
                dict())  # [B,L+1,D], [B,L+1], Dict
