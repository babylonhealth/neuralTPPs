import abc

import torch as th
import torch.nn as nn

from typing import Optional, Tuple

from tpp.utils.events import Events

from tpp.models.encoders.base.encoder import Encoder

from tpp.pytorch.models import LAYER_CLASSES, MLP

from tpp.utils.history import get_prev_times
from tpp.utils.encoding import SinusoidalEncoding, event_encoder, encoding_size


class VariableHistoryEncoder(Encoder, abc.ABC):
    """Variable history encoder. Here, the size H depends on the encoding type.
       It can be either 1, emb_dim or emb_dim+1.

    Args:
        name: The name of the encoder class.
        output_size: The output size (dimensionality) of the representations
            formed by the encoder.
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
            name: str,
            output_size: int,
            emb_dim: Optional[int] = 1,
            embedding_constraint: Optional[str] = None,
            temporal_scaling: Optional[float] = 1.,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            **kwargs):
        super(VariableHistoryEncoder, self).__init__(
            name=name, output_size=output_size, marks=marks, **kwargs)
        self.emb_dim = emb_dim
        self.encoding = encoding
        self.time_encoding = time_encoding
        self.embedding_constraint = embedding_constraint
        self.encoding_size = encoding_size(
            encoding=self.encoding, emb_dim=self.emb_dim)

        self.embedding = None
        if encoding in ["marks_only", "concatenate", "temporal_with_labels",
                        "learnable_with_labels"]:
            embedding_layer_class = nn.Linear
            if self.embedding_constraint is not None:
                embedding_layer_class = LAYER_CLASSES[
                    self.embedding_constraint]
            self.embedding = embedding_layer_class(
                in_features=self.marks, out_features=self.emb_dim, bias=False)

        self.temporal_enc = None
        if encoding in ["temporal", "temporal_with_labels"]:
            self.temporal_enc = SinusoidalEncoding(
                emb_dim=self.emb_dim, scaling=temporal_scaling)
        elif encoding in ["learnable", "learnable_with_labels"]:
            self.temporal_enc = MLP(
                units=[self.emb_dim],
                activations=None,
                constraint=self.embedding_constraint,
                dropout_rates=0,
                input_shape=1,
                activation_final=None)

    def get_events_representations(
            self, events: Events) -> Tuple[th.Tensor, th.Tensor]:
        """Compute the history vectors.

        Args:
            events: [B,L] Times and labels of events.

        Returns:
            merged_embeddings: [B,L+1,emb_dim] Histories of each event.
            histories_mask: [B,L+1] Mask indicating which histories
                are well-defined.
        """
        times = events.get_times(prepend_window=True)      # [B,L+1]
        histories_mask = events.get_mask(prepend_window=True)  # [B,L+1]

        # Creates a delta_t tensor, with first time set to zero
        # Masks it and sets masked values to padding id
        prev_times, is_event, pos_delta_mask = get_prev_times(
            query=times,
            events=events,
            allow_window=True)            # ([B,L+1],[B,L+1]), [B,L+1], [B,L+1]

        if self.time_encoding == "relative":
            prev_times, prev_times_idxs = prev_times  # [B,L+1], [B,L+1]
            times = times - prev_times

        histories_mask = histories_mask * pos_delta_mask

        if self.encoding != "marks_only" and self.time_encoding == "relative":
            histories_mask = histories_mask * is_event

        labels = events.labels
        labels = th.cat(
            (th.zeros(
                size=(labels.shape[0], 1, labels.shape[-1]),
                dtype=labels.dtype, device=labels.device),
             labels), dim=1)  # [B,L+1,M]

        return event_encoder(
            times=times,
            mask=histories_mask,
            encoding=self.encoding,
            labels=labels,
            embedding_layer=self.embedding,
            temporal_enc=self.temporal_enc)
