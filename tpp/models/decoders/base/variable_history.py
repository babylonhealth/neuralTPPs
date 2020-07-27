import abc

import torch as th
import torch.nn as nn

from typing import Optional

from tpp.models.base.process import Events
from tpp.models.decoders.base.decoder import Decoder

from tpp.pytorch.models import MLP

from tpp.utils.encoding import SinusoidalEncoding
from tpp.utils.encoding import event_encoder
from tpp.utils.encoding import encoding_size
from tpp.utils.index import take_3_by_2, take_2_by_2


class VariableHistoryDecoder(Decoder, abc.ABC):
    """Variable history decoder. Here, the size H depends on the encoding type.
       It can be either 1, emb_dim or emb_dim+1.

    Args:
        name: The name of the decoder class.
        input_size: The dimensionality of the input required from the encoder.
            Defaults to `None`. This is mainly just for tracking/debugging
            ease.
        emb_dim: Size of the embeddings. Defaults to 1.
        temporal_scaling: Scaling parameter for temporal encoding
        encoding: Way to encode the queries: either times_only, marks_only,
                  concatenate or temporal. Defaults to times_only
        marks: The distinct number of marks (classes) for the process. Defaults
            to 1.
    """
    def __init__(
            self,
            name: str,
            input_size: Optional[int] = None,
            emb_dim: Optional[int] = 1,
            embedding_constraint: Optional[str] = None,
            temporal_scaling: Optional[float] = 1.,
            encoding: Optional[str] = "times_only",
            time_encoding: Optional[str] = "relative",
            marks: Optional[int] = 1,
            **kwargs):
        super(VariableHistoryDecoder, self).__init__(
            name=name,
            input_size=input_size,
            marks=marks,
            **kwargs)
        self.emb_dim = emb_dim
        self.encoding = encoding
        self.time_encoding = time_encoding
        self.embedding_constraint = embedding_constraint
        self.encoding_size = encoding_size(
            encoding=self.encoding, emb_dim=self.emb_dim)

        self.embedding = None
        if encoding in ["marks_only", "concatenate", "temporal_with_labels",
                        "learnable_with_labels"]:
            self.embedding = MLP(
                units=[self.emb_dim],
                activations=None,
                constraint=self.embedding_constraint,
                dropout_rates=0,
                input_shape=self.marks,
                activation_final=None,
                use_bias=False)

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

    def get_query_representations(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None):
        """Computes the query representations

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

        Returns:
            query_representations: [B,T,D] Representations of the queries
            intensities_mask: [B,T]   Which intensities are valid for further
                computation based on e.g. sufficient history available.
        """
        if self.time_encoding == "relative":
            query = query - prev_times

        labels = events.labels
        labels = th.cat(
            (th.zeros(
                size=(labels.shape[0], 1, labels.shape[-1]),
                dtype=labels.dtype, device=labels.device),
             labels), dim=1)

        query_representations, representations_mask = event_encoder(
            times=query,
            mask=representations_mask,
            encoding=self.encoding,
            labels=take_3_by_2(labels, index=prev_times_idxs),
            embedding_layer=self.embedding,
            temporal_enc=self.temporal_enc)  # [B,T,D], [B,T]

        intensity_mask = pos_delta_mask  # [B,T]
        if representations_mask is not None:
            history_representations_mask = take_2_by_2(
                representations_mask, index=prev_times_idxs)  # [B,T]
            intensity_mask = intensity_mask * history_representations_mask

        return query_representations, intensity_mask  # [B,T,D], [B,T]
