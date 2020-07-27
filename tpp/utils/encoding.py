import torch as th
import torch.nn as nn
import math
import numpy as np

from typing import Optional, Callable


class SinusoidalEncoding(nn.Module):
    def __init__(self, emb_dim, scaling):
        super(SinusoidalEncoding, self).__init__()
        self.emb_dim = emb_dim
        self.scaling = scaling

    def forward(
            self,
            times,
            min_timescale: float = 1.0,
            max_timescale: float = 1e4
    ):
        """
        Adaptation of positional encoding to include temporal information
        """

        assert self.emb_dim % 2 == 0, "hidden size must be a multiple of 2 " \
                                      "with pos_enc, pos_dec"
        num_timescales = self.emb_dim // 2
        log_timescale_increment = np.log(max_timescale / min_timescale
                                         ) / (num_timescales - 1)
        inv_timescales = (
                min_timescale * th.exp(
                    th.arange(
                        num_timescales, dtype=th.float, device=times.device
                    ) * -log_timescale_increment))
        scaled_time = times.type(
            th.FloatTensor).to(times.device) * inv_timescales.unsqueeze(
            0).unsqueeze(0) * self.scaling
        signal = th.cat([th.sin(scaled_time), th.cos(scaled_time)], dim=2)
        return signal


def encoding_size(encoding, emb_dim):
    if encoding == "times_only":
        return 1
    elif encoding in ["marks_only", "temporal", "learnable",
                      "temporal_with_labels", "learnable_with_labels"]:
        return emb_dim
    elif encoding == "concatenate":
        return emb_dim + 1
    raise ValueError("Time encoding not understood")


def event_encoder(
        times: th.Tensor,
        mask: th.Tensor,
        encoding: str,
        labels: Optional[th.Tensor] = None,
        embedding_layer: Optional[Callable[[th.Tensor], th.Tensor]] = None,
        temporal_enc: Optional[Callable[[th.Tensor], th.Tensor]] = None):
    """Representation encoder. Switch to determine inputs.

    Args:
        times: [B,L+1 or T] Delta times of events
        labels: [B,L+1,D] Representations of the events
        mask: [B,L+1 or T] Mask of event representations
        temporal_enc: Positional encoding function
        encoding: Switch. Either times_only, marks_only, concatenate,
                  temporal, learnable, temporal_with_labels or
                  learnable_with_labels"
        embedding_layer: Optional, embedding layer to encoder labels

    Returns:
        Encoded representations [B,L+1 or T,1 or D or D+1]
        Mask: [B,L+1 or T]

    """
    # Returns only times
    if encoding == "times_only":
        return times.unsqueeze(-1), mask  # [B,L+1 or T,1]

    if encoding == "temporal" or encoding == "learnable":
        embeddings = temporal_enc(times.unsqueeze(-1))  # [B,L+1 or T,D]
        return embeddings, mask

    # Returns only representation
    embeddings = embedding_layer(labels)
    if encoding == "marks_only":
        return embeddings, mask  # [B,L+1,D], [B,L+1]

    # Returns times concatenated to representations
    if encoding == "concatenate":
        merged_representations = th.cat(
            (embeddings, times.unsqueeze(-1)),
            dim=-1)  # [B,L+1,D+1], [B,L+1]
        return merged_representations, mask

    if encoding in ["temporal_with_labels", "learnable_with_labels"]:
        representations = embeddings * math.sqrt(
            embeddings.shape[-1])
        merged_embeddings = representations + temporal_enc(
            times.unsqueeze(-1))  # [B,L+1,D]
        return merged_embeddings, mask  # [B,L+1,D], [B,L+1]

