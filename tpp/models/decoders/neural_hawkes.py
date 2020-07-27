"""
Based on the implementation of Xiao Liu on Jan. 31, 2019.
https://github.com/xiao03/nh
"""

from typing import List, Optional, Tuple, Dict

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from tpp.models.decoders.base.monte_carlo import MCDecoder
from tpp.models.base.process import Events

from tpp.pytorch.layers.log import Log
from tpp.pytorch.models import MLP

from tpp.utils.index import take_3_by_2
from tpp.utils.stability import check_tensor


class NeuralHawkesDecoder(MCDecoder):
    """Continuous time LSTM network with decay function.
    """
    def __init__(self,
                 # RNN args
                 units_rnn: int,
                 # MLP args
                 units_mlp: List[int],
                 activation_mlp: Optional[str] = "relu",
                 dropout_mlp: Optional[float] = 0.,
                 constraint_mlp: Optional[str] = None,
                 activation_final_mlp: Optional[str] = "parametric_softplus",
                 # Other params
                 mc_prop_est: Optional[float] = 1.,
                 input_size: Optional[int] = None,
                 emb_dim: Optional[int] = 1,
                 temporal_scaling: Optional[float] = 1.,
                 encoding: Optional[str] = "times_only",
                 time_encoding: Optional[str] = "relative",
                 marks: Optional[int] = 1,
                 **kwargs):
        super(NeuralHawkesDecoder, self).__init__(
            name="neural-hawkes",
            mc_prop_est=mc_prop_est,
            input_size=input_size,
            emb_dim=emb_dim,
            temporal_scaling=temporal_scaling,
            encoding=encoding,
            time_encoding=time_encoding,
            marks=marks,
            **kwargs)
        # Parameters
        self.weight_ih = nn.Parameter(th.Tensor(units_rnn, units_rnn * 7))
        self.weight_hh = nn.Parameter(th.Tensor(units_rnn, units_rnn * 7))
        self.bias = nn.Parameter(th.Tensor(units_rnn * 7))
        self.mlp = MLP(
            units=units_mlp,
            activations=activation_mlp,
            constraint=constraint_mlp,
            dropout_rates=dropout_mlp,
            input_shape=self.encoding_size,
            activation_final=activation_final_mlp)
        self.units_rnn = units_rnn
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight_ih, b=0.001)
        nn.init.uniform_(self.weight_hh, b=0.001)
        nn.init.uniform_(self.bias, b=0.001)

    def recurrence(self, emb_event_t, h_d_tm1, c_tm1, c_bar_tm1):
        gates = (emb_event_t @ self.weight_ih +
                 h_d_tm1 @ self.weight_hh + self.bias)
        # B * 2H
        (gate_i,
         gate_f,
         gate_z,
         gate_o,
         gate_i_bar,
         gate_f_bar,
         gate_delta) = th.chunk(gates, 7, -1)

        gate_i = th.sigmoid(gate_i)
        gate_f = th.sigmoid(gate_f)
        gate_z = th.tanh(gate_z)
        gate_o = th.sigmoid(gate_o)
        gate_i_bar = th.sigmoid(gate_i_bar)
        gate_f_bar = th.sigmoid(gate_f_bar)
        gate_delta = F.softplus(gate_delta)

        c_t = gate_f * c_tm1 + gate_i * gate_z
        c_bar_t = gate_f_bar * c_bar_tm1 + gate_i_bar * gate_z

        return c_t, c_bar_t, gate_o, gate_delta

    @staticmethod
    def decay(c_t, c_bar_t, o_t, delta_t, duration_t):
        c_d_t = c_bar_t + (c_t - c_bar_t) * \
                th.exp(-delta_t * duration_t.view(-1, 1))

        h_d_t = o_t * th.tanh(c_d_t)

        return c_d_t, h_d_t

    def log_intensity(
            self,
            events: Events,
            query: th.Tensor,
            prev_times: th.Tensor,
            prev_times_idxs: th.Tensor,
            pos_delta_mask: th.Tensor,
            is_event: th.Tensor,
            representations: th.Tensor,
            representations_mask: Optional[th.Tensor] = None,
            artifacts: Optional[dict] = None
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
        Returns:
            log_intensity: [B,T,M] The intensities for each query time for
                each mark (class).
            intensities_mask: [B,T]   Which intensities are valid for further
                computation based on e.g. sufficient history available.
        """
        batch_size, query_length = query.size()
        query_representations, intensity_mask = self.get_query_representations(
            events=events,
            query=query,
            prev_times=prev_times,
            prev_times_idxs=prev_times_idxs,
            pos_delta_mask=pos_delta_mask,
            is_event=is_event,
            representations=representations,
            representations_mask=representations_mask)  # [B,T,D], [B,T]

        history_representations = take_3_by_2(
            representations, index=prev_times_idxs)
        query = query * intensity_mask
        prev_times = prev_times * intensity_mask

        h_seq = th.zeros(
            query_length,
            batch_size,
            self.units_rnn,
            dtype=th.float,
            device=representations.device)
        h_d = th.zeros(
            batch_size,
            self.units_rnn,
            dtype=th.float,
            device=representations.device)
        c_d = th.zeros(
            batch_size,
            self.units_rnn,
            dtype=th.float,
            device=representations.device)
        c_bar = th.zeros(
            batch_size,
            self.units_rnn,
            dtype=th.float,
            device=representations.device)

        for t in range(query_length):
            c, new_c_bar, o_t, delta_t = self.recurrence(
                history_representations[:, t], h_d, c_d, c_bar)
            new_c_d, new_h_d = self.decay(
                c, new_c_bar, o_t, delta_t, query[:, t] - prev_times[:, t])
            mask = intensity_mask[:, t].unsqueeze(-1)
            h_d = new_h_d * mask + h_d * (1. - mask)
            c_d = new_c_d * mask + c_d * (1. - mask)
            c_bar = new_c_bar * mask + c_bar * (1. - mask)
            h_seq[t] = h_d

        hidden = h_seq.transpose(0, 1)
        hidden = F.normalize(hidden, dim=-1, p=2)

        outputs = self.mlp(hidden)  # [B,L,output_size]
        check_tensor(outputs, positive=True, strict=True)
        log = Log.apply
        outputs = log(outputs)

        return outputs, intensity_mask, artifacts
