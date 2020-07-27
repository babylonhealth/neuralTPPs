import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from tpp.pytorch.activations import ACTIVATIONS
from tpp.pytorch.activations import AdaptiveGumbel
from tpp.pytorch.activations import AdaptiveGumbelSoftplus
from tpp.pytorch.layers import LAYER_CLASSES
from tpp.pytorch.layers import BatchNorm1d, LayerNorm

from tpp.utils.multi_head_attention import MultiheadAttention


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def generate_square_subsequent_mask(sz, device):
    """Generate a square mask for the sequence. The masked positions are
    filled with float('-inf'). Unmasked positions are filled with
    float(0.0).
    """
    mask = (th.triu(th.ones(sz, sz)) == 1).transpose(0, 1).to(device)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    return mask


class TransformerEncoderNetwork(nn.Module):
    """TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class
            (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoderNetwork, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layers in turn.
        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper
    "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar,
    Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017.
    Attention is all you need. In Advances in
    Neural Information Processing Systems,
    pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the
                         feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer,
                    relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation="relu",
            attn_activation="softmax"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = ACTIVATIONS[activation]
        self.attn_activation = attn_activation

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the
                                  src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        if type(src) == tuple:
            src = src[0]
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            need_weights=True,
            key_padding_mask=src_key_padding_mask,
            activation=self.attn_activation)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(
                self.dropout(
                    self.activation(
                        self.linear1(src)
                    )
                )
            )
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


class TransformerDecoderNetwork(nn.Module):
    """TransformerDecoder is a stack of N decoder layers
    Args:
        decoder_layer: an instance of the TransformerDecoderLayer()
        num_layers: the number of sub-decoder-layers in the decoder.
        norm: the layer normalization component (optional).
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoderNetwork, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            update_running_stats=True):
        """Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch
                                  (optional).
            memory_key_padding_mask: the mask for the memory keys per batch
                                     (optional).
            update_running_stats: whether running stats are updated or not
                                     (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                update_running_stats=update_running_stats)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and
       feedforward network. This standard decoder layer is based on the paper
       "Attention Is All You Need". Ashish Vaswani, Noam Shazeer, Niki Parmar,
       Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
       Illia Polosukhin. 2017. Attention is all you need. In Advances in
       Neural Information Processing Systems, pages 6000-6010. Users may modify
        or implement in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model
                         (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu
                    (default=relu).
         attn_activation: 'identity', 'softmax' or 'sigmoid' to use for the
                          attention coefficients.
        cumulative: Whether to use the cumulative form of the
            attention mechanism. If this is `True`, the behaviour follows
            <link to Neural TPPs paper>.
    """

    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            constraint=None,
            activation="relu",
            attn_activation="softmax",
            normalisation=None):
        super(TransformerDecoderLayer, self).__init__()
        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            constraint=constraint)
        self.constraint = constraint
        self.normalisation = normalisation

        # Implementation of Feedforward model
        self.layer_class = nn.Linear
        if self.constraint is not None:
            self.layer_class = LAYER_CLASSES[self.constraint]

        self.linear1 = self.layer_class(
            d_model, dim_feedforward, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = self.layer_class(
            dim_feedforward, d_model, bias=True)

        if self.normalisation == "layernorm_with_running_stats":
            self.norm2 = LayerNorm(d_model, use_running_stats=True)
            self.norm3 = LayerNorm(d_model, use_running_stats=True)
        elif self.normalisation == "layernorm":
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        elif self.normalisation == "batchnorm":
            self.norm2 = BatchNorm1d(d_model)
            self.norm3 = BatchNorm1d(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation == "gumbel":
            self.activation = AdaptiveGumbel(units=dim_feedforward)
        elif activation == "gumbel_softplus":
            self.activation = AdaptiveGumbelSoftplus(units=dim_feedforward)
        else:
            self.activation = ACTIVATIONS[activation]
        self.attn_activation = attn_activation

    def forward(
            self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
            update_running_stats=True):
        """Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch
                                  (optional).
            memory_key_padding_mask: the mask for the memory keys per batch
                                     (optional).
            update_running_stats: whether running stats are updated or not
                                     (optional).
        Shape:
            see the docs in Transformer class.
        """
        if type(tgt) == tuple:
            tgt = tgt[0]
        tgt2 = tgt
        if self.normalisation is not None:
            if self.normalisation == "layernorm_with_running_stats":
                tgt2 = self.norm2(
                    tgt2, update_running_stats=update_running_stats)
            else:
                tgt2 = self.norm2(tgt2)
        tgt2, attn_weights = self.multihead_attn(
            tgt2, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            activation=self.attn_activation)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = tgt
        if self.normalisation is not None:
            if self.normalisation == "layernorm_with_running_stats":
                tgt2 = self.norm3(
                    tgt2, update_running_stats=update_running_stats)
            else:
                tgt2 = self.norm2(tgt2)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(
                self.dropout(self.activation(self.linear1(tgt2))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, attn_weights
