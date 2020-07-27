import torch.nn as nn

from collections.abc import Iterable
from typing import List, Optional

from tpp.pytorch.activations import ParametricSoftplus, AdaptiveGumbel
from tpp.pytorch.activations import AdaptiveGumbelSoftplus
from tpp.pytorch.layers import LAYER_CLASSES
from tpp.pytorch.activations import ACTIVATIONS


class MLP(nn.Module):
    """Neural network which can put constraints on its weights

    Args:

        units: list of sizes of linear layers
        activations: activation functions. Either a string or a list of strings
        constraint: constraint on the weights. Either none, nonneg or softplus
        dropout_rates: Either a float or a list of floats
        input_shape: shape of the input
        activation_final: final activation function
        activation_final: final activation function
        use_bias: True if we want to use bias, False otherwise.

    """
    def __init__(
            self,
            units: List[int],
            input_shape: int,
            activations: List[str],
            constraint: str,
            dropout_rates: Optional[float] = None,
            activation_final: Optional[float] = None,
            use_bias=True,
            **kwargs):
        super(MLP, self).__init__()
        self.units = units
        self.activations = activations
        self.constraint = constraint
        self.dropout_rates = dropout_rates
        self.input_shape = input_shape
        self.activation_final = activation_final
        self.net = nn.Sequential()
        self.layers = list()
        self.n_layers = len(units)
        self.use_bias = use_bias

        if not isinstance(self.dropout_rates, Iterable):
            self.dropout_rates = [self.dropout_rates] * len(self.units)

        if (not isinstance(self.activations, Iterable) or
                isinstance(self.activations, str)):
            self.activations = [self.activations] * len(self.units)

        self.layer_class = nn.Linear
        if self.constraint is not None:
            self.layer_class = LAYER_CLASSES[self.constraint]

        self.units = [self.input_shape] + self.units

        for i in range(len(self.units) - 1):
            final_layer = i == self.n_layers - 1
            in_features, out_features = self.units[i], self.units[i + 1]

            layer = self.layer_class(
                in_features=in_features,
                out_features=out_features,
                bias=use_bias)
            if self.constraint == "nonneg":
                layer = self.layer_class(
                    in_features=in_features,
                    out_features=out_features,
                    bias=use_bias,
                    eps=1e-30)

            self.layers.append(("linear{}".format(i), layer))

            activation = self.activations[i]
            if final_layer:
                activation = self.activation_final

            if activation is not None:
                if activation == "parametric_softplus":
                    activation_fn = ParametricSoftplus(units=out_features)
                elif activation == "gumbel":
                    activation_fn = AdaptiveGumbel(units=out_features)
                elif activation == "gumbel_softplus":
                    activation_fn = AdaptiveGumbelSoftplus(units=out_features)
                else:
                    activation_fn = ACTIVATIONS[activation]

                self.layers.append(("activation{}".format(i), activation_fn))

            dropout_rate = self.dropout_rates[i]
            if dropout_rate is not None and dropout_rate > 0.0:
                dropout_fn = nn.Dropout(p=dropout_rate)
                self.layers.append(("dropout{}".format(i), dropout_fn))

        for n, l in self.layers:
            self.net.add_module(n, l)

    def forward(self, inputs):
        output = self.net(inputs)
        return output
