from torch import nn

from tpp.pytorch.activations.arctan import Arctan
from tpp.pytorch.activations.gumbel import AdaptiveGumbel
from tpp.pytorch.activations.gumbel import AdaptiveGumbelSoftplus
from tpp.pytorch.activations.pau import PAU
from tpp.pytorch.activations.softplus import MonotonicSoftplus
from tpp.pytorch.activations.softplus import ParametricSoftplus


ACTIVATIONS = {
    "arctan": Arctan(),
    'celu': nn.CELU(),
    'elu': nn.ELU(),
    'leaky_relu': nn.LeakyReLU(),
    'log_softmax': nn.LogSoftmax(dim=-1),
    'monotonic_softplus': MonotonicSoftplus(),
    'pau': PAU(init_coefficients="pade_sigmoid_3"),
    'relu': nn.ReLU(),
    'relu6': nn.ReLU6(),
    'sigmoid': nn.Sigmoid(),
    'softmax': nn.Softmax(dim=-1),
    'softplus': nn.Softplus(),
    'softsign': nn.Softsign(),
    'tanh': nn.Tanh(),
    'tanhshrink': nn.Tanhshrink()}
