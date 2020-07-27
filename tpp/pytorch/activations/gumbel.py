import math

import torch as th
from torch import nn

from tpp.utils.nnplus import non_neg_param
from tpp.utils.stability import epsilon_like
from tpp.pytorch.activations.softplus import ParametricSoftplus


class AdaptiveGumbel(nn.Module):
    def __init__(self, units):
        super(AdaptiveGumbel, self).__init__()
        self.units = units
        self.alpha = nn.Parameter(th.Tensor(self.units))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.alpha, 1.)

    def forward(self, x):
        if self.units != 1:
            assert x.shape[-1] == self.units, (
                "Final dim of input shape must match num parameters.")

        target_shape = [1] * (len(x.shape) - 1) + [self.units]

        self.alpha.data = non_neg_param(self.alpha)
        alpha = self.alpha.reshape(target_shape)
        alpha = alpha + epsilon_like(alpha)

        x = th.clamp(x, max=math.log(1e30))

        x = 1. + alpha * th.exp(x)
        x = x ** (-1 / alpha)
        x = 1. - x
        return x


class AdaptiveGumbelSoftplus(nn.Module):
    def __init__(self, units):
        super(AdaptiveGumbelSoftplus, self).__init__()
        self.units = units
        self.gumbel = AdaptiveGumbel(units=self.units)
        self.softplus = ParametricSoftplus(units=self.units)

    def forward(self, x):
        gumbel = self.gumbel(x)
        softplus = self.softplus(x)
        return gumbel * (1 + softplus)


def test_gumbel(do_derivative=False):
    x = th.linspace(start=-10., end=100.).reshape(-1, 1)
    x.requires_grad = True
    gumbel = AdaptiveGumbel(units=1)
    gumbel.alpha.data = th.Tensor([1.2])
    y = gumbel(x)
    y_unsafe = gumbel.forward_unsafe(x)
    if do_derivative:
        y = th.autograd.grad(y, x, grad_outputs=th.ones_like(y))[0]
        # y_unsafe = th.autograd.grad(y_unsafe, x, grad_outputs=th.ones_like(y))[0]
    import matplotlib.pyplot as plt
    plt.plot(x.detach().numpy(), y.detach().numpy(), label="normal")
    # plt.plot(x.detach().numpy(), y_unsafe.detach().numpy(), label="unsafe")
    if do_derivative:
        plt.title("d/dx act(x)")
    else:
        plt.title("act(x)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_gumbel()
    test_gumbel(do_derivative=True)
