import torch as th
from torch import nn

from tpp.utils.nnplus import non_neg_param
from tpp.utils.stability import epsilon_like


class MonotonicSoftplus(nn.Module):
    """A version of the softplus that stays monotonic
    """
    def __init__(self, beta=1, threshold=20):
        super(MonotonicSoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x):
        mask = x < self.threshold
        mask = mask.float().to(device=x.device)
        return(th.log(1. + th.exp(x * mask * self.beta)) / self.beta
               + (x + 1e-8) * (1. - mask))


class ParametricSoftplus(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))

    SoftPlus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.

    """
    def __init__(self, units, threshold=20):
        super(ParametricSoftplus, self).__init__()
        self.units = units
        self.beta = nn.Parameter(th.Tensor(self.units))
        self.threshold = threshold
        self.softplus = nn.Softplus(beta=1., threshold=self.threshold)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.beta)

    def forward_alt(self, inputs):
        # log(1 + exp (x*beta)) / beta
        beta = self.beta
        beta_x = beta * inputs
        zeros = th.zeros_like(inputs)
        exponents = th.stack([zeros, beta_x], dim=-1)
        log_1_exp_xbeta = th.logsumexp(exponents, dim=-1)
        return log_1_exp_xbeta / beta

    def forward(self, inputs):
        if self.units != 1:
            assert inputs.shape[-1] == self.units, (
                "Final dim of input shape must match num parameters.")

        target_shape = [1] * (len(inputs.shape) - 1) + [self.units]

        self.beta.data = non_neg_param(self.beta)
        beta = self.beta.reshape(target_shape)
        beta = beta + epsilon_like(beta)
        
        inputs = beta * inputs
        outputs = self.softplus(inputs)
        outputs = outputs / beta
        return outputs


def test_softplus():
    units = 1
    x = th.linspace(start=-10, end=10)
    parametric_softplus = ParametricSoftplus(units=units)
    y = parametric_softplus(x)
    y2 = parametric_softplus.forward_alt(x)
    import matplotlib.pyplot as plt
    plt.plot(x.detach().numpy(), y.detach().numpy(), label="normal")
    plt.plot(x.detach().numpy(), y.detach().numpy(), label="new")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_softplus()
