"""
Taken from
https://github.com/ml-research/pau/blob/master/pau/cuda/python_imp/Pade.py
"""

import torch as th

from tpp.pytorch.activations.pau_utils import PADEACTIVATION_Function_based


class PAU(PADEACTIVATION_Function_based):
    def __init__(
            self,
            init_coefficients="pade_optimized_leakyrelu",
            monotonic=False):
        super(PAU, self).__init__(init_coefficients=init_coefficients)
        self.monotonic = monotonic

        if self.monotonic:
            assert self.n_numerator > (self.n_denominator + 1)

    def forward(self, x):
        # if self.monotonic:
        #     weight_numerator_subset = self.weight_numerator[
        #                               1:self.n_denominator + 1]
        #     denom_greater = self.weight_denominator > weight_numerator_subset
        #
        #     a = 0

        out = self.activation_function(
            x,
            self.weight_numerator,
            self.weight_denominator)
        return out


def test_pau(do_derivative=False):
    x = th.linspace(start=-10., end=10.).reshape(-1, 1)
    x.requires_grad = True
    pau = PAU(init_coefficients="pade_sigmoid_3", monotonic=True)
    y = pau(x)
    if do_derivative:
        y = th.autograd.grad(y, x, grad_outputs=th.ones_like(y))[0]
    import matplotlib.pyplot as plt
    plt.plot(x.detach().numpy(), y.detach().numpy())
    if do_derivative:
        plt.title("d/dx act(x)")
    else:
        plt.title("act(x)")
    plt.show()


if __name__ == "__main__":
    test_pau()
    test_pau(do_derivative=True)
