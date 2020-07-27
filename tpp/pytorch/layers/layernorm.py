import torch as th

from torch import nn
from typing import Optional

from tpp.utils.nnplus import non_neg_param


class LayerNorm(nn.LayerNorm):
    def __init__(
            self,
            normalized_shape,
            eps=1e-5,
            elementwise_affine=True,
            momentum=.1,
            use_running_stats=False):
        super(LayerNorm, self).__init__(
            normalized_shape, eps, elementwise_affine)
        assert isinstance(normalized_shape, int), (
            "Only implemented this for final layer normalisation")

        self.use_running_stats = use_running_stats
        self.momentum = momentum

        if self.use_running_stats:
            self.register_buffer('running_mean', th.zeros(1))
            self.register_buffer('running_var', th.ones(1))
            self.register_buffer('num_batches_tracked',
                                 th.tensor(0, dtype=th.long))
            self.reset_running_stats()

        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

    def reset_running_stats(self):
        if self.use_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def forward(
            self,
            inputs: th.Tensor,
            update_running_stats: Optional[bool] = True) -> th.Tensor:
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.use_running_stats:
            running_mean, running_var = self.running_mean, self.running_var

        mu = th.mean(inputs, dim=-1, keepdim=True)
        var = th.var(inputs, dim=-1, keepdim=True, unbiased=False)

        if self.training and self.use_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

            n = th.Tensor(list(inputs.shape[:-1]))
            n = th.prod(n)

            if update_running_stats:
                with th.no_grad():
                    self.running_mean = (
                            exponential_average_factor * th.mean(mu)
                            + (1 - exponential_average_factor
                               ) * self.running_mean)
                    # update running_var with unbiased var
                    self.running_var = (
                            exponential_average_factor * th.mean(var) * n / (n - 1)
                            + (1 - exponential_average_factor
                               ) * self.running_var)

        if self.use_running_stats:
            mu, var = running_mean, running_var

        outputs = (inputs - mu) / th.sqrt(var + self.eps)
        if self.use_running_stats:
            self.weight.data = non_neg_param(self.weight)
        outputs = self.weight * outputs
        outputs = outputs + self.bias
        return outputs
