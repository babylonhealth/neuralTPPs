import torch as th
from torch import nn


class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True,
                 use_running_estimates=False,
                 normalise_over_final=False):
        super(BatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.use_running_estimates = use_running_estimates
        self.normalise_over_final = normalise_over_final

    def forward(self, inputs):
        self._check_input_dim(inputs)

        input_rank, exponential_average_factor = len(inputs.shape), 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.use_running_estimates:
            with th.no_grad():
                running_mean, running_var = self.running_mean, self.running_var

        # calculate running estimates
        if self.training:
            if input_rank == 2:
                # use biased var in train
                mean, var = inputs.mean([0]), inputs.var([0], unbiased=False)
            elif input_rank == 3:
                if self.normalise_over_final:
                    mean = inputs.mean([0, 1])
                    var = inputs.var([0, 1], unbiased=False)
                else:
                    mean = inputs.mean([0, 2])
                    var = inputs.var([0, 2], unbiased=False)
            else:
                raise ValueError("Incorrect input shape.")

            if self.normalise_over_final and input_rank == 3:
                n = inputs.numel() / inputs.size(2)
            else:
                n = inputs.numel() / inputs.size(1)

            with th.no_grad():
                self.running_mean = (
                        exponential_average_factor * mean
                        + (1 - exponential_average_factor) * self.running_mean)
                # update running_var with unbiased var
                self.running_var = (
                        exponential_average_factor * var * n / (n - 1)
                        + (1 - exponential_average_factor) * self.running_var)
        else:
            mean = self.running_mean
            var = self.running_var

        if self.use_running_estimates:
            mean, var = running_mean, running_var

        if input_rank == 2:
            mean = mean.unsqueeze(dim=0)
            var = var.unsqueeze(dim=0)
            weight = self.weight.unsqueeze(dim=0)
            bias = self.bias.unsqueeze(dim=0)
        elif input_rank == 3:
            if self.normalise_over_final:
                mean = mean.unsqueeze(dim=0).unsqueeze(dim=0)
                var = var.unsqueeze(dim=0).unsqueeze(dim=0)
                weight = self.weight.unsqueeze(dim=0).unsqueeze(dim=0)
                bias = self.bias.unsqueeze(dim=0).unsqueeze(dim=0)
            else:
                mean = mean.unsqueeze(dim=0).unsqueeze(dim=-1)
                var = var.unsqueeze(dim=0).unsqueeze(dim=-1)
                weight = self.weight.unsqueeze(dim=0).unsqueeze(dim=-1)
                bias = self.bias.unsqueeze(dim=0).unsqueeze(dim=-1)

        inputs = (inputs - mean) / th.sqrt(var + self.eps)
        if self.affine:
            inputs = inputs * weight + bias

        return inputs
