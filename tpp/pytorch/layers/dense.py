import torch as th
import torch.nn.functional as F

from torch import nn


class NonNegLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, eps=0.):
        super(NonNegLinear, self).__init__(in_features, out_features, bias)
        self.eps = eps
        self.positivify_weights()

    def positivify_weights(self):
        mask = (self.weight < 0).float() * - 1
        mask = mask + (self.weight >= 0).float()
        self.weight.data = self.weight.data * mask

    def forward(self, inputs):
        weight = self.weight > 0
        weight = self.weight * weight.float()
        self.weight.data = th.clamp(weight, min=self.eps)
        return F.linear(inputs, self.weight, self.bias)


class SigmoidLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SigmoidLinear, self).__init__(in_features, out_features, bias)
        self.positivify_weights()

    def positivify_weights(self):
        mask = (self.weight < 0).float() * - 1
        mask = mask + (self.weight >= 0).float()
        self.weight.data = self.weight.data * mask

    def forward(self, inputs):
        weight = F.sigmoid(self.weight)
        return F.linear(inputs, weight, self.bias)


class SoftPlusLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 beta=1., threshold=20):
        super(SoftPlusLinear, self).__init__(in_features, out_features, bias)
        self.beta = beta
        self.threshold = threshold
        self.positivify_weights()

    def positivify_weights(self):
        mask = (self.weight < 0).float() * - 1
        mask = mask + (self.weight >= 0).float()
        self.weight.data = self.weight.data * mask

    def forward(self, inputs):
        weight = F.softplus(
            self.weight, beta=self.beta, threshold=self.threshold)
        return F.linear(inputs, weight, self.bias)


LAYER_CLASSES = {
    "nonneg": NonNegLinear,
    "sigmoid": SigmoidLinear,
    "softplus": SoftPlusLinear}
