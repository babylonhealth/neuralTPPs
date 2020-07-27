import torch as th
import torch.nn as nn


class Arctan(nn.Module):
    """Arctan activation function
    """
    def __init__(self):
        super(Arctan, self).__init__()

    def forward(self, x):
        return th.atan(x)
