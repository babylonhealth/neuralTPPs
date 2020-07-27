import torch as th
from torch import nn

from tpp.pytorch.layers import LayerNorm
from debug.batchnorm import multidim_grad

th.manual_seed(0)

pytorch_norm = nn.LayerNorm(3)
my_norm = LayerNorm(3, use_running_stats=True)

x = th.rand([3]).reshape(1, -1).float().repeat(2, 1)
x.requires_grad = True

pytorch_y = pytorch_norm(x)
my_y = my_norm(x)

# assert th.allclose(pytorch_y, my_y, rtol=1.e-3)

pytorch_y
my_y
pytorch_y.var(-1)
my_y.var(-1)

multidim_grad(pytorch_y, x)
multidim_grad(my_y, x)

x = th.rand(10, 2, 4)

norm1 = nn.LayerNorm(4)

for i in range(1,10):
    for j in range(1, 10):
        for k in range(1, 10):
            norm = LayerNorm(k, use_running_stats=True)
            for _ in range(3):
                x = th.rand(i, j, k)
                y = norm(x)
                assert list(x.shape) == list(y.shape)
