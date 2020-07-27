import torch as th
from torch import nn

from tpp.pytorch.layers import NonNegLinear
from tpp.pytorch.layers import BatchNorm1d


def multidim_grad(a, b):
    a_split = th.split(a, split_size_or_sections=1, dim=-1)
    grads = [th.autograd.grad(
        outputs=a_split[i],
        inputs=b,
        grad_outputs=th.ones_like(a_split[i]),
        retain_graph=True,
        create_graph=True)[0] for i in range(a.shape[-1])]
    grads = th.stack(grads, dim=-1)
    return grads


th.manual_seed(0)

dense1 = NonNegLinear(1, 3, bias=True)
batchnorm1 = nn.BatchNorm1d(3)
batchnorm2 = BatchNorm1d(3)

x = th.rand(10, 1)

y1 = batchnorm1(dense1(x))
y2 = batchnorm2(dense1(x))

assert th.allclose(y1, y2, rtol=1.e-3)

x = th.rand(10, 4, 1)

batchnorm1 = nn.BatchNorm1d(4)
batchnorm2 = BatchNorm1d(4)
batchnorm3 = BatchNorm1d(4, normalise_over_final=True)

y1 = batchnorm1(dense1(x))
y2 = batchnorm2(dense1(x))
y3 = batchnorm3(dense1(x).transpose(1, 2))

assert th.allclose(y1, y2, rtol=1.e-3)
assert th.allclose(y1, y3.transpose(1, 2), rtol=1.e-3)
