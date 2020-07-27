import torch as th


class Log(th.autograd.Function):
    """Safe implementation of x ↦ log(x)."""
    @staticmethod
    def forward(ctx, x):
        log = x.log()
        ctx.save_for_backward(x)
        return log

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return th.clamp(grad_output / x, min=-1e1, max=1e1)
