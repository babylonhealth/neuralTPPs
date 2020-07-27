import torch
import torch.optim as optim

from torch.optim.lr_scheduler import _LRScheduler


def create_lr_scheduler(optimizer, args):
    if not isinstance(optimizer, optim.Optimizer):
        # assume FP16_Optimizer
        optimizer = optimizer.optimizer

    if args.lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            # NB This threshold is (not) used so that we only change lr if
            # there is a significant difference.
            threshold=0,
            patience=args.lr_scheduler_patience,
            factor=args.lr_scheduler_gamma)
    elif args.lr_scheduler == 'step':
        step_size = args.lr_scheduler_step_size
        gamma = args.lr_scheduler_gamma
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)
    elif args.lr_scheduler == 'cos':
        max_epochs = args.max_epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max_epochs)
    elif args.lr_scheduler == 'milestones':
        milestones = args.lr_scheduler_milestones
        gamma = args.lr_scheduler_gamma
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma)
    elif args.lr_scheduler == 'findlr':
        max_steps = args.max_steps
        lr_scheduler = FindLR(optimizer, max_steps)
    elif args.lr_scheduler == 'noam':
        warmup_steps = args.lr_scheduler_warmup
        lr_scheduler = NoamLR(optimizer, warmup_steps=warmup_steps)
    elif args.lr_scheduler == "clr":
        step_size = args.lr_scheduler_step_size
        learning_rate = args.lr_rate_init
        lr_scheduler_gamma = args.lr_scheduler_gamma
        mode = "exp_range"
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=learning_rate * 1.e-2,
            max_lr=learning_rate,
            step_size_up=step_size,
            step_size_down=step_size,
            mode=mode,
            cycle_momentum=False,
            gamma=lr_scheduler_gamma)
    elif args.lr_scheduler == 'calr':
        step_size = args.lr_scheduler_step_size
        learning_rate = args.lr_rate_init
        lr_scheduler_gamma = args.lr_scheduler_gamma
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=step_size,
            eta_min=learning_rate * lr_scheduler_gamma)
    else:
        raise NotImplementedError("unknown lr_scheduler " + args.lr_scheduler)
    return lr_scheduler


class FindLR(_LRScheduler):
    """
    inspired by fast.ai @https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    def __init__(self, optimizer, max_steps, max_lr=10):
        self.max_steps = max_steps
        self.max_lr = max_lr
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * ((self.max_lr / base_lr) ** (
                self.last_epoch / (self.max_steps - 1)))
                for base_lr in self.base_lrs]


class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(
            last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]
