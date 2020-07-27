import git
import torch
import numpy as np


def check_repo(allow_uncommitted):
    repo = git.Repo()
    if repo.is_dirty() and not allow_uncommitted:
        raise Warning("Repo contains uncommitted changes!")
    return repo


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_deterministic(seed):
    set_seed(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
