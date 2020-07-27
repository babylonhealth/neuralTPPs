import numpy as np
import torch as th


TORCH_TO_NUMPY = {
    th.float32: np.float32}
NUMPY_TO_TORCH = {k: v for v, k in TORCH_TO_NUMPY.items()}
