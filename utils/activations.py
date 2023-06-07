import torch
import numpy as np
from scipy.special import logit, expit

def wrapper(number_fn, tensor_fn):
    def wrapped(x):
        if isinstance(x, torch.Tensor):
            return tensor_fn(x)
        else:
            return number_fn(x)
    return wrapped

activations = dict(
    abs=torch.abs,
    relu=torch.nn.functional.relu,
    sigmoid=torch.sigmoid,
    nothing=lambda x: x,
    exp=torch.exp,
)

inv_activations = dict(
    abs=wrapper(np.abs, torch.abs),
    nothing=lambda x: x,
    sigmoid=wrapper(logit, torch.logit),
    relu=lambda x: x,
    exp=wrapper(np.log, torch.log),
)
