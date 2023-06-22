# learning rate schedulers
import torch
import numpy as np


def linear_warmup(warmup_steps, lr_start, lr_end):
    def _warmup(step):
        if step < warmup_steps:
            return lr_start + (lr_end - lr_start) * step / warmup_steps
        else:
            return lr_end

    return _warmup


def cosine_warmup(warmup_steps, lr_start, lr_end):
    def _warmup(step):
        if step < warmup_steps:
            return lr_start + (lr_end - lr_start) * step / warmup_steps
        else:
            return lr_end

    return _warmup


lr_warmups = {}

lr_schedulers = {}
