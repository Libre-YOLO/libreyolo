"""Exponential Moving Average (EMA) for model weights.

Adapted from the official YOLOX repository.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn


def is_parallel(model):
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


class ModelEMA:
    """Model Exponential Moving Average.

    From https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, model, decay=0.9999, updates=0, tau=2000):
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        self.tau = tau
        if tau <= 0:
            self.decay = lambda x: decay
        else:
            self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = (
                model.module.state_dict() if is_parallel(model) else model.state_dict()
            )
            # EMA update: v = d*v + (1-d)*model[k] for every float tensor.
            ema_vals, model_vals = [], []
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_vals.append(v)
                    model_vals.append(msd[k])
            if ema_vals and ema_vals[0].is_cuda:
                # CUDA / ROCm: fuse the ~N per-parameter mul_/add_ launches into
                # two multi-tensor _foreach_ calls — a large win when the step is
                # launch-bound. Numerically identical (~1e-7).
                torch._foreach_mul_(ema_vals, d)
                torch._foreach_add_(ema_vals, model_vals, alpha=1.0 - d)
            else:
                # CPU / MPS / other backends: portable per-tensor path (identical
                # math; _foreach_ coverage is incomplete on MPS, and the launch
                # win is CUDA-specific anyway).
                for v, mv in zip(ema_vals, model_vals):
                    v.mul_(d).add_(mv, alpha=1.0 - d)

    def set_decay(self, decay: float, ramp: bool = False):
        """Replace the decay schedule (used for D-FINE-style EMA restart).

        ``ramp=True`` keeps the early-epoch ramp-up; ``ramp=False`` (default)
        sets a constant decay — the right choice when ``set_decay`` is called
        mid-training and the model is already past its noisy initial phase.
        """
        if ramp and self.tau > 0:
            self.decay = lambda x: decay * (1 - math.exp(-x / self.tau))
        else:
            self.decay = lambda x: decay
