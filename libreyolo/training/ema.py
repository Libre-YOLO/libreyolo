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


def _strided_storage_signature(tensor: torch.Tensor):
    """Return the storage and exact layout identity for a strided tensor."""
    storage = tensor.untyped_storage()
    layout = (
        storage,
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.is_conj(),
        tensor.is_neg(),
    )
    return layout, storage


def _tensor_alias_signature(tensor: torch.Tensor):
    """Identify exact aliases while keeping different storage views distinct."""
    if tensor.layout != torch.strided:
        # State-dict wrappers for uncommon layouts do not expose a general
        # storage-region contract. Keep them independent rather than risk
        # merging tensors that merely look alike.
        return (tensor.layout, id(tensor)), ()

    layout, storage = _strided_storage_signature(tensor)
    return (torch.strided, layout), (storage,)


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
            # EMA update: v = d*v + (1-d)*model[k] for every unique float
            # tensor layout. state_dict() retains every alias path, so updating
            # each key would apply the decay repeatedly to shared tensors.
            ema_vals, model_vals = [], []
            seen_layouts = set()
            storage_counts = {}
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    layout, storages = _tensor_alias_signature(v)
                    if layout in seen_layouts:
                        continue
                    seen_layouts.add(layout)
                    ema_vals.append(v)
                    model_vals.append(msd[k])
                    for storage in set(storages):
                        storage_counts[storage] = storage_counts.get(storage, 0) + 1
            has_overlapping_views = any(count > 1 for count in storage_counts.values())
            if has_overlapping_views:
                raise RuntimeError(
                    "ModelEMA does not support distinct state-dict tensor views "
                    "that share storage; make the views independent or exact aliases"
                )
            if (
                ema_vals
                and all(
                    v.is_cuda
                    and v.layout == torch.strided
                    and mv.layout == torch.strided
                    and v.device == mv.device
                    and v.dtype == mv.dtype
                    for v, mv in zip(ema_vals, model_vals)
                )
            ):
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
