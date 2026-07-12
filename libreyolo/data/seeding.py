"""Deterministic, rank-aware random seeding for training data loaders."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch

_SEED_MODULUS = 2**63


def normalize_data_seed(seed: int) -> int:
    """Normalize an integer seed to the non-negative range Torch accepts."""
    try:
        value = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"data-loader seed must be an integer, got {seed!r}") from exc
    return value % _SEED_MODULUS


def data_seed_for_rank(
    seed: int,
    *,
    rank: int = 0,
    distributed: bool = False,
) -> int:
    """Return the configured seed, offset once per distributed rank.

    This mirrors the trainer's rank seed contract: single-process training uses
    the configured seed unchanged, while distributed rank ``r`` uses
    ``seed + 1 + r``. The rank offset belongs on the DataLoader generator and
    worker streams, not on ``DistributedSampler.seed``; sampler ranks must
    share one permutation before partitioning it.
    """
    rank = int(rank)
    if rank < 0:
        raise ValueError(f"data-loader rank must be non-negative, got {rank}")
    base_seed = normalize_data_seed(seed)
    if not distributed:
        return base_seed
    return (base_seed + rank + 1) % _SEED_MODULUS


def distributed_sampler_seed(seed: Optional[int]) -> int:
    """Return the common seed all ranks must pass to DistributedSampler."""
    return 0 if seed is None or int(seed) < 0 else normalize_data_seed(seed)


def make_data_generator(
    seed: int,
    *,
    rank: int = 0,
    distributed: bool = False,
) -> torch.Generator:
    """Build an isolated Torch generator for shuffling and worker base seeds."""
    generator = torch.Generator()
    generator.manual_seed(data_seed_for_rank(seed, rank=rank, distributed=distributed))
    return generator


def seed_data_worker(worker_id: int) -> None:
    """Seed Python, NumPy, and Torch from this DataLoader worker's seed.

    PyTorch assigns ``torch.initial_seed() == base_seed + worker_id`` before
    calling ``worker_init_fn``. Do not add ``worker_id`` a second time.
    """
    del worker_id
    worker_seed = torch.initial_seed()
    random.seed(worker_seed)
    np.random.seed(worker_seed % 2**32)
    torch.manual_seed(worker_seed)


def dataloader_seed_kwargs(
    seed: Optional[int],
    *,
    rank: int = 0,
    distributed: bool = False,
) -> dict:
    """Return deterministic ``generator`` and ``worker_init_fn`` kwargs.

    ``seed=None`` preserves DataLoader's default ambient-RNG behavior.
    """
    if seed is None or int(seed) < 0:
        return {}
    return {
        "generator": make_data_generator(
            seed,
            rank=rank,
            distributed=distributed,
        ),
        "worker_init_fn": seed_data_worker,
    }


__all__ = [
    "data_seed_for_rank",
    "dataloader_seed_kwargs",
    "distributed_sampler_seed",
    "make_data_generator",
    "normalize_data_seed",
    "seed_data_worker",
]
