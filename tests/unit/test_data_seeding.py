"""Deterministic data-loader, worker, and distributed-sampler seeding."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from libreyolo.data.dataset import create_dataloader
from libreyolo.data.seeding import (
    data_seed_for_rank,
    dataloader_seed_kwargs,
    distributed_sampler_seed,
    seed_data_worker,
)

pytestmark = pytest.mark.unit


class _RandomDataset(Dataset):
    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int):
        return (
            index,
            random.randrange(2**31),
            int(np.random.randint(0, 2**31)),
            int(torch.randint(0, 2**31, ()).item()),
        )


class _YoloDataset(Dataset):
    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int):
        return (
            np.full((3, 2, 2), index, dtype=np.float32),
            np.zeros((1, 5), dtype=np.float32),
            (2, 2),
            index,
        )


def _collect_worker_stream(seed: int, *, rank: int = 0, distributed: bool = False):
    loader = DataLoader(
        _RandomDataset(),
        batch_size=1,
        shuffle=True,
        num_workers=2,
        **dataloader_seed_kwargs(seed, rank=rank, distributed=distributed),
    )
    return [tuple(int(value.item()) for value in batch) for batch in loader]


def test_worker_streams_reproduce_for_same_seed_and_diverge_by_rank_and_seed():
    first = _collect_worker_stream(41, rank=0, distributed=True)
    repeated = _collect_worker_stream(41, rank=0, distributed=True)
    other_rank = _collect_worker_stream(41, rank=1, distributed=True)
    other_seed = _collect_worker_stream(42, rank=0, distributed=True)

    assert first == repeated
    assert first != other_rank
    assert first != other_seed

    # Compare values by dataset index as well as shuffled order. This proves
    # Python, NumPy, and Torch worker streams changed, not merely the sampler.
    first_by_index = {row[0]: row[1:] for row in first}
    rank_by_index = {row[0]: row[1:] for row in other_rank}
    seed_by_index = {row[0]: row[1:] for row in other_seed}
    assert first_by_index != rank_by_index
    assert first_by_index != seed_by_index


def test_create_dataloader_preserves_contract_and_uses_rank_generator():
    loader = create_dataloader(
        _YoloDataset(),
        batch_size=4,
        num_workers=0,
        shuffle=True,
        pin_memory=False,
        seed=17,
        rank=2,
        distributed=True,
    )

    assert loader.batch_size == 4
    assert loader.drop_last is True
    assert loader.worker_init_fn is seed_data_worker
    assert loader.generator.initial_seed() == data_seed_for_rank(
        17, rank=2, distributed=True
    )

    ambient_loader = create_dataloader(
        _YoloDataset(),
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        seed=None,
    )
    assert ambient_loader.generator is None
    assert ambient_loader.worker_init_fn is None


def test_distributed_sampler_uses_common_seed_and_partitions_one_permutation():
    dataset = list(range(12))
    seed = distributed_sampler_seed(73)
    samplers = [
        DistributedSampler(
            dataset,
            num_replicas=2,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        for rank in range(2)
    ]
    for sampler in samplers:
        sampler.set_epoch(4)

    partitions = [list(sampler) for sampler in samplers]
    assert set(partitions[0]).isdisjoint(partitions[1])
    assert sorted(partitions[0] + partitions[1]) == dataset

    repeated = DistributedSampler(
        dataset,
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=distributed_sampler_seed(73),
        drop_last=False,
    )
    repeated.set_epoch(4)
    changed = DistributedSampler(
        dataset,
        num_replicas=2,
        rank=0,
        shuffle=True,
        seed=distributed_sampler_seed(74),
        drop_last=False,
    )
    changed.set_epoch(4)

    assert list(repeated) == partitions[0]
    assert list(changed) != partitions[0]
    assert distributed_sampler_seed(73) == distributed_sampler_seed(73)
    assert data_seed_for_rank(73, rank=0, distributed=True) != data_seed_for_rank(
        73, rank=1, distributed=True
    )
