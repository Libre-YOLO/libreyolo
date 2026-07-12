"""CPU/Gloo regressions for globally count-normalized DDP losses."""

from __future__ import annotations

import contextlib
import json
import socket
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

pytestmark = pytest.mark.unit


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _distributed_average(value: torch.Tensor) -> float:
    reduced = value.detach().clone()
    dist.all_reduce(reduced)
    return float((reduced / dist.get_world_size()).item())


def _semantic_case(rank: int, loss_function) -> dict[str, float]:
    logits = torch.tensor(
        [[[[2.0, 0.0, -1.0, 3.0]], [[0.0, 1.0, 2.0, -2.0]]]],
        requires_grad=True,
    )
    if rank == 0:
        targets = torch.full((1, 1, 4), 255, dtype=torch.long)
    else:
        targets = torch.tensor([[[0, 1, 1, 0]]], dtype=torch.long)

    loss = loss_function(logits, targets, ignore_index=255)
    local_sum = F.cross_entropy(
        logits,
        targets,
        ignore_index=255,
        reduction="sum",
    ).detach()
    local_count = (targets != 255).sum().to(dtype=torch.float32)
    global_sum = local_sum.clone()
    global_count = local_count.clone()
    dist.all_reduce(global_sum)
    dist.all_reduce(global_count)
    expected = float((global_sum / global_count).item())
    averaged = _distributed_average(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    return {"averaged": averaged, "expected": expected}


def _dfine_mask_case(rank: int) -> dict[str, float]:
    from libreyolo.models.dfine.loss import DFINECriterion
    from libreyolo.training.distributed import all_reduce_avg_scalar

    instance_count = 1 if rank == 0 else 3
    logit_value = -3.0 if rank == 0 else 3.0
    pred_masks = torch.full(
        (1, instance_count, 2, 2),
        logit_value,
        dtype=torch.float32,
        requires_grad=True,
    )
    target_masks = torch.ones(instance_count, 2, 2)
    targets = [
        {
            "labels": torch.zeros(instance_count, dtype=torch.long),
            "boxes": torch.tensor(
                [[0.5, 0.5, 1.0, 1.0]] * instance_count,
                dtype=torch.float32,
            ),
            "masks": target_masks,
        }
    ]
    matched = torch.arange(instance_count, dtype=torch.long)
    indices = [(matched, matched)]
    normalizer = all_reduce_avg_scalar(
        instance_count,
        device=pred_masks.device,
        min_value=1.0,
    )

    criterion = DFINECriterion(
        matcher=None,
        weight_dict={},
        losses=["masks"],
        num_classes=1,
    )
    losses = criterion.loss_masks(
        {"pred_masks": pred_masks},
        targets,
        indices,
        normalizer,
    )

    bce_per_instance = F.binary_cross_entropy_with_logits(
        pred_masks[0],
        target_masks,
        reduction="none",
    ).mean(dim=(1, 2))
    probabilities = pred_masks[0].sigmoid().flatten(1)
    target_flat = target_masks.flatten(1)
    intersection = (probabilities * target_flat).sum(dim=1)
    denominator = probabilities.sum(dim=1) + target_flat.sum(dim=1) + 1e-6
    dice_per_instance = 1.0 - (2.0 * intersection + 1e-6) / denominator

    local_bce_sum = bce_per_instance.sum().detach()
    local_dice_sum = dice_per_instance.sum().detach()
    global_bce_sum = local_bce_sum.clone()
    global_dice_sum = local_dice_sum.clone()
    global_count = torch.tensor(float(instance_count))
    dist.all_reduce(global_bce_sum)
    dist.all_reduce(global_dice_sum)
    dist.all_reduce(global_count)

    result = {
        "bce_averaged": _distributed_average(losses["loss_mask_bce"]),
        "bce_expected": float((global_bce_sum / global_count).item()),
        "dice_averaged": _distributed_average(losses["loss_mask_dice"]),
        "dice_expected": float((global_dice_sum / global_count).item()),
        "normalizer": normalizer,
    }
    (losses["loss_mask_bce"] + losses["loss_mask_dice"]).backward()
    assert pred_masks.grad is not None
    assert torch.isfinite(pred_masks.grad).all()
    return result


def _sparse_count_normalizer_case(rank: int) -> dict[str, float]:
    from libreyolo.models.deim.loss import DEIMCriterion
    from libreyolo.models.ec.pose_loss import ECPoseCriterion
    from libreyolo.models.rfdetr.loss import SetCriterion

    local_count = rank
    device = torch.device("cpu")
    ec_criterion = ECPoseCriterion.__new__(ECPoseCriterion)
    targets = [{"labels": torch.zeros(local_count, dtype=torch.long)}]
    matched = torch.arange(local_count, dtype=torch.long)
    indices = [(matched, matched)]

    return {
        "deim": DEIMCriterion._global_count_normalizer(local_count, device),
        # DEIMv2 inherits this exact normalization seam and calls it from its
        # own forward without importing the restricted backbone package here.
        "deimv2": DEIMCriterion._global_count_normalizer(local_count, device),
        "rfdetr": SetCriterion._global_count_normalizer(local_count, device),
        "ec": ec_criterion._global_count_normalizer(local_count, device),
        "ec_boxes": ec_criterion._num_boxes(targets, device),
        "ec_indices": ec_criterion._num_index_pairs(indices, device),
    }


def _count_normalization_worker(
    rank: int,
    world_size: int,
    port: int,
    output_dir: str,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        from libreyolo.models.rfdetr.nn import (
            _semantic_cross_entropy as dinov2_semantic_loss,
        )
        from libreyolo.models.segformer.nn import (
            _semantic_cross_entropy as segformer_semantic_loss,
        )

        record = {
            "dinov2": _semantic_case(rank, dinov2_semantic_loss),
            "segformer": _semantic_case(rank, segformer_semantic_loss),
            "dfine_masks": _dfine_mask_case(rank),
            "sparse_counts": _sparse_count_normalizer_case(rank),
        }
        (Path(output_dir) / f"rank-{rank}.json").write_text(
            json.dumps(record, sort_keys=True),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("count", "expected"), [(0, 1.0), (1, 1.0), (3, 3.0)])
def test_sparse_count_normalizers_preserve_single_process_behavior(count, expected):
    from libreyolo.models.deim.loss import DEIMCriterion
    from libreyolo.models.ec.pose_loss import ECPoseCriterion
    from libreyolo.models.rfdetr.loss import SetCriterion

    device = torch.device("cpu")
    assert DEIMCriterion._global_count_normalizer(count, device) == expected
    assert SetCriterion._global_count_normalizer(count, device) == expected
    assert ECPoseCriterion._global_count_normalizer(count, device) == expected


def test_ddp_losses_match_true_global_count_weighted_means(tmp_path):
    world_size = 2
    mp.spawn(
        _count_normalization_worker,
        args=(world_size, _free_port(), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    records = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    assert records[0] == records[1]
    for family in ("dinov2", "segformer"):
        assert records[0][family]["averaged"] == pytest.approx(
            records[0][family]["expected"], abs=1e-6
        )

    mask_result = records[0]["dfine_masks"]
    assert mask_result["normalizer"] == pytest.approx(2.0)
    assert mask_result["bce_averaged"] == pytest.approx(
        mask_result["bce_expected"], abs=1e-6
    )
    assert mask_result["dice_averaged"] == pytest.approx(
        mask_result["dice_expected"], abs=1e-6
    )

    for normalizer in records[0]["sparse_counts"].values():
        assert normalizer == pytest.approx(0.5)
