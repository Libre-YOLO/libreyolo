"""Rank-zero pose validation must not enter training-loss collectives."""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.ec.pose_trainer import ECPoseTrainer
from libreyolo.models.yolonas.pose_trainer import YOLONASPoseTrainer

pytestmark = pytest.mark.unit


class _ForbiddenLoader:
    def __iter__(self):
        raise AssertionError("DDP rank-zero validation iterated the loss loader")


@pytest.mark.parametrize("trainer_class", [YOLONASPoseTrainer, ECPoseTrainer])
def test_ddp_pose_validation_skips_collective_loss(trainer_class):
    trainer = trainer_class.__new__(trainer_class)
    trainer.is_distributed = True
    trainer.val_loader = _ForbiddenLoader()
    trainer.model = torch.nn.Identity()
    trainer.ema_model = None
    trainer.best_metric_key = "metrics/keypoints_mAP50-95"
    trainer._run_pose_metric_validation = lambda *args, **kwargs: {
        "metrics/keypoints_mAP50": 0.6,
        "metrics/keypoints_mAP50-95": 0.4,
    }

    result = trainer_class._run_validation(trainer, 0)

    assert result["mAP50"] == pytest.approx(0.6)
    assert result["mAP50_95"] == pytest.approx(0.4)
    assert "loss/val" not in result["metrics"]
