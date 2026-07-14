"""DDP loader sharding for trainers that own their data pipeline.

``batch`` is the global batch under DDP: each rank's loader must be built
with ``batch // world_size`` over a DistributedSampler shard. Regression
coverage for issue #484 where multi-GPU YOLO-NAS-Pose (and the DEIM/D-FINE
trainers, inherited by DEIMv2/RT-DETRv4/EC) put the full global batch and
the full dataset on every rank.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

cv2 = pytest.importorskip("cv2")

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------


def _write_pose_dataset(tmp_path, num_keypoints=4, num_samples=4):
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(num_samples):
        cv2.imwrite(
            str(img_dir / f"sample{i}.jpg"),
            np.full((480, 640, 3), 127, dtype=np.uint8),
        )
        row = ["0", "0.5", "0.5", "0.3", "0.4"]
        for k in range(num_keypoints):
            row += [f"{0.4 + 0.02 * k:.3f}", f"{0.45 + 0.02 * k:.3f}", "2"]
        (lbl_dir / f"sample{i}.txt").write_text(" ".join(row) + "\n")

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "names": ["object"],
                "kpt_shape": [num_keypoints, 3],
            }
        )
    )
    return data_yaml


def _write_detect_dataset(tmp_path, num_samples=4):
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(num_samples):
        cv2.imwrite(
            str(img_dir / f"sample{i}.jpg"),
            np.full((480, 640, 3), 127, dtype=np.uint8),
        )
        (lbl_dir / f"sample{i}.txt").write_text("0 0.5 0.5 0.3 0.4\n")

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "nc": 1,
                "names": ["object"],
            }
        )
    )
    return data_yaml


def _fake_two_rank_ddp(trainer, rank=0):
    """Make the trainer believe it is rank ``rank`` of a 2-rank run.

    ``DistributedSampler`` takes explicit ``num_replicas``/``rank`` so no
    process group is required for loader wiring.
    """
    trainer.is_distributed = True
    trainer.world_size = 2
    trainer.rank = rank
    trainer.local_rank = rank


# ---------------------------------------------------------------------------
# YOLO-NAS-Pose
# ---------------------------------------------------------------------------


def _build_pose_trainer(data_yaml, **overrides):
    from libreyolo.models.yolonas.pose_trainer import YOLONASPoseTrainer

    kwargs = dict(
        model=torch.nn.Identity(),
        size="s",
        num_keypoints=4,
        data=str(data_yaml),
        batch=4,
        workers=0,
        device="cpu",
    )
    kwargs.update(overrides)
    return YOLONASPoseTrainer(**kwargs)


def test_yolonas_pose_loader_shards_batch_across_ranks(tmp_path):
    data_yaml = _write_pose_dataset(tmp_path)
    trainer = _build_pose_trainer(data_yaml)
    _fake_two_rank_ddp(trainer)

    trainer._setup_data()

    assert isinstance(trainer.train_loader.sampler, DistributedSampler)
    assert trainer.train_loader.batch_size == 2
    # 4 samples / 2 ranks / per-rank batch 2 = 1 iteration per rank.
    assert len(trainer.train_loader) == 1


def test_yolonas_pose_single_process_loader_unchanged(tmp_path):
    data_yaml = _write_pose_dataset(tmp_path)
    trainer = _build_pose_trainer(data_yaml)

    trainer._setup_data()

    assert not isinstance(trainer.train_loader.sampler, DistributedSampler)
    assert trainer.train_loader.batch_size == 4


# ---------------------------------------------------------------------------
# DEIM / D-FINE (inherited by DEIMv2, RT-DETRv4, EC-detect)
# ---------------------------------------------------------------------------


def _build_detr_trainer(trainer_cls, data_yaml, **overrides):
    kwargs = dict(
        model=torch.nn.Identity(),
        size="n",
        num_classes=1,
        data=str(data_yaml),
        epochs=1,
        batch=4,
        imgsz=640,
        device="cpu",
        amp=False,
        ema=False,
        workers=0,
        eval_interval=-1,
    )
    kwargs.update(overrides)
    return trainer_cls(**kwargs)


def _detr_trainer_classes():
    from libreyolo.models.deim.trainer import DEIMTrainer
    from libreyolo.models.dfine.trainer import DFINETrainer

    return {"deim": DEIMTrainer, "dfine": DFINETrainer}


@pytest.mark.parametrize("family", ["deim", "dfine"])
def test_detr_loader_shards_batch_across_ranks(tmp_path, family):
    trainer_cls = _detr_trainer_classes()[family]
    data_yaml = _write_detect_dataset(tmp_path)
    trainer = _build_detr_trainer(trainer_cls, data_yaml)
    _fake_two_rank_ddp(trainer)

    trainer._setup_data()

    assert isinstance(trainer.train_loader.sampler, DistributedSampler)
    assert trainer.train_loader.batch_size == 2
    assert len(trainer.train_loader) == 1


@pytest.mark.parametrize("family", ["deim", "dfine"])
def test_detr_single_process_loader_unchanged(tmp_path, family):
    trainer_cls = _detr_trainer_classes()[family]
    data_yaml = _write_detect_dataset(tmp_path)
    trainer = _build_detr_trainer(trainer_cls, data_yaml)

    trainer._setup_data()

    assert not isinstance(trainer.train_loader.sampler, DistributedSampler)
    assert trainer.train_loader.batch_size == 4


# ---------------------------------------------------------------------------
# YOLO-NAS-Pose validation gating
# ---------------------------------------------------------------------------


class _ExplodingLoader:
    """Iterating this loader means the rank ran validation when it must not."""

    def __iter__(self):
        raise AssertionError("validation ran on a non-main rank")


def test_yolonas_pose_validation_skipped_on_non_main_rank(tmp_path, monkeypatch):
    """Every rank used to run pose mAP validation, racing on the shared
    save_dir's predictions.json (issue #484 screenshots). Non-zero ranks must
    barrier and return without validating.
    """
    import libreyolo.training.distributed as dist_mod

    data_yaml = _write_pose_dataset(tmp_path)
    trainer = _build_pose_trainer(data_yaml)
    _fake_two_rank_ddp(trainer, rank=1)
    trainer.val_loader = _ExplodingLoader()

    barrier_calls = []
    monkeypatch.setattr(dist_mod, "barrier", lambda: barrier_calls.append(1))
    monkeypatch.setattr(dist_mod, "is_main_process", lambda: False)

    result = trainer._validate_epoch(0)

    assert result is None
    assert barrier_calls == [1]


def test_yolonas_pose_main_rank_barriers_after_validation(tmp_path, monkeypatch):
    import libreyolo.training.distributed as dist_mod

    data_yaml = _write_pose_dataset(tmp_path)
    trainer = _build_pose_trainer(data_yaml)
    _fake_two_rank_ddp(trainer, rank=0)
    trainer.val_loader = None

    barrier_calls = []
    monkeypatch.setattr(dist_mod, "barrier", lambda: barrier_calls.append(1))
    monkeypatch.setattr(dist_mod, "is_main_process", lambda: True)

    result = trainer._validate_epoch(0)

    assert result is None
    assert barrier_calls == [1]


def test_yolonas_pose_validate_epoch_accepts_save_plots(tmp_path):
    """BaseTrainer calls ``_validate_epoch(epoch, save_plots=True)`` on the
    final epoch when save_plots is set; the override must accept the kwarg.
    """
    data_yaml = _write_pose_dataset(tmp_path)
    trainer = _build_pose_trainer(data_yaml)
    trainer.val_loader = None

    assert trainer._validate_epoch(0, save_plots=True) is None


class _SpySampler(DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)
        super().set_epoch(epoch)


@pytest.mark.parametrize("family", ["deim", "dfine"])
def test_detr_train_epoch_sets_sampler_epoch(tmp_path, family):
    """The DEIM/D-FINE ``_train_epoch`` overrides must set the sampler epoch
    like ``BaseTrainer._train_epoch`` does, or every DDP epoch reuses the
    same shuffle order.
    """
    trainer_cls = _detr_trainer_classes()[family]
    data_yaml = _write_detect_dataset(tmp_path)
    model = torch.nn.Linear(2, 2)
    trainer = _build_detr_trainer(trainer_cls, data_yaml, model=model)

    empty_ds = TensorDataset(torch.empty(0, 2))
    sampler = _SpySampler(empty_ds, num_replicas=2, rank=0, shuffle=True)
    trainer.train_loader = DataLoader(empty_ds, batch_size=1, sampler=sampler)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainer._train_epoch(epoch=3)

    assert sampler.epochs == [3]
