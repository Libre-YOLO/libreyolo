"""DDP loader sharding for trainers that own their data pipeline.

``batch`` is the global batch under DDP: each rank's loader must be built
with ``batch // world_size`` over a DistributedSampler shard. Regression
coverage for issue #484 where multi-GPU YOLO-NAS-Pose (and the DEIM/D-FINE
trainers, inherited by DEIMv2/RT-DETRv4/EC) put the full global batch and
the full dataset on every rank.
"""

from __future__ import annotations

import contextlib
import socket
import sys
from types import SimpleNamespace

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
def test_detr_single_process_batch_and_sampler_unchanged(tmp_path, family):
    trainer_cls = _detr_trainer_classes()[family]
    data_yaml = _write_detect_dataset(tmp_path)
    trainer = _build_detr_trainer(trainer_cls, data_yaml)

    trainer._setup_data()

    assert not isinstance(trainer.train_loader.sampler, DistributedSampler)
    assert trainer.train_loader.batch_size == 4
    # 4 samples at batch 4: exactly one full batch survives drop_last.
    assert len(trainer.train_loader) == 1


@pytest.mark.parametrize("family", ["deim", "dfine"])
def test_detr_dataset_smaller_than_batch_keeps_partial_batch(tmp_path, family):
    """Deliberate behavior change from unconditional ``drop_last=True``: a
    dataset smaller than ``batch`` used to yield ZERO training iterations
    silently. It now keeps the partial batch, matching BaseTrainer's
    visible-count drop_last policy.
    """
    trainer_cls = _detr_trainer_classes()[family]
    data_yaml = _write_detect_dataset(tmp_path, num_samples=2)
    trainer = _build_detr_trainer(trainer_cls, data_yaml)

    trainer._setup_data()

    assert trainer.train_loader.drop_last is False
    assert len(trainer.train_loader) == 1


# ---------------------------------------------------------------------------
# YOLO-NAS-Pose validation gating
# ---------------------------------------------------------------------------


def test_yolonas_pose_non_main_rank_runs_loss_but_not_metric_phase(
    tmp_path, monkeypatch
):
    """Non-zero ranks must run the val-loss pass (YoloNASPoseLoss all-reduces
    its normalizer, a collective, so skipping the pass on some ranks pairs
    rank 0's all_reduce with the others' barrier and deadlocks) but must NOT
    run the pose mAP validator, which races on the shared save_dir's
    predictions.json (issue #484 screenshots).
    """
    import libreyolo.training.distributed as dist_mod

    data_yaml = _write_pose_dataset(tmp_path)
    trainer = _build_pose_trainer(data_yaml)
    _fake_two_rank_ddp(trainer, rank=1)
    trainer.ema_model = None
    trainer.val_loader = [(torch.zeros(1, 3, 64, 64), torch.zeros(1, 2, 17))]

    loss_calls = []

    def _stub_loss(outputs, targets):
        loss_calls.append(1)
        return torch.tensor(0.5), None

    trainer.loss_fn = _stub_loss
    trainer._run_pose_metric_validation = lambda *a, **k: pytest.fail(
        "pose mAP validation ran on a non-main rank"
    )

    barrier_calls = []
    monkeypatch.setattr(dist_mod, "barrier", lambda: barrier_calls.append(1))
    monkeypatch.setattr(dist_mod, "is_main_process", lambda: False)

    result = trainer._validate_epoch(0)

    assert result is None
    assert loss_calls == [1]
    assert barrier_calls == [1]


def test_yolonas_pose_main_rank_barriers_after_validation(tmp_path, monkeypatch):
    """Rank 0 must hit the finally-barrier after running the validation body
    (empty val loader drives the full body without needing a real model).
    """
    import libreyolo.training.distributed as dist_mod

    data_yaml = _write_pose_dataset(tmp_path)
    trainer = _build_pose_trainer(data_yaml)
    _fake_two_rank_ddp(trainer, rank=0)
    trainer.val_loader = []
    trainer.ema_model = None
    trainer.wrapper_model = None

    barrier_calls = []
    monkeypatch.setattr(dist_mod, "barrier", lambda: barrier_calls.append(1))
    monkeypatch.setattr(dist_mod, "is_main_process", lambda: True)

    result = trainer._validate_epoch(0)

    assert result is not None
    assert result["best_metric_key"] == "loss/val"
    assert barrier_calls == [1]


def test_yolonas_pose_validate_epoch_accepts_save_plots(tmp_path):
    """BaseTrainer calls ``_validate_epoch(epoch, save_plots=True)`` on the
    final epoch when save_plots is set; the override must accept the kwarg.
    """
    data_yaml = _write_pose_dataset(tmp_path)
    trainer = _build_pose_trainer(data_yaml)
    trainer.val_loader = None

    assert trainer._validate_epoch(0, save_plots=True) is None


@pytest.mark.parametrize(
    "save_plots,expected",
    [(True, True), (False, False), (None, False)],
)
def test_yolonas_pose_save_plots_reaches_validation_config(
    tmp_path, monkeypatch, save_plots, expected
):
    """``save_plots`` must flow into ``ValidationConfig`` (None resolves to the
    config default, final-epoch only), not be accepted and dropped.
    """
    import libreyolo.validation as validation_mod

    data_yaml = _write_pose_dataset(tmp_path)
    trainer = _build_pose_trainer(data_yaml, epochs=10)
    trainer.save_dir = tmp_path
    trainer.wrapper_model = SimpleNamespace(model=None)

    captured = {}

    class _FakeValidator:
        def __init__(self, model, config):
            captured["config"] = config

        def run(self):
            return {}

    monkeypatch.setattr(validation_mod, "PoseValidator", _FakeValidator)

    trainer._run_pose_metric_validation(
        torch.nn.Identity(), epoch=0, save_plots=save_plots
    )

    assert captured["config"].save_plots is expected


class _SpySampler(DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)
        super().set_epoch(epoch)


@pytest.mark.parametrize("family", ["deim", "dfine"])
@pytest.mark.parametrize("accum", [False, True])
def test_detr_train_epoch_sets_sampler_epoch(tmp_path, family, accum):
    """The DEIM/D-FINE ``_train_epoch`` AND ``_train_epoch_accum`` overrides
    must set the sampler epoch like ``BaseTrainer._train_epoch`` does, or
    every DDP epoch reuses the same shuffle order.
    """
    trainer_cls = _detr_trainer_classes()[family]
    data_yaml = _write_detect_dataset(tmp_path)
    model = torch.nn.Linear(2, 2)
    # nbs=8 with batch=4 makes _accum_steps=2, so _train_epoch delegates to
    # _train_epoch_accum.
    overrides = {"model": model}
    if accum:
        overrides["nbs"] = 8
    trainer = _build_detr_trainer(trainer_cls, data_yaml, **overrides)
    assert (trainer._accum_steps > 1) is accum

    empty_ds = TensorDataset(torch.empty(0, 2))
    sampler = _SpySampler(empty_ds, num_replicas=2, rank=0, shuffle=True)
    trainer.train_loader = DataLoader(empty_ds, batch_size=1, sampler=sampler)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainer._train_epoch(epoch=3)

    assert sampler.epochs == [3]


# ---------------------------------------------------------------------------
# Global-batch divisibility guard (BaseTrainer.setup, covers every family)
# ---------------------------------------------------------------------------


class _GuardPassed(Exception):
    """Sentinel proving setup() got past the divisibility guard."""


@pytest.mark.parametrize("batch", [6, 2])
def test_setup_rejects_non_divisible_global_batch(tmp_path, monkeypatch, batch):
    """batch is the global batch; batch=6 or 2 on 4 ranks would silently
    train at 4. setup() must fail fast instead.
    """
    data_yaml = _write_pose_dataset(tmp_path)
    trainer = _build_pose_trainer(data_yaml, batch=batch)
    _fake_two_rank_ddp(trainer)
    trainer.world_size = 4

    with pytest.raises(ValueError, match="divisible by world_size"):
        trainer.setup()


@pytest.mark.parametrize("distributed,batch", [(True, 8), (False, 7)])
def test_setup_accepts_divisible_or_single_process_batch(
    tmp_path, monkeypatch, distributed, batch
):
    data_yaml = _write_pose_dataset(tmp_path)
    trainer = _build_pose_trainer(data_yaml, batch=batch)
    if distributed:
        _fake_two_rank_ddp(trainer)
        trainer.world_size = 4

    def _sentinel():
        raise _GuardPassed

    monkeypatch.setattr(trainer, "_setup_data", _sentinel)
    with pytest.raises(_GuardPassed):
        trainer.setup()


# ---------------------------------------------------------------------------
# Real 2-rank gloo regression: validation collectives must stay aligned
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pose_val_gate_worker(rank, world_size, port, out_dir):
    import os
    from pathlib import Path

    import torch
    import torch.distributed as dist

    out_path = Path(out_dir) / f"rank_{rank}.txt"
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        # Short op timeout: if the collective mismatch this test guards
        # against ever returns, fail within a minute instead of hanging the
        # unit job on PyTorch's long default.
        from datetime import timedelta

        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )

        from libreyolo.data import default_oks_sigmas
        from libreyolo.models.yolonas.loss import YoloNASPoseLoss
        from libreyolo.models.yolonas.nn import LibreYOLONASPoseModel
        from libreyolo.models.yolonas.pose_trainer import YOLONASPoseTrainer

        K = 4
        torch.manual_seed(0)
        model = LibreYOLONASPoseModel(config="s", num_keypoints=K).eval()

        targets = torch.zeros(1, 4, 5 + 3 * K)
        targets[0, 0, 1:5] = torch.tensor([320.0, 300.0, 120.0, 200.0])
        for k in range(K):
            targets[0, 0, 5 + 3 * k] = 320.0 + k * 5.0
            targets[0, 0, 5 + 3 * k + 1] = 300.0 + k * 5.0
            targets[0, 0, 5 + 3 * k + 2] = 2.0

        trainer = YOLONASPoseTrainer.__new__(YOLONASPoseTrainer)
        trainer.is_distributed = True
        trainer.rank = rank
        trainer.world_size = world_size
        trainer.device = torch.device("cpu")
        trainer.model = model
        trainer.ema_model = None
        trainer.loss_fn = YoloNASPoseLoss(oks_sigmas=default_oks_sigmas(K))
        trainer.val_loader = [(torch.zeros(1, 3, 640, 640), targets)]
        trainer._run_pose_metric_validation = lambda *a, **k: None

        result = trainer._validate_epoch(0)

        if rank == 0:
            assert result is not None and "loss/val" in result["metrics"]
        else:
            assert result is None
        out_path.write_text("ok\n")
    except Exception as exc:
        out_path.write_text(f"error: {type(exc).__name__}: {exc}\n")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    sys.platform == "win32" and sys.version_info < (3, 8),
    reason="mp.spawn on Windows needs Python 3.8+",
)
def test_yolonas_pose_validation_collectives_align_2_ranks_gloo(tmp_path):
    """Regression for the collective mismatch: YoloNASPoseLoss calls
    all_reduce_avg_scalar, so the val-loss pass must run on every rank. With
    a rank-0-only gate around the whole body, gloo pairs rank 0's all_reduce
    with rank 1's barrier and the run errors or hangs.
    """
    import torch.multiprocessing as mp

    port = _free_port()
    mp.spawn(
        _pose_val_gate_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True
    )
    for rank in range(2):
        text = (tmp_path / f"rank_{rank}.txt").read_text()
        assert text == "ok\n", f"rank {rank}: {text!r}"
