"""Unit coverage for opt-in YOLO9 and RF-DETR validation loss."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from libreyolo.models.rfdetr import loss as rfdetr_loss_module
from libreyolo.models.rfdetr.config import RFDETRConfig
from libreyolo.models.rfdetr.loss import SetCriterion
from libreyolo.models.rfdetr.trainer import RFDETRTrainer
from libreyolo.models.rfdetr.validation_loss import RFDETRValidationLoss
from libreyolo.models.yolo9 import loss as yolo9_loss_module
from libreyolo.models.yolo9.loss import YOLO9Loss
from libreyolo.models.yolo9.trainer import YOLO9Trainer
from libreyolo.models.yolo9.validation_loss import YOLO9ValidationLoss
from libreyolo.models.yolo9_e2e.trainer import YOLO9E2ETrainer
from libreyolo.models.yolo9_p2.trainer import YOLO9P2Trainer
from libreyolo.training.config import TrainConfig, YOLO9Config
from libreyolo.training.trainer import BaseTrainer
from libreyolo.training.loggers.base import epoch_metrics
from libreyolo.ui.train_monitor_page import INDEX_HTML
from libreyolo.validation.config import ValidationConfig
from libreyolo.validation.detection_validator import DetectionValidator

pytestmark = pytest.mark.unit


class _Adapter:
    max_labels = 300

    def __init__(self, *, fail_on: int | None = None):
        self.calls = 0
        self.fail_on = fail_on
        self.image_sizes = []

    def __call__(self, predictions, targets, *, image_size):
        del predictions, targets
        self.calls += 1
        self.image_sizes.append(image_size)
        if self.calls == self.fail_on:
            raise RuntimeError("synthetic adapter failure")
        value = float(self.calls * 2 - 1)
        return {
            "loss": torch.tensor(value),
            "loss/box": value + 1.0,
        }


def _validator(adapter: _Adapter, *, augment: bool = False) -> DetectionValidator:
    model = SimpleNamespace(nb_classes=2)
    config = ValidationConfig(
        data_dir=".",
        device="cpu",
        verbose=False,
        augment=augment,
    )
    return DetectionValidator(model, config, loss_adapter=adapter)


def test_validation_loss_defaults_to_auto():
    """Unset means auto: enabled wherever the family implements it."""
    assert TrainConfig().val_loss is None
    assert YOLO9Config().val_loss is None
    assert RFDETRConfig().val_loss is None


def _resolved(trainer_class, *, task, model, val_loss):
    trainer = trainer_class.__new__(trainer_class)
    trainer.config = SimpleNamespace(val_loss=val_loss)
    trainer.wrapper_model = SimpleNamespace(task=task)
    trainer.model = model
    trainer.validate_validation_loss_config()
    return trainer.val_loss_enabled


def _yolo9_model():
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    return LibreYOLO9Model(config="t", nb_classes=80, img_size=640)


@pytest.mark.parametrize(
    "trainer_class,task",
    [
        (YOLO9E2ETrainer, "detect"),
        (YOLO9P2Trainer, "detect"),
        (YOLO9Trainer, "segment"),
    ],
)
def test_auto_default_disables_unsupported_yolo9_variants(trainer_class, task):
    """Regression: the auto default must never break a run nobody configured.

    yolo9_e2e and yolo9_p2 subclass YOLO9Config, so a plain ``True`` default
    would have raised at setup for every one of their runs.
    """
    assert (
        _resolved(trainer_class, task=task, model=SimpleNamespace(), val_loss=None)
        is False
    )


@pytest.mark.parametrize(
    "task", ["segment", "pose", "obb", "classify", "semantic"]
)
def test_auto_default_disables_non_detection_rfdetr_tasks(task):
    """RFDETRConfig serves every RF-DETR task, so auto must stay silent."""
    assert (
        _resolved(RFDETRTrainer, task=task, model=SimpleNamespace(), val_loss=None)
        is False
    )


def test_auto_default_enables_standard_yolo9_detection():
    assert (
        _resolved(YOLO9Trainer, task="detect", model=_yolo9_model(), val_loss=None)
        is True
    )


def test_explicit_false_disables_supported_family():
    assert (
        _resolved(YOLO9Trainer, task="detect", model=_yolo9_model(), val_loss=False)
        is False
    )


def test_unsupported_family_rejects_explicit_request():
    """A family with no adapter must fail loudly rather than drop the request."""

    trainer = SimpleNamespace(
        config=SimpleNamespace(val_loss=True),
        get_model_family=lambda: "yolox",
        val_loss_enabled=None,
    )
    trainer.resolve_val_loss = BaseTrainer.resolve_val_loss.__get__(trainer)
    trainer.validate_validation_loss_config = (
        BaseTrainer.validate_validation_loss_config.__get__(trainer)
    )
    with pytest.raises(ValueError, match="not supported by yolox"):
        trainer.validate_validation_loss_config()

    trainer.config = SimpleNamespace(val_loss=None)
    trainer.validate_validation_loss_config()
    assert trainer.val_loss_enabled is False


def test_non_finite_loss_is_discarded_and_reported(caplog):
    """A NaN must never reach the metric history."""

    class _NanAdapter:
        max_labels = 100

        def __call__(self, predictions, targets, *, image_size):
            del predictions, targets, image_size
            return {"loss": torch.tensor(float("nan")), "loss/box": 1.0}

    validator = _validator(_NanAdapter())
    validator._update_batch_metrics(
        {}, torch.zeros(1, 3, 16, 16), torch.zeros(1, 2, 5)
    )

    assert validator._validation_loss_metrics() == {}
    assert "non-finite validation loss" in caplog.text
    assert "amp=False" in caplog.text


def test_infinite_loss_is_also_discarded():
    class _InfAdapter:
        max_labels = 100

        def __call__(self, predictions, targets, *, image_size):
            del predictions, targets, image_size
            return {"loss": torch.tensor(float("inf"))}

    validator = _validator(_InfAdapter())
    validator._update_batch_metrics(
        {}, torch.zeros(1, 3, 16, 16), torch.zeros(1, 2, 5)
    )
    assert validator._validation_loss_metrics() == {}


def test_validation_loss_time_has_its_own_speed_bucket():
    """The cost of an optional metric must be visible, not hidden in a gap."""
    validator = _validator(_Adapter())
    assert "val_loss" in validator.speed


@pytest.mark.parametrize("trainer_class", [YOLO9E2ETrainer, YOLO9P2Trainer])
def test_yolo9_derived_variants_reject_validation_loss(trainer_class):
    trainer = trainer_class.__new__(trainer_class)
    trainer.config = SimpleNamespace(val_loss=True)
    trainer.wrapper_model = SimpleNamespace(task="detect")
    trainer.model = SimpleNamespace()

    with pytest.raises(ValueError, match="standard YOLO9 detection only"):
        trainer.validate_validation_loss_config()


def test_rfdetr_non_detection_task_rejects_validation_loss():
    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.config = SimpleNamespace(val_loss=True)
    trainer.wrapper_model = SimpleNamespace(task="segment")
    trainer.model = SimpleNamespace()

    with pytest.raises(ValueError, match="RF-DETR detection only"):
        trainer.validate_validation_loss_config()


def test_detection_validator_averages_loss_and_expands_target_capacity():
    adapter = _Adapter()
    validator = _validator(adapter)
    validator.val_preproc = SimpleNamespace(max_labels=120)
    validator._ensure_validation_loss_target_capacity()
    assert validator.val_preproc.max_labels == 300

    images = torch.zeros(2, 3, 32, 48)
    targets = torch.zeros(2, 4, 5)
    validator._update_batch_metrics({}, images, targets)
    validator._update_batch_metrics({}, images, targets)

    assert adapter.image_sizes == [(32, 48), (32, 48)]
    assert validator._validation_loss_metrics() == pytest.approx(
        {
            "metrics/loss": 2.0,
            "metrics/loss/box": 3.0,
        }
    )


def test_detection_validator_discards_partial_loss_after_adapter_failure(caplog):
    adapter = _Adapter(fail_on=2)
    validator = _validator(adapter)
    images = torch.zeros(1, 3, 16, 16)
    targets = torch.zeros(1, 2, 5)

    validator._update_batch_metrics({}, images, targets)
    validator._update_batch_metrics({}, images, targets)
    validator._update_batch_metrics({}, images, targets)

    assert adapter.calls == 2
    assert validator._validation_loss_metrics() == {}
    assert "detection metrics will continue" in caplog.text


def test_validation_loss_rejects_augmented_validation():
    with pytest.raises(ValueError, match="augmented validation"):
        _validator(_Adapter(), augment=True)


def test_yolo9_validation_target_conversion_compacts_and_normalizes():
    targets = torch.tensor(
        [
            [[20.0, 10.0, 60.0, 30.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
            [[50.0, 25.0, 150.0, 75.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        ]
    )

    converted = YOLO9ValidationLoss._prepare_targets(
        targets,
        image_size=(100, 200),
        num_classes=3,
        device=torch.device("cpu"),
    )

    assert converted.shape == (2, 1, 5)
    assert converted[0, 0].tolist() == pytest.approx([2.0, 0.1, 0.1, 0.3, 0.3])
    assert converted[1, 0].tolist() == pytest.approx([1.0, 0.25, 0.25, 0.75, 0.75])


def test_yolo9_adapter_reuses_raw_eval_outputs():
    class _Loss:
        def __init__(self):
            self.image_size = None
            self.raw_outputs = None

        def update_anchors(self, image_size):
            self.image_size = image_size

        def __call__(self, raw_outputs, targets):
            self.raw_outputs = raw_outputs
            assert targets.shape == (1, 1, 5)
            return {
                "total_loss": torch.tensor(10.0),
                "box_loss": torch.tensor(4.0),
                "cls_loss": torch.tensor(3.0),
                "dfl_loss": torch.tensor(3.0),
            }

    adapter = object.__new__(YOLO9ValidationLoss)
    adapter.device = torch.device("cpu")
    adapter.num_classes = 2
    adapter.max_labels = 100
    adapter.loss = _Loss()
    raw_outputs = [torch.zeros(1, 66, 4, 4)]
    targets = torch.tensor([[[1.0, 2.0, 5.0, 6.0, 1.0]]])

    values = adapter(
        {"predictions": torch.empty(0), "raw_outputs": raw_outputs},
        targets,
        image_size=(8, 8),
    )

    assert adapter.loss.raw_outputs is raw_outputs
    assert adapter.loss.image_size == [8, 8]
    assert set(values) == {"loss", "loss/box", "loss/cls", "loss/dfl"}


def test_yolo9_rank_local_normalizer_skips_collective(monkeypatch):
    def _unexpected_collective(value):
        del value
        raise AssertionError("rank-local validation entered a collective")

    monkeypatch.setattr(
        yolo9_loss_module, "all_reduce_avg_scalar", _unexpected_collective
    )
    loss = YOLO9Loss(
        num_classes=2,
        reg_max=16,
        strides=[8, 16, 32],
        image_size=None,
        device=torch.device("cpu"),
        distributed_normalize=False,
    )

    assert loss._global_cls_norm(torch.tensor([2.0, 3.0])) == pytest.approx(5.0)
    assert loss._global_cls_norm(torch.tensor([0.0])) == pytest.approx(1.0)


def test_rfdetr_validation_target_conversion_to_normalized_cxcywh():
    targets = torch.tensor([[[20.0, 10.0, 60.0, 30.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0]]])

    converted = RFDETRValidationLoss._prepare_targets(
        targets,
        image_size=(100, 200),
        num_classes=3,
        device=torch.device("cpu"),
    )

    assert len(converted) == 1
    assert converted[0]["labels"].tolist() == [2]
    assert converted[0]["boxes"][0].tolist() == pytest.approx([0.2, 0.2, 0.2, 0.2])


def test_rfdetr_adapter_uses_full_weighted_criterion_output():
    class _Criterion:
        weight_dict = {
            "loss_ce": 2.0,
            "loss_ce_0": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        }

        def __call__(self, predictions, targets):
            assert predictions["pred_logits"].shape == (1, 4, 3)
            assert targets[0]["labels"].tolist() == [1]
            return {
                "loss_ce": torch.tensor(1.0),
                "loss_ce_0": torch.tensor(2.0),
                "loss_bbox": torch.tensor(3.0),
                "loss_giou": torch.tensor(4.0),
            }

    adapter = object.__new__(RFDETRValidationLoss)
    adapter.device = torch.device("cpu")
    adapter.num_classes = 2
    adapter.criterion = _Criterion()
    predictions = {
        "pred_logits": torch.zeros(1, 4, 3),
        "pred_boxes": torch.zeros(1, 4, 4),
    }
    targets = torch.tensor([[[2.0, 2.0, 6.0, 6.0, 1.0]]])

    values = adapter(predictions, targets, image_size=(8, 8))

    assert float(values["loss"]) == pytest.approx(29.0)
    assert float(values["loss/ce"]) == pytest.approx(3.0)
    assert float(values["loss/bbox"]) == pytest.approx(3.0)
    assert float(values["loss/giou"]) == pytest.approx(4.0)


def _criterion(*, distributed_normalize: bool) -> SetCriterion:
    return SetCriterion(
        num_classes=2,
        matcher=None,
        weight_dict={},
        focal_alpha=0.25,
        losses=[],
        distributed_normalize=distributed_normalize,
    )


def test_rfdetr_rank_local_normalizer_skips_collective(monkeypatch):
    monkeypatch.setattr(
        rfdetr_loss_module, "is_dist_avail_and_initialized", lambda: True
    )
    monkeypatch.setattr(rfdetr_loss_module, "get_world_size", lambda: 4)

    def _unexpected_collective(value):
        del value
        raise AssertionError("rank-local validation entered a collective")

    monkeypatch.setattr(torch.distributed, "all_reduce", _unexpected_collective)
    criterion = _criterion(distributed_normalize=False)
    outputs = {"pred_logits": torch.zeros(1, 1, 3)}
    targets = [{"labels": torch.tensor([0, 1])}]

    assert criterion._box_count_normalizer(outputs, targets, 1) == pytest.approx(2.0)


def test_rfdetr_training_normalizer_keeps_global_average(monkeypatch):
    calls = []

    def _all_reduce(value):
        calls.append(True)
        value.add_(6.0)  # local 2 + other ranks 6 = global 8

    monkeypatch.setattr(
        rfdetr_loss_module, "is_dist_avail_and_initialized", lambda: True
    )
    monkeypatch.setattr(rfdetr_loss_module, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    criterion = _criterion(distributed_normalize=True)
    outputs = {"pred_logits": torch.zeros(1, 1, 3)}
    targets = [{"labels": torch.tensor([0, 1])}]

    assert criterion._box_count_normalizer(outputs, targets, 1) == pytest.approx(2.0)
    assert calls == [True]


def test_monitor_overlays_validation_loss_when_present():
    assert 'includes("metrics/loss")' in INDEX_HTML
    assert 'name: "val/loss"' in INDEX_HTML
    assert 'card("Val loss"' in INDEX_HTML


def test_experiment_loggers_emit_canonical_validation_loss_name():
    event = SimpleNamespace(
        train_loss=2.0,
        train_loss_items={},
        lr={},
        val_metrics={"metrics/loss": 1.5, "metrics/loss/box": 0.5},
        epoch_seconds=3.0,
    )

    metrics = epoch_metrics(event)

    assert metrics["val/loss"] == pytest.approx(1.5)
    assert metrics["val/loss/box"] == pytest.approx(0.5)
