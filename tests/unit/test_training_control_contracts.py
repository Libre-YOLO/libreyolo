from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch.nn as nn

from libreyolo.training.config import require_training_choice


pytestmark = pytest.mark.unit


def _trainer(module_name: str, class_name: str):
    return getattr(importlib.import_module(module_name), class_name)


@pytest.mark.parametrize(
    ("module_name", "class_name", "requested"),
    [
        ("libreyolo.models.yolox.trainer", "YOLOXTrainer", "constant"),
        ("libreyolo.models.yolo7.trainer", "YOLOv7Trainer", "constant"),
        ("libreyolo.models.convnext.trainer", "ConvNeXtTrainer", "constant"),
        (
            "libreyolo.models.efficientnetv2.trainer",
            "EfficientNetV2Trainer",
            "constant",
        ),
        (
            "libreyolo.models.mobilenetv4.trainer",
            "MobileNetV4Trainer",
            "constant",
        ),
        ("libreyolo.models.resnet.trainer", "ResNetTrainer", "constant"),
        ("libreyolo.models.nafnet.trainer", "NAFNetTrainer", "constant"),
        ("libreyolo.models.picodet.trainer", "PICODETTrainer", "constant"),
        ("libreyolo.models.rtmdet.trainer", "RTMDetTrainer", "constant"),
        ("libreyolo.models.yolonas.trainer", "YOLONASTrainer", "constant"),
        (
            "libreyolo.models.yolonas.pose_trainer",
            "YOLONASPoseTrainer",
            "constant",
        ),
        ("libreyolo.models.dfine.trainer", "DFINETrainer", "constant"),
        ("libreyolo.models.deim.trainer", "DEIMTrainer", "constant"),
        ("libreyolo.models.ec.seg_trainer", "ECSegTrainer", "constant"),
        ("libreyolo.models.ec.pose_trainer", "ECPoseTrainer", "constant"),
        ("libreyolo.models.fomo.trainer", "FOMOTrainer", "typo"),
        ("libreyolo.models.rfdetr.trainer", "RFDETRTrainer", "typo"),
        ("libreyolo.models.segformer.trainer", "SegformerTrainer", "typo"),
    ],
)
def test_family_scheduler_rejects_unimplemented_choice(
    module_name, class_name, requested
):
    trainer_cls = _trainer(module_name, class_name)
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.config = trainer_cls._config_class()(scheduler=requested)

    with pytest.raises(ValueError, match="does not support scheduler"):
        trainer.create_scheduler(iters_per_epoch=2)


@pytest.mark.parametrize(
    ("module_name", "class_name", "scheduler_class"),
    [
        (
            "libreyolo.models.yolox.trainer",
            "YOLOXTrainer",
            "WarmupCosineScheduler",
        ),
        (
            "libreyolo.models.yolo7.trainer",
            "YOLOv7Trainer",
            "WarmupCosineScheduler",
        ),
        (
            "libreyolo.models.convnext.trainer",
            "ConvNeXtTrainer",
            "WarmupCosineScheduler",
        ),
        (
            "libreyolo.models.efficientnetv2.trainer",
            "EfficientNetV2Trainer",
            "WarmupCosineScheduler",
        ),
        (
            "libreyolo.models.mobilenetv4.trainer",
            "MobileNetV4Trainer",
            "WarmupCosineScheduler",
        ),
        (
            "libreyolo.models.resnet.trainer",
            "ResNetTrainer",
            "WarmupCosineScheduler",
        ),
        (
            "libreyolo.models.nafnet.trainer",
            "NAFNetTrainer",
            "WarmupCosineScheduler",
        ),
        (
            "libreyolo.models.picodet.trainer",
            "PICODETTrainer",
            "WarmupCosineScheduler",
        ),
        (
            "libreyolo.models.rtmdet.trainer",
            "RTMDetTrainer",
            "WarmupCosineScheduler",
        ),
        (
            "libreyolo.models.yolonas.trainer",
            "YOLONASTrainer",
            "CosineAnnealingScheduler",
        ),
        (
            "libreyolo.models.yolonas.pose_trainer",
            "YOLONASPoseTrainer",
            "CosineAnnealingScheduler",
        ),
        (
            "libreyolo.models.dfine.trainer",
            "DFINETrainer",
            "FlatCosineScheduler",
        ),
        (
            "libreyolo.models.deim.trainer",
            "DEIMTrainer",
            "FlatCosineScheduler",
        ),
        (
            "libreyolo.models.ec.seg_trainer",
            "ECSegTrainer",
            "FlatCosineScheduler",
        ),
        (
            "libreyolo.models.ec.pose_trainer",
            "ECPoseTrainer",
            "FlatCosineScheduler",
        ),
        (
            "libreyolo.models.fomo.trainer",
            "FOMOTrainer",
            "CosineAnnealingScheduler",
        ),
        (
            "libreyolo.models.rfdetr.trainer",
            "RFDETRTrainer",
            "RFDETRStepScheduler",
        ),
        (
            "libreyolo.models.segformer.trainer",
            "SegformerTrainer",
            "LinearLRScheduler",
        ),
    ],
)
def test_family_scheduler_keeps_recipe_default(
    module_name, class_name, scheduler_class
):
    trainer_cls = _trainer(module_name, class_name)
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.config = trainer_cls._config_class()()

    scheduler = trainer.create_scheduler(iters_per_epoch=2)

    assert type(scheduler).__name__ == scheduler_class


@pytest.mark.parametrize(
    ("module_name", "class_name", "requested", "scheduler_class"),
    [
        (
            "libreyolo.models.fomo.trainer",
            "FOMOTrainer",
            "constant",
            "ConstantLRScheduler",
        ),
        (
            "libreyolo.models.rfdetr.trainer",
            "RFDETRTrainer",
            "cosine",
            "CosineAnnealingScheduler",
        ),
        (
            "libreyolo.models.rfdetr.trainer",
            "RFDETRTrainer",
            "flat_cosine",
            "FlatCosineScheduler",
        ),
    ],
)
def test_family_scheduler_honors_implemented_nondefault(
    module_name, class_name, requested, scheduler_class
):
    trainer_cls = _trainer(module_name, class_name)
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.config = trainer_cls._config_class()(scheduler=requested)

    scheduler = trainer.create_scheduler(iters_per_epoch=2)

    assert type(scheduler).__name__ == scheduler_class


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("libreyolo.models.dfine.trainer", "DFINETrainer"),
        ("libreyolo.models.deim.trainer", "DEIMTrainer"),
        ("libreyolo.models.ec.trainer", "ECTrainer"),
        ("libreyolo.models.ec.seg_trainer", "ECSegTrainer"),
        ("libreyolo.models.ec.pose_trainer", "ECPoseTrainer"),
        ("libreyolo.models.rtdetrv4.trainer", "RTDETRv4Trainer"),
        ("libreyolo.models.rfdetr.trainer", "RFDETRTrainer"),
        ("libreyolo.models.dinov2.trainer", "DINOv2Trainer"),
        ("libreyolo.models.segformer.trainer", "SegformerTrainer"),
    ],
)
def test_recipe_specific_optimizer_rejects_unimplemented_choice(
    module_name, class_name
):
    trainer_cls = _trainer(module_name, class_name)
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.config = trainer_cls._config_class()(optimizer="sgd")

    with pytest.raises(ValueError, match="does not support optimizer"):
        trainer._setup_optimizer()


class _GroupedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("libreyolo.models.dfine.trainer", "DFINETrainer"),
        ("libreyolo.models.deim.trainer", "DEIMTrainer"),
        ("libreyolo.models.ec.seg_trainer", "ECSegTrainer"),
        ("libreyolo.models.ec.pose_trainer", "ECPoseTrainer"),
    ],
)
def test_detr_recipe_optimizer_honors_configured_beta1(module_name, class_name):
    trainer_cls = _trainer(module_name, class_name)
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.config = trainer_cls._config_class()(momentum=0.42)
    trainer.model = _GroupedModel()

    optimizer = trainer._setup_optimizer()

    assert optimizer.defaults["betas"] == pytest.approx((0.42, 0.999))


def test_rfdetr_optimizer_honors_configured_beta1():
    trainer_cls = _trainer("libreyolo.models.rfdetr.trainer", "RFDETRTrainer")
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.config = trainer_cls._config_class()(momentum=0.42)
    trainer.model = _GroupedModel()
    trainer.wrapper_model = SimpleNamespace(task="classify")

    optimizer = trainer._setup_optimizer()

    assert optimizer.defaults["betas"] == pytest.approx((0.42, 0.999))


def test_segformer_optimizer_honors_configured_beta1():
    trainer_cls = _trainer("libreyolo.models.segformer.trainer", "SegformerTrainer")
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.config = trainer_cls._config_class()(momentum=0.42)
    trainer.model = _GroupedModel()

    optimizer = trainer._setup_optimizer()

    assert optimizer.defaults["betas"] == pytest.approx((0.42, 0.999))


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("libreyolo.training.config", "DFINEConfig"),
        ("libreyolo.training.config", "DEIMConfig"),
        ("libreyolo.training.config", "DEIMv2Config"),
        ("libreyolo.training.config", "ECConfig"),
        ("libreyolo.training.config", "SegformerConfig"),
        ("libreyolo.training.config", "FOMOConfig"),
        ("libreyolo.models.convnext.config", "ConvNeXtConfig"),
        ("libreyolo.models.efficientnetv2.config", "EfficientNetV2Config"),
        ("libreyolo.models.mobilenetv4.config", "MobileNetV4Config"),
        ("libreyolo.models.resnet.config", "ResNetConfig"),
        ("libreyolo.models.nafnet.config", "NAFNetConfig"),
        ("libreyolo.models.rfdetr.config", "RFDETRConfig"),
    ],
)
def test_adam_family_defaults_keep_beta1_point_nine(module_name, class_name):
    config_cls = getattr(importlib.import_module(module_name), class_name)
    assert config_cls().momentum == pytest.approx(0.9)


def test_training_choice_normalizes_case_and_whitespace():
    assert (
        require_training_choice(
            " Flat_Cosine ",
            field="scheduler",
            supported=("flat_cosine",),
            family="test",
        )
        == "flat_cosine"
    )
