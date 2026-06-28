"""Training configuration for LibreEfficientNetV2 classification fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

from ...training.config import TrainConfig


@dataclass(kw_only=True)
class EfficientNetV2Config(TrainConfig):
    """Classification fine-tuning defaults.

    AdamW + light warmup + cosine is a forgiving recipe for transferring an
    ImageNet-pretrained backbone to a custom class set. The image pipeline is the
    classification ImageFolder path (RandomResizedCrop + flip), so the detection
    mosaic/mixup fields are unused here.
    """

    size: str = "b0"
    imgsz: int = 0  # 0 = derive the per-variant eval resolution in __post_init__
    epochs: int = 100
    batch: int = 64
    optimizer: str = "adamw"
    lr0: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.01
    workers: int = 8
    ema: bool = True

    def __post_init__(self) -> None:
        # EfficientNetV2-base evaluates each variant at a different resolution;
        # derive it from ``size`` unless the caller set ``imgsz`` explicitly.
        if not self.imgsz:
            self.imgsz = {"b0": 224, "b1": 240, "b2": 260, "b3": 300}.get(self.size, 224)
