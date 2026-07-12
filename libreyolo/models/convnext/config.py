"""Training configuration for LibreConvNeXt classification fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

from ...training.config import TrainConfig


@dataclass(kw_only=True)
class ConvNeXtConfig(TrainConfig):
    """Classification fine-tuning defaults.

    AdamW + light warmup + cosine is a forgiving recipe for transferring an
    ImageNet-pretrained backbone to a custom class set. The image pipeline is the
    classification ImageFolder path (RandomResizedCrop + flip), so the detection
    mosaic/mixup fields are unused here.
    """

    size: str = "t"
    imgsz: int = 224
    epochs: int = 100
    batch: int = 64
    optimizer: str = "adamw"
    momentum: float = 0.9
    lr0: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.01
    workers: int = 8
    ema: bool = True
    # Classification validation is cheap (a few seconds over the val split), so
    # validate every epoch rather than inheriting the detection default of 10.
    # Short fine-tunes (the documented ``epochs=5`` example) then still report
    # top-1/top-5 each epoch and save ``best.pt``.
    eval_interval: int = 1
