"""Training configuration for LibreNAFNet restoration fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...training.config import TrainConfig


@dataclass(kw_only=True)
class NAFNetConfig(TrainConfig):
    """NAFNet restoration fine-tuning defaults.

    ``imgsz`` is the train-patch crop size. Predict/val run at native image
    resolution and pad only to the network downsample factor.
    """

    size: str = "s"
    imgsz: int = 256
    epochs: int = 100
    batch: int = 16
    optimizer: str = "adamw"
    momentum: float = 0.9
    lr0: float = 1e-3
    weight_decay: float = 0.0
    warmup_epochs: int = 0
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.01
    workers: int = 8
    ema: bool = True
    amp: bool = True
    eval_interval: int = 1
    name: str = "nafnet_restore_exp"

    # Optional checkpoint provenance metadata.
    degradation: Optional[str] = None
    dataset: Optional[str] = None

