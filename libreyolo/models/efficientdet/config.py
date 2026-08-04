"""Model-scale configuration for the EfficientDet museum family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EfficientDetScale:
    """Architecture settings from ``effdet`` 0.4.1's TF checkpoint configs."""

    image_size: int
    backbone_size: str
    channel_multiplier: float
    depth_multiplier: float
    fpn_channels: int
    fpn_repeats: int
    head_repeats: int


SCALE_CONFIGS = {
    "d0": EfficientDetScale(512, "b0", 1.0, 1.0, 64, 3, 3),
    "d1": EfficientDetScale(640, "b1", 1.0, 1.1, 88, 4, 3),
    "d2": EfficientDetScale(768, "b2", 1.1, 1.2, 112, 5, 3),
    "d3": EfficientDetScale(896, "b3", 1.2, 1.4, 160, 6, 4),
    "d4": EfficientDetScale(1024, "b4", 1.4, 1.8, 224, 7, 4),
}

INPUT_SIZES = {size: cfg.image_size for size, cfg in SCALE_CONFIGS.items()}

__all__ = ["EfficientDetScale", "INPUT_SIZES", "SCALE_CONFIGS"]
