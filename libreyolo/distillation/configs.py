"""
Distillation configuration helpers.

Delegates to each model wrapper's ``get_distill_config()`` method — the
models themselves are the source of truth for their tap points, channel
dimensions, and strides.

Prefer calling ``model.get_distill_config()`` directly when you have a
model instance. This module is useful when you only have a family + size
string (e.g., from a CLI or config file).
"""

from __future__ import annotations

from typing import Dict, List


def get_distill_config(family: str, size: str) -> Dict:
    """Get the distillation config for a given model family and size.

    This is a convenience wrapper that instantiates a lightweight model
    wrapper and calls its ``get_distill_config()`` method. When you already
    have a model instance, call ``model.get_distill_config()`` directly.

    Args:
        family: Model family string (e.g., "yolo9", "yolox", "rfdetr").
        size: Model size string (e.g., "t", "s", "m", "c", "l", "x"). Sizes
            are validated against the family's detect task; for sizes that
            only exist for another task, instantiate the model with that
            task and call ``model.get_distill_config()`` directly.

    Returns:
        Dict with keys:
            - tap_points: List[str] — module paths for forward hooks
            - channels: List[int] — channel dimensions per tap point
            - strides: List[int] — spatial strides per tap point

    Raises:
        ValueError: If the family/size combination is not supported.
    """
    if family == "yolo9":
        from ..models.yolo9.model import LibreYOLO9

        model = LibreYOLO9(model_path=None, size=size)
        return model.get_distill_config()

    elif family == "yolox":
        from ..models.yolox.model import LibreYOLOX

        model = LibreYOLOX(model_path=None, size=size)
        return model.get_distill_config()

    elif family == "rfdetr":
        from ..models.rfdetr.model import LibreRFDETR

        # model_path={} is the build-without-weights sentinel: model_path=None
        # would resolve and download the default pretrained checkpoint, and
        # this helper must stay lightweight and offline-safe. The probe that
        # measures tap shapes only needs the architecture, not the weights.
        model = LibreRFDETR(model_path={}, size=size)
        return model.get_distill_config()

    else:
        raise ValueError(
            f"Distillation not yet configured for family '{family}'. "
            f"Supported: {list_supported()}. "
            f"To add support, implement get_distill_config() on your model class."
        )


def list_supported() -> List[str]:
    """Return list of model families with distillation support."""
    return ["yolo9", "yolox", "rfdetr"]
