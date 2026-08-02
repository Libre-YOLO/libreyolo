"""Inference-side preprocessing and checkpoint helpers for LW-DETR.

Upstream's evaluation transform is
``Compose([SquareResize([640]), ToTensor(), Normalize(ImageNet)])`` with
``--square_resize_div_64`` (models/../datasets/coco.py::
``make_coco_transforms_square_div_64``). ``SquareResize`` is a torchvision
``F.resize`` on a PIL image to ``(size, size)`` — i.e. PIL BILINEAR, aspect
ratio *not* preserved, no letterbox padding. That is reproduced here.

Postprocessing lives in ``libreyolo.postprocess.lwdetr``.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 640,
) -> Tuple[np.ndarray, float]:
    """Resize an RGB HWC uint8 image to a square canvas and ImageNet-normalize.

    Args:
        img_rgb_hwc: RGB HWC uint8 image.
        input_size: Square side of the model canvas.

    Returns:
        ``(CHW float32 array, ratio)``. ``ratio`` is always 1.0: LW-DETR
        stretches to the square canvas rather than letterboxing, so boxes are
        rescaled by the original width/height directly and there is no single
        scalar ratio to undo.
    """
    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1), 1.0


def preprocess_image(
    image: ImageInput,
    input_size: int = 640,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load and preprocess one image for LW-DETR inference."""
    img = ImageLoader.load(image, color_format=color_format)
    orig_w, orig_h = img.size
    img_chw, ratio = preprocess_numpy(np.array(img), input_size)
    return torch.from_numpy(img_chw).unsqueeze(0), img, (orig_w, orig_h), ratio


def unwrap_lwdetr_checkpoint(checkpoint: Any) -> dict:
    """Return the raw parameter dict from a LibreYOLO or upstream checkpoint."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dict, got {type(checkpoint).__name__}")
    for key in ("model", "model_state_dict", "state_dict", "ema"):
        inner = checkpoint.get(key)
        if isinstance(inner, dict):
            return inner
    return checkpoint
