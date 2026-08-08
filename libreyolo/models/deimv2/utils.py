"""LibreDEIMv2 preprocessing and postprocessing helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...postprocess.deim import postprocess
from ...utils.image_loader import ImageInput, ImageLoader
from ..deim.utils import unwrap_deim_checkpoint
from ...preprocess.deimv2 import (  # noqa: F401  (moved; re-exported for backward compatibility)
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_numpy,
)


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "postprocess",
    "preprocess_image",
    "preprocess_numpy",
    "unwrap_deim_checkpoint",
]




def preprocess_image(
    image: ImageInput,
    input_size: int = 640,
    color_format: str = "auto",
    *,
    imagenet_norm: bool = False,
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()

    img_chw, ratio = preprocess_numpy(
        np.array(img), input_size=input_size, imagenet_norm=imagenet_norm
    )
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
    return img_tensor, original_img, original_size, ratio
