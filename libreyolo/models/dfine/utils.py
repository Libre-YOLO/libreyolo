"""LibreDFINE preprocessing and postprocessing helpers.

Postprocessing lives in ``libreyolo.postprocess.dfine`` and is re-exported
here for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np
import torch
from PIL import Image

from ...postprocess.dfine import postprocess  # noqa: F401  (backward-compatible re-export)
from ...utils.image_loader import ImageInput, ImageLoader
from ...preprocess.dfine import (  # noqa: F401  (moved; re-exported for backward compatibility)
    preprocess_numpy,
)


def unwrap_dfine_checkpoint(checkpoint: Mapping | Any):
    """Extract the state_dict from a D-FINE checkpoint.

    Upstream saves ``{"ema": {"module": state_dict, ...}, "model": state_dict, ...}``.
    Prefer EMA when present (matches upstream ``tools/inference_torch.py``).
    """
    if not isinstance(checkpoint, Mapping):
        return checkpoint

    ema = checkpoint.get("ema")
    if isinstance(ema, Mapping):
        module = ema.get("module")
        if isinstance(module, Mapping):
            return module

    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value

    return checkpoint




def preprocess_image(
    image: ImageInput,
    input_size: int = 640,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()

    img_chw, ratio = preprocess_numpy(np.array(img), input_size=input_size)
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
    return img_tensor, original_img, original_size, ratio
