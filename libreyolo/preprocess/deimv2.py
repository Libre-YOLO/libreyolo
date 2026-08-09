"""DEIMv2 input preprocessing.

Moved verbatim from ``libreyolo/models/deimv2/utils.py``, which re-exports it for backward
compatibility. Lives outside ``models/`` so the ONNX backend can import it
without pulling torch; see the package docstring.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

# Sizes whose DINO backbone expects ImageNet-normalised input. Lives here
# rather than in models/deimv2/nn.py (which re-exports it) so the backend can
# read it without importing the network definition, and therefore torch.
DINO_SIZES = {"s", "m", "l", "x"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 640,
    *,
    imagenet_norm: bool = False,
) -> Tuple[np.ndarray, float]:
    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    chw = (np.array(img_resized, dtype=np.float32) / 255.0).transpose(2, 0, 1)
    if imagenet_norm:
        chw = (chw - IMAGENET_MEAN) / IMAGENET_STD
    return chw.astype(np.float32), 1.0


__all__ = ["DINO_SIZES", "IMAGENET_MEAN", "IMAGENET_STD", "preprocess_numpy"]
