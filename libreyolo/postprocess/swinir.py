"""Postprocessing for LibreSwinIR super-resolution outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .realesrgan import postprocess as _postprocess_super_resolution


def postprocess(
    output: Any, original_size: tuple[int, int], scale: int = 4
) -> np.ndarray:
    """Crop padding and quantize an upscaled SwinIR RGB tensor to uint8."""

    return _postprocess_super_resolution(output, original_size, scale=scale)


__all__ = ["postprocess"]
