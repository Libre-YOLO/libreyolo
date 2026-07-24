"""Utility helpers for LibreRealESRGAN: preprocessing and seam-free tiling.

The tiled forward pass is ported from the Real-ESRGAN inference helper
(https://github.com/xinntao/Real-ESRGAN, BSD-3-Clause). Each tile is processed
with a padding halo and only its interior (unpadded) region is written to the
output canvas, so tile boundaries are seam-free.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader


def _reflect_pad_to_multiple(tensor: torch.Tensor, multiple: int) -> torch.Tensor:
    """Reflect-pad the bottom/right so H and W are divisible by ``multiple``."""

    if multiple <= 1:
        return tensor
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if not pad_h and not pad_w:
        return tensor
    mode = "reflect" if h > 1 and w > 1 and pad_h < h and pad_w < w else "replicate"
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)


def preprocess_image(
    image: ImageInput,
    *,
    pad_multiple: int,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load an RGB image as a float [0, 1] tensor, reflect-padded for divisibility.

    Real-ESRGAN processes at native resolution. The x2 (RRDBNet scale 2) variant
    pixel-unshuffles by 2 and therefore needs even spatial dims; the x4 and x4t
    variants have no divisibility constraint (``pad_multiple == 1``).
    """

    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size  # (W, H)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    tensor = _reflect_pad_to_multiple(tensor, int(pad_multiple))
    return tensor, img, original_size, 1.0


def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size: int | tuple[int, int]):
    """Exporter calibration helper: convert RGB HWC to CHW float [0, 1]."""

    del input_size
    arr = np.asarray(img_rgb_hwc, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr.transpose(2, 0, 1).astype(np.float32), 1.0


def tiled_forward(
    model: nn.Module,
    img: torch.Tensor,
    *,
    scale: int,
    tile_size: int,
    tile_pad: int = 10,
) -> torch.Tensor:
    """Run ``model`` tile-by-tile and stitch a seam-free upscaled output.

    Ported from Real-ESRGAN's ``RealESRGANer.tile_process`` (BSD-3-Clause).
    ``img`` is ``[B, C, H, W]``; the result is ``[B, C, scale*H, scale*W]``.
    Each tile is padded by ``tile_pad`` on every side for context and only the
    central (unpadded) region is copied into the output, which removes seams.
    """

    batch, channel, height, width = img.shape
    output_height = height * scale
    output_width = width * scale
    output_shape = (batch, channel, output_height, output_width)

    # start with a black canvas
    output = img.new_zeros(output_shape)
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    for y in range(tiles_y):
        for x in range(tiles_x):
            ofs_x = x * tile_size
            ofs_y = y * tile_size
            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            # input tile area on total image with padding halo
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y
            input_tile = img[
                :,
                :,
                input_start_y_pad:input_end_y_pad,
                input_start_x_pad:input_end_x_pad,
            ]

            output_tile = model(input_tile)

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without the padding halo
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            output[
                :, :, output_start_y:output_end_y, output_start_x:output_end_x
            ] = output_tile[
                :,
                :,
                output_start_y_tile:output_end_y_tile,
                output_start_x_tile:output_end_x_tile,
            ]
    return output


def forward_with_optional_tiling(
    model: nn.Module,
    img: torch.Tensor,
    *,
    scale: int,
    tile_size: int,
    tile_pad: int = 10,
) -> torch.Tensor:
    """Forward ``img`` through ``model``, tiling when ``tile_size > 0``.

    On a CUDA out-of-memory error the message suggests using tiles.
    """

    try:
        if tile_size and tile_size > 0:
            return tiled_forward(
                model, img, scale=scale, tile_size=tile_size, tile_pad=tile_pad
            )
        return model(img)
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware dependent
        torch.cuda.empty_cache()
        raise torch.cuda.OutOfMemoryError(
            "Real-ESRGAN ran out of GPU memory on a "
            f"{tuple(img.shape[-2:])} input. Retry with tiling, e.g. "
            "model.predict(image, tile=512) (optionally tile_pad=10), "
            "or use a smaller image / the faster x4t model."
        ) from exc
    except RuntimeError as exc:  # pragma: no cover - hardware dependent
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "Real-ESRGAN ran out of memory on a "
                f"{tuple(img.shape[-2:])} input. Retry with tiling, e.g. "
                "model.predict(image, tile=512) (optionally tile_pad=10), "
                "or use a smaller image / the faster x4t model."
            ) from exc
        raise


__all__ = [
    "forward_with_optional_tiling",
    "preprocess_image",
    "preprocess_numpy",
    "tiled_forward",
]
