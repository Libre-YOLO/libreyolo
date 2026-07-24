"""Preprocessing and tiled inference helpers for LibreSwinIR."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..realesrgan.utils import preprocess_image, preprocess_numpy, tiled_forward


def forward_with_optional_tiling(
    model: nn.Module,
    image: torch.Tensor,
    *,
    scale: int,
    tile_size: int,
    tile_pad: int = 16,
) -> torch.Tensor:
    """Run a full-frame or halo-padded tiled SwinIR forward pass."""

    try:
        if tile_size > 0:
            return tiled_forward(
                model,
                image,
                scale=scale,
                tile_size=tile_size,
                tile_pad=tile_pad,
            )
        return model(image)
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware dependent
        torch.cuda.empty_cache()
        raise torch.cuda.OutOfMemoryError(
            "SwinIR ran out of GPU memory. Retry with tiling, for example "
            "model.predict(image, tile=256, tile_pad=16), or use size='s'."
        ) from exc
    except RuntimeError as exc:  # pragma: no cover - hardware dependent
        if "out of memory" not in str(exc).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(
            "SwinIR ran out of memory. Retry with tiling, for example "
            "model.predict(image, tile=256, tile_pad=16), or use size='s'."
        ) from exc


__all__ = ["forward_with_optional_tiling", "preprocess_image", "preprocess_numpy"]
