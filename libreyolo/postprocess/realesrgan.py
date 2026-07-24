"""Postprocessing for LibreRealESRGAN super-resolution outputs.

The network output canvas is ``scale`` times the (reflect-padded) input. This
crops it back to ``scale`` times the *original* input size, clamps to [0, 1],
and quantizes to HWC uint8 RGB - the same quantization Real-ESRGAN uses.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch


def _as_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        output = output.get("restored", output.get("predictions", output.get("output")))
    if isinstance(output, (list, tuple)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        output = torch.as_tensor(output)
    return output


def postprocess(
    output: Any,
    original_size: Tuple[int, int],
    scale: int = 1,
) -> np.ndarray:
    """Crop to ``scale`` x original, clamp, and convert to HWC uint8 RGB.

    Args:
        output: model output tensor ``[B, 3, scale*Hp, scale*Wp]`` (padded).
        original_size: original input ``(W, H)`` before divisibility padding.
        scale: integer upscale factor of the model.
    """

    restored = _as_tensor(output).detach().float().cpu()
    if restored.ndim == 3:
        restored = restored.unsqueeze(0)
    if restored.ndim != 4 or restored.shape[1] != 3:
        raise ValueError(
            "Real-ESRGAN postprocess expects output shape [B, 3, H, W], "
            f"got {tuple(restored.shape)}."
        )
    orig_w, orig_h = original_size
    out_h = int(orig_h) * int(scale)
    out_w = int(orig_w) * int(scale)
    restored = restored[0, :, :out_h, :out_w].clamp_(0.0, 1.0)
    restored = restored.permute(1, 2, 0).mul(255.0).round().to(torch.uint8)
    return restored.numpy()


__all__ = ["postprocess"]
