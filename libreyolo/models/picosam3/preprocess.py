"""PicoSAM3 ROI geometry and ImageNet preprocessing."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


def padded_square_roi(
    box: Iterable[float], image_width: int, image_height: int, padding: float = 0.1
) -> tuple[int, int, int, int]:
    """Convert an xyxy box to upstream's padded, square-clamped crop."""

    x1, y1, x2, y2 = (float(v) for v in box)
    if not all(np.isfinite((x1, y1, x2, y2))):
        raise ValueError(f"bbox coordinates must be finite; got {list(box)!r}.")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox must satisfy x2>x1 and y2>y1; got {list(box)!r}.")
    if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
        raise ValueError(f"bbox {list(box)!r} does not intersect the image.")

    width, height = x2 - x1, y2 - y1
    x1 -= width * padding
    y1 -= height * padding
    width *= 1.0 + 2.0 * padding
    height *= 1.0 + 2.0 * padding
    size = max(width, height)
    cx, cy = x1 + width / 2.0, y1 + height / 2.0
    crop = (
        max(0, int(cx - size / 2.0)),
        max(0, int(cy - size / 2.0)),
        min(image_width, int(cx + size / 2.0)),
        min(image_height, int(cy + size / 2.0)),
    )
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise ValueError(f"bbox {list(box)!r} produces an empty ROI after clipping.")
    return crop


def preprocess_roi(image: Image.Image, roi: tuple[int, int, int, int]) -> torch.Tensor:
    """Crop, bilinear-resize to 96, convert RGB to CHW, and normalize."""

    crop = image.crop(roi).resize((96, 96), Image.Resampling.BILINEAR)
    array = np.asarray(crop, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - _MEAN) / _STD


def place_roi_logits(
    logits: torch.Tensor,
    rois: list[tuple[int, int, int, int]],
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize ROI logits into full-image boolean masks and certainty scores."""

    masks = torch.zeros(
        (len(rois), image_height, image_width), dtype=torch.bool, device="cpu"
    )
    scores = torch.zeros(len(rois), dtype=torch.float32)
    for index, (logit, (x1, y1, x2, y2)) in enumerate(zip(logits, rois)):
        resized = (
            F.interpolate(
                logit.unsqueeze(0),
                size=(y2 - y1, x2 - x1),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            .detach()
            .cpu()
        )
        roi_mask = resized > 0.0
        masks[index, y1:y2, x1:x2] = roi_mask
        if roi_mask.any():
            scores[index] = resized.sigmoid()[roi_mask].mean()
    return masks, scores


__all__ = ["padded_square_roi", "place_roi_logits", "preprocess_roi"]
