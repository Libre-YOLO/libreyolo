"""Preprocessing and decoding helpers for LibrePAGE gaze-target inference.

The preprocessing mirrors the upstream PaGE HF image processor exactly:
PIL bilinear resize (scene 512x512, head crops 256x256), scale to [0, 1],
ImageNet mean/std normalization, RGB channel order. Head rects are the
normalized head boxes projected onto the 32x32 scene patch grid and
rounded, matching upstream ``get_input_head_maps``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .nn import HEAD_SIZE, IMAGE_MEAN, IMAGE_STD, SCENE_SIZE

SCENE_GRID = (SCENE_SIZE[0] // 16, SCENE_SIZE[1] // 16)  # (32, 32)


def pil_to_tensor(pil_img: Image.Image, size: Tuple[int, int]) -> torch.Tensor:
    """Resize + normalize a PIL image to a [3, H, W] float tensor."""
    pil_img = pil_img.convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGE_MEAN, dtype=np.float32)) / np.array(
        IMAGE_STD, dtype=np.float32
    )
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))


def clamp_pixel_box(
    box: Sequence[float], width: int, height: int
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    """Clamp an xyxy pixel box to the image; return (pixel_box, normalized_box)."""
    x1 = max(0.0, min(float(width - 1), float(box[0])))
    y1 = max(0.0, min(float(height - 1), float(box[1])))
    x2 = max(0.0, min(float(width), float(box[2])))
    y2 = max(0.0, min(float(height), float(box[3])))
    pixel = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
    if pixel[2] <= pixel[0] or pixel[3] <= pixel[1]:
        raise ValueError(f"head box is empty or outside the image: {box!r}")
    normalized = (pixel[0] / width, pixel[1] / height, pixel[2] / width, pixel[3] / height)
    return pixel, normalized


def expand_box(
    box: Sequence[float], width: int, height: int, factor: float
) -> Tuple[float, float, float, float]:
    """Symmetrically scale an xyxy box about its center by ``factor``."""
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) * factor / 2.0, (y2 - y1) * factor / 2.0
    return (
        max(0.0, cx - hw),
        max(0.0, cy - hh),
        min(float(width), cx + hw),
        min(float(height), cy + hh),
    )


def head_rects_grid(norm_boxes: Sequence[Sequence[float]]) -> torch.Tensor:
    """Project normalized head boxes onto the scene grid, upstream-style.

    Returns [N, 4] float tensor (ymin, xmin, ymax, xmax) in grid units,
    rounded exactly as upstream ``get_input_head_maps`` does.
    """
    grid_h, grid_w = SCENE_GRID
    rects = []
    for xmin, ymin, xmax, ymax in norm_boxes:
        rects.append(
            [
                float(round(ymin * grid_h)),
                float(round(xmin * grid_w)),
                float(round(ymax * grid_h)),
                float(round(xmax * grid_w)),
            ]
        )
    return torch.tensor(rects, dtype=torch.float32)


def preprocess_scene_and_heads(
    pil_img: Image.Image,
    pixel_boxes: Sequence[Tuple[int, int, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the (scene, heads) input tensors from an image and pixel boxes."""
    scene = pil_to_tensor(pil_img, SCENE_SIZE).unsqueeze(0)
    crops = [pil_img.crop(box) for box in pixel_boxes]
    heads = torch.stack([pil_to_tensor(c, HEAD_SIZE) for c in crops], dim=0)
    return scene, heads


def decode_heatmaps(
    heatmap_probs: torch.Tensor, orig_w: int, orig_h: int
) -> torch.Tensor:
    """Decode gaze points from [N, H, W] sigmoid heatmaps.

    The argmax cell's center is mapped onto the original canvas by plain
    scaling, matching the upstream inference script:
    ``x = (col + 0.5) / W * img_w``.
    """
    n, hm_h, hm_w = heatmap_probs.shape
    flat = heatmap_probs.reshape(n, -1)
    idx = flat.argmax(dim=1)
    rows = (idx // hm_w).to(torch.float32)
    cols = (idx % hm_w).to(torch.float32)
    x = (cols + 0.5) / hm_w * orig_w
    y = (rows + 0.5) / hm_h * orig_h
    return torch.stack([x, y], dim=1)


def crop_boxes_from_faces(
    faces: List,
    width: int,
    height: int,
    expand: float,
) -> List[Tuple[int, int, int, int]]:
    """Expand detector face boxes into head boxes and clamp to the image."""
    out = []
    for f in faces:
        expanded = expand_box(f.xyxy, width, height, expand)
        pixel, _ = clamp_pixel_box(expanded, width, height)
        out.append(pixel)
    return out
