"""Native MobileSAM preprocessing and mask upscaling."""

from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image


class ResizeLongestSide:
    """Resize image and spatial prompts so the longest side hits target size."""

    def __init__(self, target_length: int) -> None:
        self.target_length = int(target_length)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(
        self, coords: np.ndarray, original_size: Tuple[int, int]
    ) -> np.ndarray:
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(old_h, old_w, self.target_length)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, int]):
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        return int(newh + 0.5), int(neww + 0.5)


def encode_image_and_prompts(
    img: Image.Image,
    *,
    target_length: int,
    points=None,
    labels=None,
    boxes=None,
) -> dict[str, torch.Tensor]:
    """Return LibreSAM-style tensors for a MobileSAM image and prompts."""

    original_size = (img.height, img.width)
    transform = ResizeLongestSide(target_length)
    resized = transform.apply_image(np.asarray(img.convert("RGB")))
    image_tensor = torch.as_tensor(resized, dtype=torch.float32).permute(2, 0, 1)

    enc: dict[str, torch.Tensor] = {
        "pixel_values": image_tensor.unsqueeze(0),
        "original_sizes": torch.tensor([original_size], dtype=torch.long),
        "reshaped_input_sizes": torch.tensor(
            [[resized.shape[0], resized.shape[1]]],
            dtype=torch.long,
        ),
    }
    if points is not None:
        point_arr = np.asarray(points, dtype=np.float32)
        point_arr = transform.apply_coords(point_arr, original_size)
        enc["input_points"] = torch.as_tensor(point_arr, dtype=torch.float32).unsqueeze(
            0
        )
        enc["input_labels"] = torch.as_tensor(labels, dtype=torch.long).unsqueeze(0)
    if boxes is not None:
        box_arr = np.asarray(boxes, dtype=np.float32)
        box_arr = transform.apply_boxes(box_arr, original_size)
        enc["input_boxes"] = torch.as_tensor(box_arr, dtype=torch.float32).unsqueeze(0)
    return enc


def preprocess_tensor(
    x: torch.Tensor,
    *,
    image_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    """Normalize and pad resized image tensors to MobileSAM's square frame."""

    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = image_size - h
    padw = image_size - w
    return F.pad(x, (0, padw, 0, padh))


def postprocess_masks(
    masks: torch.Tensor,
    *,
    image_size: int,
    input_size: Tuple[int, int],
    original_size: Tuple[int, int],
    mask_threshold: float = 0.0,
) -> torch.Tensor:
    """Upscale low-res mask logits to original image size and binarize."""

    masks = F.interpolate(
        masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks > mask_threshold
