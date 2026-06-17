"""FOMO training dataset."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .utils import MEAN, STD


def _transform_image(pil_image: Image.Image, input_size: int) -> torch.Tensor:
    """Resize to input_size × input_size and normalize to [-1, 1] (RGB)."""
    img = pil_image.convert("RGB").resize((input_size, input_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32))


def _boxes_xyxy_to_grid(
    boxes_xyxy: np.ndarray,
    classes: np.ndarray,
    input_size: int,
    grid_size: int,
) -> torch.Tensor:
    """Encode xyxy-pixel boxes as a FOMO grid target tensor."""
    grid = torch.zeros((grid_size, grid_size), dtype=torch.long)
    for (x1, y1, x2, y2), cls in zip(boxes_xyxy, classes):
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        gx = int((cx / input_size) * grid_size)
        gy = int((cy / input_size) * grid_size)
        gx = min(max(gx, 0), grid_size - 1)
        gy = min(max(gy, 0), grid_size - 1)
        grid[gy, gx] = int(cls) + 1
    return grid


class FOMOYOLODataset(Dataset):
    """FOMO dataset backed by a YOLO-format image directory."""

    def __init__(
        self,
        img_files: list,
        label_files: list,
        input_size: int,
        grid_size: int,
    ) -> None:
        self.img_files = img_files
        self.label_files = label_files
        self.input_size = input_size
        self.grid_size = grid_size

    def __len__(self) -> int:
        return len(self.img_files)

    def _load_labels(self, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (boxes_xyxy_pixel, classes_int) or empty arrays."""
        try:
            with open(label_path) as f:
                lines = f.read().strip().split("\n")
        except (FileNotFoundError, OSError):
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        boxes, classes = [], []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w / 2) * self.input_size
            y1 = (cy - h / 2) * self.input_size
            x2 = (cx + w / 2) * self.input_size
            y2 = (cy + h / 2) * self.input_size
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)
        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)

    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        try:
            pil_img = Image.open(img_path)
        except (FileNotFoundError, OSError) as e:
            raise FileNotFoundError(f"Cannot read image: {img_path}") from e

        img_tensor = _transform_image(pil_img, self.input_size)

        label_path = self.label_files[idx] if idx < len(self.label_files) else ""
        boxes_xyxy, classes = self._load_labels(label_path)
        grid = _boxes_xyxy_to_grid(boxes_xyxy, classes, self.input_size, self.grid_size)

        img_info = (self.input_size, self.input_size)
        return img_tensor, grid, img_info, idx


def _boxes_cxcy_to_grid(
    boxes_cxcy: np.ndarray,
    classes: np.ndarray,
    input_size: int,
    grid_size: int,
) -> torch.Tensor:
    """Encode cxcy-pixel box centers as a FOMO grid target tensor."""
    grid = torch.zeros((grid_size, grid_size), dtype=torch.long)
    for (cx, cy), cls in zip(boxes_cxcy, classes):
        gx = int((cx / input_size) * grid_size)
        gy = int((cy / input_size) * grid_size)
        gx = min(max(gx, 0), grid_size - 1)
        gy = min(max(gy, 0), grid_size - 1)
        grid[gy, gx] = int(cls) + 1
    return grid


class FOMOAugmentedDataset(Dataset):
    """Dataset wrapper for FOMO that consumes YOLOX-style augmented targets."""

    def __init__(
        self,
        augmented_dataset: Dataset,
        input_size: int,
        grid_size: int,
    ) -> None:
        self.dataset = augmented_dataset
        self.input_size = input_size
        self.grid_size = grid_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, targets, img_info, img_id = self.dataset[idx]

        img_rgb = img[::-1, :, :]  # BGR -> RGB
        img_normalized = (img_rgb / 255.0 - MEAN[:, None, None]) / STD[:, None, None]
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_normalized, dtype=np.float32))

        valid_mask = targets[:, 3] > 0
        valid_targets = targets[valid_mask]

        if len(valid_targets) > 0:
            classes = valid_targets[:, 0].astype(np.int64)
            boxes_cxcy = valid_targets[:, 1:3]
            grid = _boxes_cxcy_to_grid(boxes_cxcy, classes, self.input_size, self.grid_size)
        else:
            grid = torch.zeros((self.grid_size, self.grid_size), dtype=torch.long)

        return img_tensor, grid, img_info, img_id

    def close_mosaic(self) -> None:
        if hasattr(self.dataset, "close_mosaic"):
            self.dataset.close_mosaic()
