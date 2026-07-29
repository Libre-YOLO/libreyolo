"""Surface-normal dataset for LibreYOLO.

Normal datasets pair each RGB image with a three-channel 16-bit PNG:

    dataset/
        images/train/*.jpg
        images/val/*.jpg
        normals/train/*.png    # same stem as the paired image
        normals/val/*.png
        masks/train/*.png      # optional; nonzero pixels are valid
        masks/val/*.png

PNG channels are RGB and decode as ``normal = png / 65535 * 2 - 1``. Vectors
are renormalized after decoding and after every bilinear resize. Returned
targets use shape ``[3, H, W]``; invalid and padded pixels are all zero.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import get_img_files, load_data_config

_PAD_COLOR = 114
_NORMAL_SCALE = 65535.0
_NORM_EPS = 1e-12


def _replace_images_dir(path: Path, target_dir: str) -> Path:
    path_str = str(path)
    for separator in (os.sep, "/", "\\"):
        path_str = path_str.replace(
            f"{separator}images{separator}",
            f"{separator}{target_dir}{separator}",
        )
        path_str = path_str.replace(
            f"{separator}images",
            f"{separator}{target_dir}",
        )
    return Path(path_str)


def img2normal_paths(
    img_paths: List[Path],
    normals_dir: str = "normals",
) -> List[Path]:
    """Map image paths to same-stem PNGs under ``normals_dir``."""
    return [
        _replace_images_dir(img_path, normals_dir).with_suffix(".png")
        for img_path in img_paths
    ]


def img2normal_mask_paths(
    img_paths: List[Path],
    masks_dir: str = "masks",
) -> List[Path | None]:
    """Map image paths to optional same-stem validity-mask PNGs."""
    mask_paths: List[Path | None] = []
    for img_path in img_paths:
        candidate = _replace_images_dir(img_path, masks_dir).with_suffix(".png")
        mask_paths.append(candidate if candidate.exists() else None)
    return mask_paths


def normalize_normal_vectors(
    normals: np.ndarray,
    valid: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize an ``(H, W, 3)`` vector field and return its validity mask."""
    normals = np.asarray(normals, dtype=np.float32)
    if normals.ndim != 3 or normals.shape[-1] != 3:
        raise ValueError(
            f"expected (H, W, 3) normal vectors, got shape {normals.shape}"
        )

    finite = np.isfinite(normals).all(axis=-1)
    norms = np.linalg.norm(np.where(finite[..., None], normals, 0.0), axis=-1)
    normalized_valid = finite & (norms > _NORM_EPS)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != normals.shape[:2]:
            raise ValueError(
                f"normal validity mask shape {valid.shape} does not match "
                f"normal shape {normals.shape[:2]}"
            )
        normalized_valid &= valid

    normalized = np.zeros_like(normals, dtype=np.float32)
    normalized[normalized_valid] = (
        normals[normalized_valid] / norms[normalized_valid, None]
    )
    return normalized, normalized_valid


def load_normal_png(path: Path) -> np.ndarray:
    """Decode a strict three-channel uint16 RGB normal PNG."""
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        raise ValueError(f"Could not read normal map {path}.")
    if encoded.dtype != np.uint16:
        raise ValueError(
            f"Normal map {path} has dtype {encoded.dtype}; expected uint16."
        )
    if encoded.ndim != 3 or encoded.shape[-1] != 3:
        raise ValueError(
            f"Normal map {path} has shape {encoded.shape}; expected (H, W, 3)."
        )

    # OpenCV decodes color files as BGR; the on-disk contract is RGB.
    encoded_rgb = encoded[..., ::-1]
    normals = encoded_rgb.astype(np.float32) / _NORMAL_SCALE * 2.0 - 1.0
    normals, _ = normalize_normal_vectors(normals)
    return normals


def load_normal_mask(path: Path) -> np.ndarray:
    """Decode a single-channel PNG validity mask (nonzero means valid)."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not read normal validity mask {path}.")
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(
            f"Normal validity mask {path} has shape {mask.shape}; "
            "expected a single-channel image."
        )
    return np.isfinite(mask) & (mask > 0)


def resolve_normal_data(data: str | Path, allow_scripts: bool = False) -> Dict:
    """Load a normal dataset YAML through the shared data resolver."""
    return load_data_config(str(data), allow_scripts=allow_scripts)


class NormalDataset(Dataset):
    """Dense normal dataset returning ``(image, normal, info, id)``."""

    def __init__(
        self,
        data_config: Dict,
        split: str,
        imgsz: int,
        augment: bool = False,
        resize_mode: str = "letterbox",
    ):
        if resize_mode not in ("letterbox", "stretch"):
            raise ValueError(
                f"resize_mode must be 'letterbox' or 'stretch', got {resize_mode!r}"
            )
        self.split = split
        self.imgsz = int(imgsz)
        self.augment = bool(augment)
        self.resize_mode = resize_mode

        split_value = data_config.get(split)
        if not split_value:
            raise ValueError(f"Normal dataset config has no '{split}' split.")
        self.img_files = data_config.get(f"{split}_img_files") or get_img_files(
            split_value
        )
        if not self.img_files:
            raise FileNotFoundError(
                f"No images found for normal split '{split}' at {split_value}."
            )

        self.normals_dir = str(data_config.get("normals_dir") or "normals")
        self.normal_files = img2normal_paths(self.img_files, self.normals_dir)
        missing = [str(path) for path in self.normal_files if not path.exists()]
        if missing:
            preview = ", ".join(missing[:3])
            raise FileNotFoundError(
                f"{len(missing)} normal file(s) missing for split '{split}' "
                f"(e.g. {preview}). Expected uint16 RGB PNGs under "
                f"'{self.normals_dir}' mirroring the images tree."
            )

        self.masks_dir = str(data_config.get("masks_dir") or "masks")
        self.mask_files = img2normal_mask_paths(self.img_files, self.masks_dir)

    def __len__(self) -> int:
        return len(self.img_files)

    def _load_target(
        self,
        index: int,
        orig_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        normals = load_normal_png(self.normal_files[index])
        if normals.shape[:2] != orig_shape:
            raise ValueError(
                f"Normal map {self.normal_files[index]} shape {normals.shape[:2]} "
                f"does not match image shape {orig_shape}."
            )

        mask_path = self.mask_files[index]
        valid = np.ones(orig_shape, dtype=bool)
        if mask_path is not None:
            valid = load_normal_mask(mask_path)
            if valid.shape != orig_shape:
                raise ValueError(
                    f"Normal validity mask {mask_path} shape {valid.shape} "
                    f"does not match image shape {orig_shape}."
                )
        return normalize_normal_vectors(normals, valid)

    def _resize(
        self,
        image: np.ndarray,
        normals: np.ndarray,
        valid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
        """Resize image/vectors and pad to a square validation canvas."""
        h0, w0 = image.shape[:2]
        if self.resize_mode == "stretch":
            new_w = new_h = self.imgsz
            ratio = 1.0
        else:
            ratio = min(self.imgsz / h0, self.imgsz / w0)
            new_w = max(1, int(round(w0 * ratio)))
            new_h = max(1, int(round(h0 * ratio)))

        image = np.array(
            Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR),
            copy=True,
        )
        resized_components = [
            np.asarray(
                Image.fromarray(normals[..., component], mode="F").resize(
                    (new_w, new_h), Image.BILINEAR
                ),
                dtype=np.float32,
            )
            for component in range(3)
        ]
        normals = np.stack(resized_components, axis=-1)
        valid = (
            np.asarray(
                Image.fromarray(valid.astype(np.uint8) * 255).resize(
                    (new_w, new_h), Image.NEAREST
                )
            )
            > 0
        )
        normals, valid = normalize_normal_vectors(normals, valid)

        pad_h = self.imgsz - new_h
        pad_w = self.imgsz - new_w
        if pad_h or pad_w:
            image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                constant_values=_PAD_COLOR,
            )
            normals = np.pad(
                normals,
                ((0, pad_h), (0, pad_w), (0, 0)),
                constant_values=0.0,
            )
        return image, normals, ratio, (0, 0)

    def __getitem__(self, index: int):
        image_path = self.img_files[index]
        with Image.open(image_path) as image_file:
            image = np.array(image_file.convert("RGB"), copy=True)
        orig_shape = image.shape[:2]
        normals, valid = self._load_target(index, orig_shape)

        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            normals = np.ascontiguousarray(normals[:, ::-1])
            normals[..., 0] *= -1.0
            valid = np.ascontiguousarray(valid[:, ::-1])

        image, normals, ratio, pad = self._resize(image, normals, valid)
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        normal_tensor = (
            torch.from_numpy(np.ascontiguousarray(normals)).permute(2, 0, 1).float()
        )
        image_info = {
            "orig_shape": (int(orig_shape[0]), int(orig_shape[1])),
            "ratio": float(ratio),
            "pad": (int(pad[0]), int(pad[1])),
            "resize_mode": self.resize_mode,
            "img_path": str(image_path),
        }
        return image_tensor, normal_tensor, image_info, index


def normal_collate_fn(batch):
    """Collate samples into ``[B,3,H,W]`` images and normal targets."""
    images = torch.stack([item[0] for item in batch], dim=0)
    normals = torch.stack([item[1] for item in batch], dim=0)
    image_infos = [item[2] for item in batch]
    image_ids = [item[3] for item in batch]
    return images, normals, image_infos, image_ids


__all__ = [
    "NormalDataset",
    "img2normal_mask_paths",
    "img2normal_paths",
    "load_normal_mask",
    "load_normal_png",
    "normal_collate_fn",
    "normalize_normal_vectors",
    "resolve_normal_data",
]
