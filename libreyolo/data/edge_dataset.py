"""Dense edge-map dataset for LibreYOLO.

Edge datasets pair each RGB image with a single-channel lossless map:

    dataset/
        images/train/*.jpg
        images/val/*.jpg
        edges/train/*.png     # same stem as the paired image
        edges/val/*.png
        masks/train/*.png     # optional; nonzero pixels are valid
        masks/val/*.png

Integer maps are normalized by their dtype maximum and float maps must already
be in ``[0, 1]``. Set ``edge_invert: true`` in the dataset YAML when the source
stores edges as black pixels on a white background. Returned targets carry both
the edge probabilities and a validity mask so letterbox padding never
contributes to validation metrics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import get_img_files, load_data_config

_PAD_COLOR = 114


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


def img2edge_paths(
    img_paths: List[Path],
    edges_dir: str = "edges",
    *,
    suffix: str = "",
    extension: str = ".png",
) -> List[Path]:
    """Map image paths to same-stem edge maps under ``edges_dir``."""
    if extension and not extension.startswith("."):
        extension = f".{extension}"
    return [
        _replace_images_dir(img_path, edges_dir).with_name(
            f"{img_path.stem}{suffix}{extension}"
        )
        for img_path in img_paths
    ]


def img2edge_mask_paths(
    img_paths: List[Path],
    masks_dir: str = "masks",
) -> List[Path | None]:
    """Map image paths to optional same-stem validity masks."""
    paths: List[Path | None] = []
    for image_path in img_paths:
        candidate = _replace_images_dir(image_path, masks_dir).with_suffix(".png")
        paths.append(candidate if candidate.exists() else None)
    return paths


def load_edge_map(path: Path, *, invert: bool = False) -> np.ndarray:
    """Decode a strict single-channel edge map as float32 in ``[0, 1]``."""
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        raise ValueError(f"Could not read edge map {path}.")
    if encoded.ndim == 3 and encoded.shape[-1] == 1:
        encoded = encoded[..., 0]
    if encoded.ndim != 2:
        raise ValueError(
            f"Edge map {path} has shape {encoded.shape}; expected one channel."
        )

    if np.issubdtype(encoded.dtype, np.integer):
        scale = float(np.iinfo(encoded.dtype).max)
        edge = encoded.astype(np.float32) / scale
    elif np.issubdtype(encoded.dtype, np.floating):
        edge = encoded.astype(np.float32)
    else:
        raise ValueError(f"Edge map {path} has unsupported dtype {encoded.dtype}.")

    if not bool(np.isfinite(edge).all()):
        raise ValueError(f"Edge map {path} contains non-finite values.")
    if bool(np.any((edge < 0.0) | (edge > 1.0))):
        raise ValueError(f"Edge map {path} values must be in [0, 1].")
    return np.ascontiguousarray(1.0 - edge if invert else edge)


def load_edge_mask(path: Path) -> np.ndarray:
    """Decode a single-channel validity mask (nonzero means valid)."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not read edge validity mask {path}.")
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(
            f"Edge validity mask {path} has shape {mask.shape}; expected one channel."
        )
    return np.isfinite(mask) & (mask > 0)


def resolve_edge_data(data: str | Path, allow_scripts: bool = False) -> Dict:
    """Load an edge dataset YAML through the shared data resolver."""
    return load_data_config(str(data), allow_scripts=allow_scripts)


class EdgeDataset(Dataset):
    """Dense edge dataset returning images, edge targets, metadata, and ids."""

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
            raise ValueError(f"Edge dataset config has no '{split}' split.")
        self.img_files = data_config.get(f"{split}_img_files") or get_img_files(
            split_value
        )
        if not self.img_files:
            raise FileNotFoundError(
                f"No images found for edge split '{split}' at {split_value}."
            )

        self.edges_dir = str(data_config.get("edges_dir") or "edges")
        self.edge_stem_suffix = str(data_config.get("edge_stem_suffix") or "")
        self.edge_extension = str(data_config.get("edge_extension") or ".png")
        self.edge_files = img2edge_paths(
            self.img_files,
            self.edges_dir,
            suffix=self.edge_stem_suffix,
            extension=self.edge_extension,
        )
        missing = [str(path) for path in self.edge_files if not path.exists()]
        if missing:
            preview = ", ".join(missing[:3])
            raise FileNotFoundError(
                f"{len(missing)} edge file(s) missing for split '{split}' "
                f"(e.g. {preview}). Expected single-channel lossless maps under "
                f"'{self.edges_dir}' mirroring the images tree."
            )

        self.edge_invert = bool(data_config.get("edge_invert", False))
        self.masks_dir = str(data_config.get("masks_dir") or "masks")
        self.mask_files = img2edge_mask_paths(self.img_files, self.masks_dir)

    def __len__(self) -> int:
        return len(self.img_files)

    def _load_target(
        self,
        index: int,
        orig_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        edge = load_edge_map(self.edge_files[index], invert=self.edge_invert)
        if edge.shape != orig_shape:
            raise ValueError(
                f"Edge map {self.edge_files[index]} shape {edge.shape} does not "
                f"match image shape {orig_shape}."
            )

        valid = np.ones(orig_shape, dtype=bool)
        mask_path = self.mask_files[index]
        if mask_path is not None:
            valid = load_edge_mask(mask_path)
            if valid.shape != orig_shape:
                raise ValueError(
                    f"Edge validity mask {mask_path} shape {valid.shape} does "
                    f"not match image shape {orig_shape}."
                )
        return edge, valid

    def _resize(
        self,
        image: np.ndarray,
        edge: np.ndarray,
        valid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Tuple[int, int]]:
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
        edge = np.array(
            Image.fromarray(edge, mode="F").resize((new_w, new_h), Image.NEAREST),
            dtype=np.float32,
            copy=True,
        )
        valid = (
            np.asarray(
                Image.fromarray(valid.astype(np.uint8) * 255).resize(
                    (new_w, new_h), Image.NEAREST
                )
            )
            > 0
        )

        pad_h = self.imgsz - new_h
        pad_w = self.imgsz - new_w
        if pad_h or pad_w:
            image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                constant_values=_PAD_COLOR,
            )
            edge = np.pad(edge, ((0, pad_h), (0, pad_w)), constant_values=0.0)
            valid = np.pad(
                valid,
                ((0, pad_h), (0, pad_w)),
                constant_values=False,
            )
        return image, edge, valid, ratio, (0, 0)

    def __getitem__(self, index: int):
        image_path = self.img_files[index]
        with Image.open(image_path) as image_file:
            image = np.array(image_file.convert("RGB"), copy=True)
        orig_shape = image.shape[:2]
        edge, valid = self._load_target(index, orig_shape)

        image, edge, valid, ratio, pad = self._resize(image, edge, valid)
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        target = {
            "edges": torch.from_numpy(np.ascontiguousarray(edge)).unsqueeze(0).float(),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)).unsqueeze(0).bool(),
        }
        image_info = {
            "orig_shape": (int(orig_shape[0]), int(orig_shape[1])),
            "ratio": float(ratio),
            "pad": (int(pad[0]), int(pad[1])),
            "resize_mode": self.resize_mode,
            "img_path": str(image_path),
        }
        return image_tensor, target, image_info, index


def edge_collate_fn(batch):
    """Collate samples into images and dense edge/validity tensors."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = {
        "edges": torch.stack([item[1]["edges"] for item in batch], dim=0),
        "valid": torch.stack([item[1]["valid"] for item in batch], dim=0),
    }
    image_infos = [item[2] for item in batch]
    image_ids = [item[3] for item in batch]
    return images, targets, image_infos, image_ids


__all__ = [
    "EdgeDataset",
    "edge_collate_fn",
    "img2edge_mask_paths",
    "img2edge_paths",
    "load_edge_map",
    "load_edge_mask",
    "resolve_edge_data",
]
