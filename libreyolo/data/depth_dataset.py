"""Monocular-depth dataset for LibreYOLO.

Depth datasets pair each image with a dense single-channel depth map; pixels
with value ``0`` (or negative / non-finite values) are invalid and excluded
from loss and metrics::

    dataset/
        images/train/*.jpg
        images/val/*.jpg
        depths/train/*.png     # same stem as the paired image
        depths/val/*.png

The dataset YAML follows the common contract (``path``/``train``/``val``/
``test``) plus two depth keys:

- ``depths_dir``: depth directory name substituted for ``images`` in each
  image path (default ``depths``).
- ``depth_scale``: divisor applied to integer-typed depth files (default
  ``256.0``, the common 16-bit PNG encoding where ``value / 256`` is the
  stored depth). Float files (``.npy``) are used as-is.

Ground-truth values are plain depth (distance from the camera) in whatever
unit the dataset uses; the relative-depth training objective is
scale-and-shift invariant, so no unit normalization is required.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import get_img_files, load_data_config

logger = logging.getLogger(__name__)

# Letterbox padding color for the image canvas (matches detection letterbox).
_PAD_COLOR = 114

DEPTH_FORMATS = (".png", ".npy", ".tif", ".tiff")

DEFAULT_DEPTH_SCALE = 256.0


def img2depth_paths(img_paths: List[Path], depths_dir: str = "depths") -> List[Path]:
    """Convert image paths to depth-map paths.

    Convention mirrors ``img2label_paths``: replace ``images`` with
    ``depths_dir`` in the path and look for a depth-capable extension with
    the same stem.
    """
    depth_paths = []
    for img_path in img_paths:
        path_str = str(img_path)
        for sep in (os.sep, "/", "\\"):
            path_str = path_str.replace(f"{sep}images{sep}", f"{sep}{depths_dir}{sep}")
            path_str = path_str.replace(f"{sep}images", f"{sep}{depths_dir}")
        base = Path(path_str)
        for suffix in DEPTH_FORMATS:
            candidate = base.with_suffix(suffix)
            if candidate.exists():
                base = candidate
                break
        else:
            base = base.with_suffix(".png")
        depth_paths.append(base)
    return depth_paths


def _load_depth_file(depth_path: Path, depth_scale: float) -> np.ndarray:
    """Read a depth file into a float32 ``(H, W)`` array of depth values.

    Integer-typed data (16-bit PNG / TIF) is divided by ``depth_scale``;
    float data (``.npy``, float TIF) is used as-is. Non-finite and negative
    values are mapped to ``0`` (invalid).
    """
    if depth_path.suffix.lower() == ".npy":
        depth = np.load(depth_path)
    else:
        with Image.open(depth_path) as depth_img:
            if depth_img.mode in ("I", "I;16", "L", "F"):
                depth = np.asarray(depth_img)
            else:
                raise ValueError(
                    f"Depth map {depth_path} has mode '{depth_img.mode}'. "
                    "Depth maps must be single-channel images "
                    "(PNG/TIF modes I, I;16, L, or F)."
                )
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(
            f"Depth map {depth_path} has shape {depth.shape}; expected (H, W)."
        )
    if np.issubdtype(depth.dtype, np.integer):
        depth = depth.astype(np.float32) / float(depth_scale)
    else:
        depth = depth.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0
    return depth


def resolve_depth_data(data: str | Path, allow_scripts: bool = False) -> Dict:
    """Load and sanity-check a depth dataset YAML config.

    Returns the ``load_data_config`` dict. ``allow_scripts`` is forwarded to
    ``load_data_config`` so explicitly allowed Python download scripts run.
    """
    config = load_data_config(str(data), allow_scripts=allow_scripts)
    return config


class DepthDataset(Dataset):
    """Dense depth dataset returning ``(img, depth, info, id)``.

    Images are letterboxed (default) or stretched to ``imgsz``; depth maps
    follow with nearest-neighbor geometry and invalid (``0``) padding.
    Training augmentation applies horizontal flips only — geometric scale
    changes are unnecessary because the training objective is
    scale-and-shift invariant.
    """

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
        self.augment = augment
        self.resize_mode = resize_mode
        self.depth_scale = float(
            data_config.get("depth_scale", DEFAULT_DEPTH_SCALE)
        )
        if self.depth_scale <= 0:
            raise ValueError(f"depth_scale must be positive, got {self.depth_scale}")

        split_value = data_config.get(split)
        if not split_value:
            raise ValueError(f"Depth dataset config has no '{split}' split.")
        self.img_files = data_config.get(f"{split}_img_files") or get_img_files(
            split_value
        )
        if not self.img_files:
            raise FileNotFoundError(
                f"No images found for depth split '{split}' at {split_value}."
            )

        self.depths_dir = str(data_config.get("depths_dir") or "depths")
        self.depth_files = img2depth_paths(self.img_files, self.depths_dir)
        missing = [str(p) for p in self.depth_files if not p.exists()]
        if missing:
            preview = ", ".join(missing[:3])
            raise FileNotFoundError(
                f"{len(missing)} depth file(s) missing for split '{split}' "
                f"(e.g. {preview}). Expected depth maps under "
                f"'{self.depths_dir}' mirroring the images tree."
            )

    def __len__(self) -> int:
        return len(self.img_files)

    def _load_target_depth(self, index: int, orig_shape: Tuple[int, int]) -> np.ndarray:
        depth = _load_depth_file(self.depth_files[index], self.depth_scale)
        if depth.shape != orig_shape:
            raise ValueError(
                f"Depth map {self.depth_files[index]} shape {depth.shape} "
                f"does not match image shape {orig_shape}."
            )
        return depth

    def _resize(
        self, img: np.ndarray, depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
        """Resize image (bilinear) and depth (nearest) and pad to imgsz."""
        h0, w0 = img.shape[:2]
        if self.resize_mode == "stretch":
            new_w = new_h = self.imgsz
            ratio = 1.0
        else:
            ratio = min(self.imgsz / h0, self.imgsz / w0)
            new_w = max(1, int(round(w0 * ratio)))
            new_h = max(1, int(round(h0 * ratio)))

        img_pil = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
        # Nearest-neighbor keeps depth discontinuities sharp instead of
        # inventing intermediate depths along object boundaries.
        depth_pil = Image.fromarray(depth, mode="F").resize(
            (new_w, new_h), Image.NEAREST
        )
        img = np.array(img_pil)
        depth = np.asarray(depth_pil).astype(np.float32)

        # Top-left anchored padding, matching the family inference letterbox
        # (content at the origin, pad at bottom/right). Pad depth with 0 so
        # padded pixels are invalid and self-exclude from loss and metrics.
        pad_h = self.imgsz - new_h
        pad_w = self.imgsz - new_w
        if pad_h or pad_w:
            img = np.pad(
                img,
                ((0, pad_h), (0, pad_w), (0, 0)),
                constant_values=_PAD_COLOR,
            )
            depth = np.pad(
                depth,
                ((0, pad_h), (0, pad_w)),
                constant_values=0.0,
            )
        return img, depth, ratio, (0, 0)

    def __getitem__(self, index: int):
        img_path = self.img_files[index]
        with Image.open(img_path) as img_pil:
            img = np.array(img_pil.convert("RGB"))
        orig_shape = img.shape[:2]
        depth = self._load_target_depth(index, orig_shape)

        if self.augment and random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            depth = np.ascontiguousarray(depth[:, ::-1])

        img, depth, ratio, pad = self._resize(img, depth)

        img_tensor = (
            torch.from_numpy(np.ascontiguousarray(img))
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        depth_tensor = torch.from_numpy(np.ascontiguousarray(depth)).float()
        img_info = {
            "orig_shape": (int(orig_shape[0]), int(orig_shape[1])),
            "ratio": float(ratio),
            "pad": (int(pad[0]), int(pad[1])),
            "resize_mode": self.resize_mode,
            "img_path": str(img_path),
        }
        return img_tensor, depth_tensor, img_info, index


def depth_collate_fn(batch):
    """Collate depth samples into the trainer's 4-tuple batch shape.

    Returns ``(imgs, depths, img_infos, img_ids)``: ``imgs`` is ``[B,3,H,W]``
    float in ``[0, 1]`` and ``depths`` is ``[B,H,W]`` float32 with invalid
    pixels set to ``0``.
    """
    imgs = torch.stack([item[0] for item in batch], dim=0)
    depths = torch.stack([item[1] for item in batch], dim=0)
    img_infos = [item[2] for item in batch]
    img_ids = [item[3] for item in batch]
    return imgs, depths, img_infos, img_ids
