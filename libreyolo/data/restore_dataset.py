"""Paired RGB restoration dataset for LibreYOLO.

Restoration datasets pair a degraded input image with a clean target image:

    inputs/train/*.jpg
    targets/train/*.jpg

Pairs are matched by stem. Training uses coupled crop/flip/rot90 augmentation so
the same geometry is applied to the input and target. Validation keeps native
resolution; the collate function pads only enough to stack a batch and records
the original shape so metrics ignore padded pixels.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .utils import IMG_FORMATS, get_img_files, load_data_config

logger = logging.getLogger(__name__)

RESTORE_FORMATS = tuple(sorted(IMG_FORMATS))


def _as_tuple(value: object, default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return tuple(value)
    raise TypeError("restore stem suffixes must be a string or a list of strings.")


def img2restore_target_paths(
    img_paths: List[Path],
    *,
    input_dir: str = "inputs",
    target_dir: str = "targets",
    stem_suffixes: Tuple[str, ...] = ("",),
) -> List[Path]:
    """Resolve target paths by replacing the input directory component."""

    target_paths = []
    for img_path in img_paths:
        path_str = str(img_path)
        replaced = False
        for sep in (os.sep, "/", "\\"):
            old = f"{sep}{input_dir}{sep}"
            if old in path_str:
                path_str = path_str.replace(old, f"{sep}{target_dir}{sep}")
                replaced = True
                break
            old_tail = f"{sep}{input_dir}"
            if path_str.endswith(old_tail):
                path_str = path_str[: -len(input_dir)] + target_dir
                replaced = True
                break
        base = Path(path_str if replaced else img_path)
        if not replaced:
            base = img_path.parent.parent / target_dir / img_path.name
        candidates = [
            base.with_name(f"{base.stem}{stem_suffix}").with_suffix(suffix)
            for stem_suffix in stem_suffixes
            for suffix in RESTORE_FORMATS
        ]
        for candidate in candidates:
            if candidate.exists():
                base = candidate
                break
        else:
            base = base.with_suffix(img_path.suffix)
        target_paths.append(base)
    return target_paths


def resolve_restore_data(data: str | Path, allow_scripts: bool = False) -> Dict:
    """Load and sanity-check a restoration dataset YAML config."""

    return load_data_config(str(data), allow_scripts=allow_scripts)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _reflect_pad_hwc(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if not pad_h and not pad_w:
        return img
    mode = "reflect" if h > 1 and w > 1 and pad_h < h and pad_w < w else "edge"
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)


class RestoreDataset(Dataset):
    """Paired RGB restoration dataset returning ``(input, target, info, id)``."""

    def __init__(
        self,
        data_config: Dict,
        split: str,
        imgsz: int,
        augment: bool = False,
        scale: int = 1,
    ):
        self.split = split
        self.imgsz = int(imgsz)
        self.augment = augment
        self.scale = int(scale) if scale else 1
        if self.scale < 1:
            raise ValueError(f"restore dataset scale must be >= 1, got {scale}.")
        if self.augment and self.scale != 1:
            raise NotImplementedError(
                "Coupled crop/flip augmentation is not implemented for "
                "super-resolution pairs (scale != 1). Real-ESRGAN training is "
                "out of scope; use augment=False for validation."
            )

        split_value = data_config.get(split)
        if not split_value:
            raise ValueError(f"Restore dataset config has no '{split}' split.")
        self.img_files = data_config.get(f"{split}_img_files") or get_img_files(
            split_value
        )
        if not self.img_files:
            raise FileNotFoundError(
                f"No input images found for restore split '{split}' at {split_value}."
            )

        self.input_dir = str(data_config.get("input_dir") or "inputs")
        self.target_dir = str(
            data_config.get("target_dir") or data_config.get("targets_dir") or "targets"
        )
        stem_suffixes = _as_tuple(
            data_config.get("target_stem_suffix")
            if "target_stem_suffix" in data_config
            else data_config.get("target_stem_suffixes"),
            ("",),
        )
        self.target_files = img2restore_target_paths(
            self.img_files,
            input_dir=self.input_dir,
            target_dir=self.target_dir,
            stem_suffixes=stem_suffixes,
        )
        missing = [str(p) for p in self.target_files if not p.exists()]
        if missing:
            preview = ", ".join(missing[:3])
            raise FileNotFoundError(
                f"{len(missing)} target image file(s) missing for split '{split}' "
                f"(e.g. {preview}). Expected targets under '{self.target_dir}' "
                "with stems matching the input images."
            )

    def __len__(self) -> int:
        return len(self.img_files)

    def _augment_pair(
        self,
        inp: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.augment:
            inp = _reflect_pad_hwc(inp, self.imgsz, self.imgsz)
            target = _reflect_pad_hwc(target, self.imgsz, self.imgsz)
            h, w = inp.shape[:2]
            top = random.randint(0, max(0, h - self.imgsz))
            left = random.randint(0, max(0, w - self.imgsz))
            inp = inp[top : top + self.imgsz, left : left + self.imgsz]
            target = target[top : top + self.imgsz, left : left + self.imgsz]
            # Coupled geometry: horizontal flip, vertical flip, and a random
            # rot90. The same op is applied to input and target so pixels stay
            # aligned. Crops are square (imgsz x imgsz) so rot90 keeps shape.
            if random.random() < 0.5:
                inp = np.ascontiguousarray(inp[:, ::-1])
                target = np.ascontiguousarray(target[:, ::-1])
            if random.random() < 0.5:
                inp = np.ascontiguousarray(inp[::-1, :])
                target = np.ascontiguousarray(target[::-1, :])
            k = random.randint(0, 3)
            if k:
                inp = np.ascontiguousarray(np.rot90(inp, k))
                target = np.ascontiguousarray(np.rot90(target, k))
        return inp, target

    def __getitem__(self, index: int):
        img_path = self.img_files[index]
        target_path = self.target_files[index]
        inp = _load_rgb(img_path)
        target = _load_rgb(target_path)
        ih, iw = inp.shape[:2]
        th, tw = target.shape[:2]
        if self.scale == 1:
            if inp.shape != target.shape:
                raise ValueError(
                    f"Restore pair shape mismatch: {img_path} has {inp.shape[:2]}, "
                    f"{target_path} has {target.shape[:2]}."
                )
        elif (th, tw) != (ih * self.scale, iw * self.scale):
            raise ValueError(
                f"Super-resolution pair shape mismatch (scale {self.scale}): input "
                f"{img_path} is {(ih, iw)} so the ground-truth target must be "
                f"{(ih * self.scale, iw * self.scale)}, but {target_path} is {(th, tw)}. "
                "Provide high-resolution targets at scale x the input size."
            )
        orig_shape = (ih, iw)
        target_shape = (th, tw)
        inp, target = self._augment_pair(inp, target)

        inp_tensor = (
            torch.from_numpy(np.ascontiguousarray(inp))
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        target_tensor = (
            torch.from_numpy(np.ascontiguousarray(target))
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        img_info = {
            "orig_shape": (int(orig_shape[0]), int(orig_shape[1])),
            "target_shape": (int(target_shape[0]), int(target_shape[1])),
            "img_path": str(img_path),
            "target_path": str(target_path),
        }
        return inp_tensor, target_tensor, img_info, index


def _pad_chw(tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
    pad_h = h - tensor.shape[-2]
    pad_w = w - tensor.shape[-1]
    if not pad_h and not pad_w:
        return tensor
    mode = (
        "reflect"
        if tensor.shape[-2] > 1
        and tensor.shape[-1] > 1
        and pad_h < tensor.shape[-2]
        and pad_w < tensor.shape[-1]
        else "replicate"
    )
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)


def restore_collate_fn(batch):
    """Collate restore samples into ``(imgs, targets, img_infos, img_ids)``.

    Inputs and targets are padded to their own maxima so super-resolution pairs
    (where the target is ``scale`` x the input) stack correctly.
    """

    max_h = max(item[0].shape[-2] for item in batch)
    max_w = max(item[0].shape[-1] for item in batch)
    tmax_h = max(item[1].shape[-2] for item in batch)
    tmax_w = max(item[1].shape[-1] for item in batch)
    imgs = torch.stack([_pad_chw(item[0], max_h, max_w) for item in batch], dim=0)
    targets = torch.stack([_pad_chw(item[1], tmax_h, tmax_w) for item in batch], dim=0)
    img_infos = [item[2] for item in batch]
    img_ids = [item[3] for item in batch]
    return imgs, targets, img_infos, img_ids


__all__ = [
    "RestoreDataset",
    "img2restore_target_paths",
    "resolve_restore_data",
    "restore_collate_fn",
]
