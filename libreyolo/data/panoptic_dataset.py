"""COCO-panoptic dataset for the ``panoptic`` task.

Panoptic ground truth is the COCO-panoptic format (`Panoptic Segmentation`,
Kirillov et al., CVPR 2019): one PNG per image whose RGB encodes a **segment
id**, plus a JSON file listing, per image, the segments present.

The PNG encoding is a data-format fact, not a choice::

    segment_id = R + 256 * G + 256 * 256 * B

Segment id ``0`` (RGB black) is VOID: unlabeled pixels that neither reward nor
penalize a prediction.

Category ids in the JSON are the dataset's raw ids and are typically
non-contiguous (COCO panoptic runs 1..200 with gaps). LibreYOLO models predict
contiguous ``0..nc-1`` class indices, so raw ids are remapped through the YAML
``names`` by category **name**, matching the rule the native COCO-JSON detect
loader already follows: when ``names`` is present, it defines the label ids.

thing-vs-stuff is a per-category property carried on the JSON ``categories``
entries (``isthing``), never on individual segments. See
``docs/dataset_schema.md``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import load_data_config

logger = logging.getLogger(__name__)

# COCO panoptic reserves segment id 0 (RGB black) for unlabeled pixels.
VOID_SEGMENT_ID = 0


def rgb2id(color: np.ndarray) -> np.ndarray:
    """Decode a COCO-panoptic RGB PNG into a 2-D segment-id map.

    ``segment_id = R + 256 * G + 256 * 256 * B``.
    """
    if color.ndim != 3 or color.shape[2] < 3:
        raise ValueError(
            f"Panoptic PNG must be an (H, W, 3) RGB image, got shape {color.shape}."
        )
    color = color[:, :, :3].astype(np.uint32)
    return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]


def resolve_panoptic_data(data: str | Path, allow_scripts: bool = False) -> Dict:
    """Load and sanity-check a panoptic dataset YAML config."""
    config = load_data_config(str(data), allow_scripts=allow_scripts)
    if not config.get("names"):
        raise ValueError(f"Panoptic dataset config {data!r} must define class names.")
    if not config.get("annotations"):
        raise ValueError(
            f"Panoptic dataset config {data!r} must define 'annotations' mapping "
            "each split to its COCO-panoptic JSON file."
        )
    if not config.get("panoptic_dir"):
        raise ValueError(
            f"Panoptic dataset config {data!r} must define 'panoptic_dir', the "
            "directory holding the segment-id PNGs."
        )
    return config


def _split_value(value, split: str) -> str:
    """Accept either a single path or a per-split mapping."""
    if isinstance(value, dict):
        if split not in value:
            raise ValueError(f"No '{split}' entry in {value!r}.")
        return str(value[split])
    return str(value)


class PanopticDataset(Dataset):
    """COCO-panoptic dataset yielding ``(img_path, seg_map, segments_info, info)``.

    Images are returned as paths, not tensors: panoptic evaluation runs at the
    original image resolution, and each model family owns its own preprocessing
    geometry (EoMT pads COCO inputs, splits ADE20K ones). The validator asks the
    model to preprocess, so the dataset never has to guess.
    """

    def __init__(self, data_config: Dict, split: str = "val"):
        self.split = split

        root = Path(data_config.get("path") or ".")

        split_value = data_config.get(split)
        if not split_value:
            raise ValueError(f"Panoptic dataset config has no '{split}' split.")
        self.img_root = self._resolve(root, split_value)

        ann_path = self._resolve(root, _split_value(data_config["annotations"], split))
        self.panoptic_dir = self._resolve(
            root, _split_value(data_config["panoptic_dir"], split)
        )
        if not ann_path.is_file():
            raise FileNotFoundError(f"Panoptic annotations JSON not found: {ann_path}")
        if not self.panoptic_dir.is_dir():
            raise NotADirectoryError(
                f"Panoptic PNG directory not found: {self.panoptic_dir}"
            )

        names = data_config.get("names") or {}
        if isinstance(names, list):
            names = {index: name for index, name in enumerate(names)}
        self.names: Dict[int, str] = {int(k): str(v) for k, v in names.items()}
        self.nc = int(data_config.get("nc") or len(self.names))
        if self.nc != len(self.names):
            raise ValueError(f"nc={self.nc} does not match {len(self.names)} names.")

        with open(ann_path, encoding="utf-8") as handle:
            payload = json.load(handle)

        self.category_id_to_train_id, self.thing_train_ids = self._build_category_map(
            payload.get("categories") or []
        )

        images_by_id = {img["id"]: img for img in payload.get("images") or []}
        self.items: List[Tuple[Path, Path, List[dict]]] = []
        for ann in payload.get("annotations") or []:
            png_path = self.panoptic_dir / ann["file_name"]
            image_entry = images_by_id.get(ann["image_id"])
            if image_entry is None:
                raise ValueError(
                    f"Annotation references image_id={ann['image_id']} which is "
                    f"absent from the 'images' list in {ann_path}."
                )
            img_path = self.img_root / image_entry["file_name"]
            self.items.append((img_path, png_path, ann.get("segments_info") or []))

        if not self.items:
            raise ValueError(f"No panoptic annotations found in {ann_path}.")

    @staticmethod
    def _resolve(root: Path, value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    def _build_category_map(self, categories: List[dict]) -> Tuple[Dict[int, int], set]:
        """Map raw COCO category ids onto contiguous train ids via YAML ``names``.

        Raises rather than silently dropping: a category the model cannot name
        would otherwise be scored as a permanent false negative.
        """
        if not categories:
            raise ValueError("Panoptic JSON has no 'categories' list.")
        name_to_train_id = {name: idx for idx, name in self.names.items()}
        mapping: Dict[int, int] = {}
        thing_train_ids: set = set()
        for entry in categories:
            name = str(entry["name"])
            if name not in name_to_train_id:
                raise ValueError(
                    f"Panoptic category {name!r} (id={entry['id']}) is missing from "
                    "the dataset YAML 'names'. Category names define the label ids."
                )
            train_id = name_to_train_id[name]
            mapping[int(entry["id"])] = train_id
            if int(entry.get("isthing", 0)) == 1:
                thing_train_ids.add(train_id)
        return mapping, thing_train_ids

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        img_path, png_path, raw_segments = self.items[index]

        with Image.open(png_path) as handle:
            seg_map = rgb2id(np.asarray(handle.convert("RGB")))

        segments_info: List[dict] = []
        for segment in raw_segments:
            raw_category = int(segment["category_id"])
            if raw_category not in self.category_id_to_train_id:
                raise ValueError(
                    f"{png_path.name}: segment category_id={raw_category} is not in "
                    "the JSON 'categories' list."
                )
            segments_info.append(
                {
                    "id": int(segment["id"]),
                    "category_id": self.category_id_to_train_id[raw_category],
                    "iscrowd": int(segment.get("iscrowd", 0)),
                    "area": int(segment.get("area", 0)),
                }
            )

        info = {
            "img_path": str(img_path),
            "panoptic_path": str(png_path),
            "height": int(seg_map.shape[0]),
            "width": int(seg_map.shape[1]),
        }
        return (
            str(img_path),
            torch.from_numpy(seg_map.astype(np.int64)),
            segments_info,
            info,
        )


def panoptic_collate_fn(batch):
    """Collate panoptic samples, keeping per-image structure intact.

    Segment maps have per-image resolutions and variable-length ``segments_info``,
    so nothing is stacked. Panoptic validation runs one image at a time.
    """
    img_paths = [item[0] for item in batch]
    seg_maps = [item[1] for item in batch]
    segments_infos = [item[2] for item in batch]
    infos = [item[3] for item in batch]
    return img_paths, seg_maps, segments_infos, infos


__all__ = [
    "VOID_SEGMENT_ID",
    "PanopticDataset",
    "panoptic_collate_fn",
    "resolve_panoptic_data",
    "rgb2id",
]
