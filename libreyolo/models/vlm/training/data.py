"""Detect-dataset reader for VLM fine-tuning.

Reads standard LibreYOLO detect datasets (``docs/dataset_schema.md``), using
either normalized ``class cx cy w h`` txt labels or native COCO JSON boxes,
and yields PIL images with rendered conversation targets. The user never
writes conversations or coordinate text; that is the library's job, via
:mod:`.targets`.

Augmentation is geometric-safe only (horizontal flip), because every geometric
transform must re-render the target text; the flip happens before rendering.
"""

from __future__ import annotations

import logging
import math
import random
import warnings
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ....data import get_img_files, img2label_paths
from ....utils.coco_geometry import clipped_coco_bbox_xyxy
from .targets import FamilyFormat, serialize_detections

logger = logging.getLogger(__name__)

__all__ = ["VLMDetectDataset", "resolve_split_annotation", "resolve_split_source"]

_MAX_COCO_JSON_BYTES = 512 * 1024 * 1024
_MAX_SAFE_JSON_INTEGER = (1 << 53) - 1


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Native COCO JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"Native COCO JSON contains non-finite value {value!r}.")


def _finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Native COCO JSON contains non-finite value {value!r}.")
    return number


def _strict_coco_id(value, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > _MAX_SAFE_JSON_INTEGER
    ):
        raise ValueError(f"{label} must be a non-negative exact integer.")
    return value


def _load_strict_coco_json(path: Path) -> Dict[str, Any]:
    before = path.stat()
    if before.st_size > _MAX_COCO_JSON_BYTES:
        raise ValueError(
            f"Native COCO annotation file exceeds {_MAX_COCO_JSON_BYTES} bytes."
        )
    with path.open("rb") as stream:
        payload = stream.read(_MAX_COCO_JSON_BYTES + 1)
    after = path.stat()
    if len(payload) > _MAX_COCO_JSON_BYTES:
        raise ValueError(
            f"Native COCO annotation file exceeds {_MAX_COCO_JSON_BYTES} bytes."
        )
    if (
        before.st_size != len(payload)
        or after.st_size != len(payload)
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError("Native COCO annotation file changed while being read.")
    try:
        root = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"Native COCO annotation file is invalid: {path}") from exc
    if not isinstance(root, dict):
        raise ValueError("Native COCO annotation root must be a JSON object.")
    for field in ("images", "categories", "annotations"):
        if not isinstance(root.get(field), list):
            raise ValueError(f"Native COCO field {field!r} must be a JSON array.")

    indexed_ids = {}
    for field in ("images", "categories", "annotations"):
        ids = set()
        for index, record in enumerate(root[field]):
            if not isinstance(record, dict):
                raise ValueError(f"Native COCO {field}[{index}] must be an object.")
            record_id = _strict_coco_id(record.get("id"), f"{field}[{index}].id")
            if record_id in ids:
                raise ValueError(
                    f"Native COCO {field} contains duplicate id {record_id}."
                )
            ids.add(record_id)
        indexed_ids[field] = ids

    for index, annotation in enumerate(root["annotations"]):
        image_id = _strict_coco_id(
            annotation.get("image_id"), f"annotations[{index}].image_id"
        )
        category_id = _strict_coco_id(
            annotation.get("category_id"), f"annotations[{index}].category_id"
        )
        if image_id not in indexed_ids["images"]:
            raise ValueError(
                f"Native COCO annotation references unknown image id {image_id}."
            )
        if category_id not in indexed_ids["categories"]:
            raise ValueError(
                f"Native COCO annotation references unknown category id {category_id}."
            )
        for flag in ("iscrowd", "ignore"):
            value = annotation.get(flag, 0)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value
                not in {
                    0,
                    1,
                }
            ):
                raise ValueError(
                    f"Native COCO annotations[{index}].{flag} must be integer 0 or 1."
                )
    return root


def _parse_label_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Parse one txt label file into ``(cls, cx, cy, w, h)`` rows.

    Missing file means no objects (dataset-schema rule). Malformed rows are
    skipped with a warning rather than aborting a long training run.
    """
    if not path.exists():
        return []
    rows: List[Tuple[int, float, float, float, float]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except (ValueError, IndexError):
            logger.warning(
                "Skipping malformed label row %s:%d: %r", path, line_no, line
            )
            continue
        if len(parts) != 5:
            logger.warning(
                "Skipping label row with %d fields (detect rows have 5) %s:%d",
                len(parts),
                path,
                line_no,
            )
            continue
        rows.append((cls, cx, cy, w, h))
    return rows


class VLMDetectDataset:
    """Map-style dataset: detect labels in, conversation samples out.

    Each item is a dict with ``image`` (PIL, RGB), ``prompt`` (the family's
    detection prompt for the training vocabulary), and ``target`` (the JSON
    answer text rendered by :func:`serialize_detections`).
    """

    def __init__(
        self,
        image_source,
        names: Dict[int, str],
        fmt: FamilyFormat,
        augment: bool = False,
        hflip_p: float = 0.5,
        seed: int = 0,
        annotation_file: str | Path | None = None,
    ) -> None:
        self.names = dict(names)
        self.fmt = fmt
        self.augment = augment
        self.hflip_p = float(hflip_p)
        self._rng = random.Random(seed)
        self._coco = None
        self._coco_image_ids: List[int] = []
        self._coco_image_info: List[Dict[str, Any]] = []
        self._coco_category_to_label: Dict[int, int] = {}
        if annotation_file is None:
            self.images = get_img_files(image_source)
            if not self.images:
                raise FileNotFoundError(f"No images found under {image_source!r}")
            self.labels: List[Path] = img2label_paths(self.images)
        else:
            self.labels = []
            self._init_coco(image_source, annotation_file)
        self._warned_unknown_class = False

    @classmethod
    def validate_native_coco_source(
        cls,
        image_source,
        names: Dict[int, str],
        annotation_file: str | Path,
    ) -> None:
        """Validate a native COCO source without constructing model weights."""
        dataset = cls.__new__(cls)
        dataset.names = dict(names)
        dataset._coco = None
        dataset._coco_image_ids = []
        dataset._coco_image_info = []
        dataset._coco_category_to_label = {}
        dataset._init_coco(image_source, annotation_file)

    def _init_coco(self, image_source, annotation_file: str | Path) -> None:
        """Load a native COCO annotation file without copying image bytes."""
        if isinstance(image_source, (list, tuple)):
            raise ValueError("Native COCO VLM training requires one image directory.")
        image_root = Path(image_source).expanduser().resolve(strict=False)
        if not image_root.is_dir():
            raise FileNotFoundError(
                f"Native COCO image directory does not exist: {image_root}"
            )
        annotation_path = Path(annotation_file).expanduser().resolve(strict=False)
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Native COCO annotation file does not exist: {annotation_path}"
            )
        try:
            from pycocotools.coco import COCO
        except ImportError as exc:
            raise ImportError(
                "Native COCO VLM training requires pycocotools. Install the "
                "standard LibreYOLO training dependencies."
            ) from exc

        root = _load_strict_coco_json(annotation_path)
        # Build the standard COCO indexes from the already validated object.
        # pycocotools prints directly to stdout while indexing; keep library
        # and structured CLI output clean.
        try:
            with redirect_stdout(StringIO()):
                coco = COCO()
                coco.dataset = root
                coco.createIndex()
        except (OSError, TypeError, ValueError, KeyError, IndexError) as exc:
            raise ValueError(
                f"Native COCO annotation file is invalid: {annotation_path}"
            ) from exc
        category_ids = sorted(int(value) for value in coco.getCatIds())
        categories = coco.loadCats(category_ids)
        name_to_label: Dict[str, int] = {}
        for label, name in self.names.items():
            if name in name_to_label:
                raise ValueError(f"Duplicate VLM dataset class name: {name!r}")
            name_to_label[name] = int(label)
        category_to_label: Dict[int, int] = {}
        label_to_category: Dict[int, int] = {}
        for category in categories:
            name = category.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"COCO category {category.get('id')!r} has no valid name."
                )
            if name not in name_to_label:
                raise ValueError(
                    f"COCO category name not found in dataset YAML names: {name!r}"
                )
            category_id = int(category["id"])
            label = name_to_label[name]
            if label in label_to_category:
                raise ValueError(
                    f"Multiple COCO categories map to class {label}; dataset "
                    "YAML names must be unique."
                )
            category_to_label[category_id] = label
            label_to_category[label] = category_id

        image_ids = sorted(int(value) for value in coco.getImgIds())
        image_info = coco.loadImgs(image_ids)
        images: List[Path] = []
        for info in image_info:
            file_name = info.get("file_name")
            if not isinstance(file_name, str) or not file_name:
                raise ValueError(
                    f"COCO image {info.get('id')!r} has no valid file_name."
                )
            path = (image_root / file_name).resolve(strict=False)
            if not path.is_relative_to(image_root):
                raise ValueError(
                    f"COCO image path escapes its image directory: {file_name!r}"
                )
            if not path.is_file():
                raise FileNotFoundError(f"COCO image file does not exist: {path}")
            width = info.get("width")
            height = info.get("height")
            if (
                isinstance(width, bool)
                or isinstance(height, bool)
                or not isinstance(width, int)
                or not isinstance(height, int)
                or width < 1
                or height < 1
            ):
                raise ValueError(
                    f"COCO image {info.get('id')!r} has invalid dimensions."
                )
            if (
                Image.MAX_IMAGE_PIXELS is not None
                and width * height > Image.MAX_IMAGE_PIXELS
            ):
                raise ValueError(
                    f"COCO image {info.get('id')!r} exceeds the safe pixel limit."
                )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(path) as source_image:
                        source_image.load()
                        actual_size = source_image.size
            except (
                OSError,
                ValueError,
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
            ) as exc:
                raise ValueError(f"COCO image file cannot be read: {path}") from exc
            expected_size = (width, height)
            if actual_size != expected_size:
                raise ValueError(
                    f"COCO image {path} has size {actual_size}, but annotations "
                    f"declare {expected_size}."
                )
            images.append(path)

        if not images:
            raise FileNotFoundError(
                f"No images are declared by native COCO file {annotation_path}"
            )
        self._coco = coco
        self._coco_image_ids = image_ids
        self._coco_image_info = image_info
        self._coco_category_to_label = category_to_label
        self.images = images
        for index in range(len(self.images)):
            self._load_coco_boxes(index)

    def __len__(self) -> int:
        return len(self.images)

    def _load_boxes(self, index: int) -> Tuple[List[List[float]], List[str]]:
        if self._coco is not None:
            return self._load_coco_boxes(index)
        boxes: List[List[float]] = []
        labels: List[str] = []
        for cls, cx, cy, w, h in _parse_label_file(self.labels[index]):
            name = self.names.get(cls)
            if name is None:
                if not self._warned_unknown_class:
                    logger.warning(
                        "Label file %s references class id %d not present in the "
                        "dataset names; such rows are skipped.",
                        self.labels[index],
                        cls,
                    )
                    self._warned_unknown_class = True
                continue
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            boxes.append(
                [
                    min(max(x1, 0.0), 1.0),
                    min(max(y1, 0.0), 1.0),
                    min(max(x2, 0.0), 1.0),
                    min(max(y2, 0.0), 1.0),
                ]
            )
            labels.append(name)
        return boxes, labels

    def _load_coco_boxes(self, index: int) -> Tuple[List[List[float]], List[str]]:
        info = self._coco_image_info[index]
        image_id = self._coco_image_ids[index]
        width = int(info["width"])
        height = int(info["height"])
        annotations = self._coco.loadAnns(self._coco.getAnnIds(imgIds=[image_id]))
        boxes: List[List[float]] = []
        labels: List[str] = []
        for annotation in annotations:
            if annotation.get("iscrowd") or annotation.get("ignore"):
                continue
            category_id = annotation.get("category_id")
            try:
                known_category = category_id in self._coco_category_to_label
            except TypeError as exc:
                raise ValueError(
                    f"COCO annotation {annotation.get('id')!r} has an invalid "
                    "category id."
                ) from exc
            if not known_category:
                raise ValueError(
                    f"COCO annotation {annotation.get('id')!r} references "
                    f"unknown category id {category_id!r}."
                )
            try:
                raw_area = annotation.get("area", 1.0)
                if isinstance(raw_area, bool) or not isinstance(raw_area, (int, float)):
                    raise ValueError("area must be numeric")
                area = float(raw_area)
                clean = clipped_coco_bbox_xyxy(annotation["bbox"], width, height)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"COCO annotation {annotation.get('id')!r} has an invalid bbox."
                ) from exc
            if not math.isfinite(area) or area <= 0.0 or clean is None:
                continue
            x1, y1, x2, y2 = clean
            boxes.append([x1 / width, y1 / height, x2 / width, y2 / height])
            label = self._coco_category_to_label[int(category_id)]
            labels.append(self.names[label])
        return boxes, labels

    def __getitem__(self, index: int) -> Dict[str, object]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(self.images[index]) as source_image:
                    if (
                        Image.MAX_IMAGE_PIXELS is not None
                        and source_image.width * source_image.height
                        > Image.MAX_IMAGE_PIXELS
                    ):
                        raise ValueError("image exceeds the safe pixel limit")
                    image = source_image.convert("RGB")
        except (
            OSError,
            ValueError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise ValueError(
                f"VLM training image cannot be decoded: {self.images[index]}"
            ) from exc
        if self._coco is not None:
            info = self._coco_image_info[index]
            expected_size = (int(info["width"]), int(info["height"]))
            if image.size != expected_size:
                raise ValueError(
                    f"COCO image {self.images[index]} has size {image.size}, "
                    f"but annotations declare {expected_size}."
                )
        boxes, labels = self._load_boxes(index)
        if self.augment and self._rng.random() < self.hflip_p:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes = [[1.0 - x2, y1, 1.0 - x1, y2] for x1, y1, x2, y2 in boxes]
        target = serialize_detections(boxes, labels, self.fmt)
        return {
            "image": image,
            "prompt": self.fmt.detection_prompt,
            "target": target,
        }


def resolve_split_source(data_cfg: Dict, split: str) -> Optional[object]:
    """Return the resolved image source for a training split, if present."""
    return data_cfg.get(split)


def resolve_split_annotation(data_cfg: Dict, split: str) -> Optional[str]:
    """Return the resolved native COCO annotation path for a split, if present."""
    value = data_cfg.get(f"{split}_annotation_file")
    return str(value) if value else None
