# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Ported for LibreYOLO from OpenSenseNova/SenseNova-Vision
# (commit 12ccd96e32b32967a11cacb6c5bd5fe3a555fc0c, Apache-2.0), files
# utils/parsing_output.py, utils/__init__.py and utils/mask.py, trimmed and
# adapted to emit the LibreYOLO detection-dict shapes. Everything here is pure
# (no torch, no model) so it can be unit-tested offline.
#
# Output grammars decoded here:
# - boxes:     ``<p>label</p><bbox>[x0,y0,x1,y1]</bbox>`` (normalized [0, 1])
# - points:    ``<p>label</p><point>[x,y]</point>``
# - keypoints: ``<p>label</p><bbox>...</bbox>name<kpt>[x,y]</kpt>...`` with
#              optional ``<ins>...</ins>`` instance spans and
#              ``<kpt>unvisible</kpt>`` for occluded joints
# - panoptic:  an RGB mask image plus a rich caption of
#              ``<p>phrase-idx<color>(R,G,B)</color></p>`` tags

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

HUMAN_KEYPOINT_NAMES = [
    "nose",
    "left eye",
    "right eye",
    "left ear",
    "right ear",
    "left shoulder",
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left hip",
    "right hip",
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
]
ANIMAL_KEYPOINT_NAMES = [
    "left eye",
    "right eye",
    "nose",
    "neck",
    "root of tail",
    "left shoulder",
    "left elbow",
    "left front paw",
    "right shoulder",
    "right elbow",
    "right front paw",
    "left hip",
    "left knee",
    "left back paw",
    "right hip",
    "right knee",
    "right back paw",
]

_COLOR_TAG = re.compile(
    r"<p>\s*(.*?)\s*<color>\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*</color>\s*</p>",
    flags=re.IGNORECASE | re.DOTALL,
)


def normalize_category(category: str) -> str:
    """Normalize category names for matching."""
    return (
        str(category)
        .replace("-merged", "")
        .replace("-other", "")
        .replace("-stuff", "")
        .replace("-negative", "")
        .replace("-", " ")
        .lower()
        .strip()
    )


def clip_normalized(values) -> List[float]:
    """Clip normalized coordinate values into the half-open visual range."""
    return [max(0.0, min(0.999, float(v))) for v in values]


def _iter_label_blocks(text: str):
    """Yield (label, rest) for each ``<p>label</p>...`` block in the text."""
    s = str(text or "").strip().rstrip(" .\n")
    for part in s.split("<p>")[1:]:
        if "</p>" not in part:
            continue
        cat_end = part.find("</p>")
        yield part[:cat_end].strip(), part[cat_end + len("</p>") :]


def parse_tagged_boxes(text: str, normalize_labels: bool = True) -> Dict[str, List[List[float]]]:
    """Parse ``<p>label</p><bbox>[x0,y0,x1,y1]</bbox>`` detection text.

    Returns ``{label: [[x0, y0, x1, y1], ...]}`` with coordinates kept in
    normalized space and clipped to ``[0, 0.999]``. OCR keeps original label
    case via ``normalize_labels=False``.
    """
    results: Dict[str, List[List[float]]] = {}
    for category, rest in _iter_label_blocks(text):
        boxes = []
        while "<bbox>[" in rest:
            start = rest.find("<bbox>[")
            end = rest.find("]</bbox>")
            if start == -1 or end == -1:
                break
            coord_str = rest[start + len("<bbox>[") : end]
            rest = rest[end + len("]</bbox>") :]
            try:
                coords = [float(x.strip()) for x in coord_str.split(",")]
            except ValueError:
                continue
            if len(coords) == 4:
                boxes.append(clip_normalized(coords))
        if boxes:
            label = normalize_category(category) if normalize_labels else category
            results.setdefault(label, []).extend(boxes)
    return results


def parse_tagged_points(text: str) -> Dict[str, List[List[float]]]:
    """Parse ``<p>label</p><point>[x,y]</point>`` point-detection text."""
    results: Dict[str, List[List[float]]] = {}
    for category, rest in _iter_label_blocks(text):
        points = []
        while "<point>[" in rest:
            start = rest.find("<point>[")
            end = rest.find("]</point>")
            if start == -1 or end == -1:
                break
            coord_str = rest[start + len("<point>[") : end]
            rest = rest[end + len("]</point>") :]
            try:
                coords = [float(x.strip()) for x in coord_str.split(",")]
            except ValueError:
                continue
            if len(coords) == 2:
                points.append(clip_normalized(coords))
        if points:
            results.setdefault(normalize_category(category), []).extend(points)
    return results


def parse_tagged_keypoints(text: str) -> Dict[str, List[dict]]:
    """Parse keypoint text grouped by category and instance.

    Expected structure per instance:
    ``<p>person</p><bbox>[...]</bbox>left shoulder<kpt>[x,y]</kpt>...``.
    ``<ins>...</ins>`` wrappers are tolerated and stripped. Invisible keypoints
    encoded as ``<kpt>unvisible</kpt>`` are stored as ``[-1, -1]``.

    Returns ``{category: [{"bbox": [...] or missing, "keypoints": {name: [x, y]}}]}``
    with all coordinates normalized.
    """
    results: Dict[str, List[dict]] = {}
    for category, rest in _iter_label_blocks(text):
        rest = rest.replace("<ins>", "").replace("</ins>", "")

        bbox = None
        if "<bbox>[" in rest:
            start = rest.find("<bbox>[")
            end = rest.find("]</bbox>")
            if start != -1 and end != -1:
                coord_str = rest[start + len("<bbox>[") : end]
                try:
                    coords = [float(x.strip()) for x in coord_str.split(",")]
                    if len(coords) == 4:
                        bbox = clip_normalized(coords)
                except ValueError:
                    pass
                rest = rest[:start] + rest[end + len("]</bbox>") :]

        keypoints: Dict[str, List[float]] = {}
        while "<kpt>" in rest:
            start = rest.find("<kpt>")
            end = rest.find("</kpt>", start)
            if start == -1 or end == -1:
                break
            keypoint_name = rest[:start].strip().strip(",.;:").strip().lower()
            content = rest[start + len("<kpt>") : end].strip()
            rest = rest[end + len("</kpt>") :]
            if not keypoint_name:
                continue
            if content == "unvisible":
                keypoints[keypoint_name] = [-1.0, -1.0]
                continue
            try:
                coords = [float(x.strip()) for x in content.strip("[]").split(",")]
            except ValueError:
                continue
            if len(coords) == 2:
                keypoints[keypoint_name] = clip_normalized(coords)

        if category:
            instance = {"keypoints": keypoints}
            if bbox is not None:
                instance["bbox"] = bbox
            results.setdefault(normalize_category(category), []).append(instance)
    return results


def keypoint_instance_to_array(
    instance: dict, skeleton: List[str], width: int, height: int
) -> np.ndarray:
    """Convert one parsed keypoint instance into a ``(K, 3)`` pixel array.

    Rows follow ``skeleton`` order as ``(x, y, visibility)``, with missing or
    invisible joints emitted as ``(0, 0, 0)`` in the LibreYOLO keypoint
    convention.
    """
    arr = np.zeros((len(skeleton), 3), dtype=np.float32)
    keypoints = instance.get("keypoints", {})
    for idx, name in enumerate(skeleton):
        coords = keypoints.get(name)
        if coords is None:
            continue
        x, y = float(coords[0]), float(coords[1])
        if x < 0 or y < 0:
            continue
        arr[idx] = (x * width, y * height, 1.0)
    return arr


def get_clean_caption(caption: str) -> str:
    """Strip color/markup tags from a rich caption, keeping the prose."""
    caption = re.sub(r"<.*?>", "", str(caption or ""))
    caption = re.sub(r"\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)", "", caption)
    return " ".join(caption.split()).strip("'").strip()


def extract_caption_instances(caption: str) -> List[Tuple[str, Tuple[int, int, int]]]:
    """Return ordered (phrase, rgb) entries from colored caption tags.

    Preserves repeated phrases, such as multiple instances of the same
    category, unlike a phrase-to-color dict.
    """
    instances = []
    for match in _COLOR_TAG.finditer(str(caption or "")):
        phrase = re.sub(r"<.*?>", "", match.group(1)).strip()
        rgb = tuple(int(match.group(i)) for i in range(2, 5))
        instances.append((phrase, rgb))
    return instances


def nearest_color_class_mask(image: Image.Image, class_colors: np.ndarray) -> np.ndarray:
    """Convert an RGB mask to a class-index mask by nearest prompt color.

    Black pixels are always assigned to the background index
    ``len(class_colors)``.
    """
    img_np = np.array(image.convert("RGB"), dtype=np.float32)
    class_colors = np.asarray(class_colors, dtype=np.float32)
    if class_colors.ndim != 2 or class_colors.shape[1] != 3:
        raise ValueError(f"class_colors must have shape [N, 3], got {class_colors.shape}")

    mask_black = np.all(img_np == 0, axis=-1)
    h, w, _ = img_np.shape
    pixels = img_np.reshape(-1, 3)

    pred_class = np.empty((pixels.shape[0],), dtype=np.int32)
    chunk_size = 262144
    for start in range(0, pixels.shape[0], chunk_size):
        end = min(start + chunk_size, pixels.shape[0])
        diff = pixels[start:end, None, :] - class_colors[None, :, :]
        dist = np.sum(diff * diff, axis=-1)
        pred_class[start:end] = np.argmin(dist, axis=1)

    mask = pred_class.reshape(h, w)
    mask[mask_black] = len(class_colors)
    return mask.astype(np.int32, copy=False)


def panoptic_from_mask_and_caption(
    mask_image: Image.Image,
    caption: str,
    names: Dict[int, str],
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, List[dict], Dict[int, str]]:
    """Assemble a panoptic id map from a generated color mask and rich caption.

    The model emits an RGB mask plus caption tags such as
    ``<p>person-0<color>(255,0,0)</color></p>``; each color keys one segment.
    Segment ids are ``(category_id + 1) * 1000 + local_instance_index``; the
    +1 keeps id 0 reserved for void pixels even though LibreYOLO category ids
    are 0-based. Phrases outside ``names`` are registered as new categories so
    open-vocabulary output is not dropped.

    Returns ``(id_map, segments_info, names)`` where ``names`` includes any
    newly registered categories.
    """
    width, height = image_size
    if mask_image.size != (width, height):
        mask_image = mask_image.resize((width, height), resample=Image.NEAREST)

    instances = extract_caption_instances(caption)
    phrases = [phrase for phrase, _ in instances]
    if instances:
        class_colors = np.array([rgb for _, rgb in instances], dtype=np.float32)
    else:
        class_colors = np.array([(255, 255, 255)], dtype=np.float32)
    class_mask = nearest_color_class_mask(mask_image, class_colors)

    names = dict(names)
    name_to_id = {normalize_category(v): int(k) for k, v in names.items()}
    max_category_id = max((int(k) for k in names), default=-1)

    id_map = np.zeros((height, width), dtype=np.int64)
    segments_info: List[dict] = []
    occurrence: Dict[int, int] = {}
    used_ids = set()

    for instance_idx, phrase in enumerate(phrases):
        base_name, separator, suffix = phrase.rpartition("-")
        if separator and suffix.isdigit():
            local_idx: Optional[int] = int(suffix)
        else:
            base_name = phrase
            local_idx = None

        normalized = normalize_category(base_name)
        if not normalized:
            continue
        category_id = name_to_id.get(normalized)
        if category_id is None:
            max_category_id += 1
            category_id = max_category_id
            names[category_id] = base_name
            name_to_id[normalized] = category_id

        segment_mask = class_mask == instance_idx
        if not np.any(segment_mask):
            continue
        if local_idx is None:
            local_idx = occurrence.get(category_id, 0)
        panoptic_id = (category_id + 1) * 1000 + local_idx
        while panoptic_id in used_ids:
            local_idx += 1
            panoptic_id = (category_id + 1) * 1000 + local_idx
        occurrence[category_id] = max(occurrence.get(category_id, 0), local_idx + 1)
        used_ids.add(panoptic_id)
        id_map[segment_mask] = panoptic_id
        segments_info.append({"id": panoptic_id, "category_id": category_id, "score": 1.0})

    return id_map, segments_info, names


def binary_mask_from_image(
    mask_image: Image.Image, image_size: Tuple[int, int], threshold: int = 127
) -> np.ndarray:
    """Threshold a generated mask image into a boolean array at image size."""
    width, height = image_size
    if mask_image.size != (width, height):
        mask_image = mask_image.resize((width, height), resample=Image.NEAREST)
    arr = np.array(mask_image.convert("L"), dtype=np.uint8)
    return arr > threshold


def depth_from_image(depth_image: Image.Image, image_size: Tuple[int, int]) -> np.ndarray:
    """Convert a generated grayscale depth image into a relative map in [0, 1].

    The model draws closer pixels brighter, which matches LibreYOLO's
    relative inverse-depth convention directly (larger value = closer).
    """
    width, height = image_size
    if depth_image.size != (width, height):
        depth_image = depth_image.resize((width, height), resample=Image.BILINEAR)
    arr = np.asarray(depth_image.convert("L"), dtype=np.float32)
    return arr / 255.0


def mask_to_xyxy(mask: np.ndarray) -> Optional[List[float]]:
    """Tight pixel bounding box of a boolean mask, or None when empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
