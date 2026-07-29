# SPDX-License-Identifier: MIT

"""Decode MODUS image and codebook outputs into LibreYOLO payloads."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image

# COCO category-id (the released token id) to contiguous COCO-80 id.
COCO91_TO_COCO80 = {
    category_id: contiguous
    for contiguous, category_id in enumerate(
        (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            27,
            28,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            67,
            70,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
        )
    )
}


def _ids(values: torch.Tensor | Iterable[int]) -> list[int]:
    if isinstance(values, torch.Tensor):
        values = values.detach().reshape(-1).cpu().tolist()
    return [int(value) for value in values]


def _probabilities(
    values: Optional[torch.Tensor | Iterable[float]],
) -> Optional[list[float]]:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        values = values.detach().float().reshape(-1).cpu().tolist()
    return [float(value) for value in values]


def decode_cocodet_tokens(
    token_ids: torch.Tensor | Iterable[int],
    *,
    x1_base: int,
    y1_base: int,
    x2_base: int,
    y2_base: int,
    cls_base: int,
    start_token: Optional[int] = None,
    end_token: Optional[int] = None,
    step_probs: Optional[torch.Tensor | Iterable[float]] = None,
) -> list[dict]:
    """Decode ``x1,y1,x2,y2,class`` groups from constrained AR output."""
    values = _ids(token_ids)
    if start_token is not None and values and values[0] == int(start_token):
        values = values[1:]
    probs = _probabilities(step_probs)
    boxes = []
    cursor = 0
    ranges = (
        (x1_base, x1_base + 1000),
        (y1_base, y1_base + 1000),
        (x2_base, x2_base + 1000),
        (y2_base, y2_base + 1000),
        (cls_base, cls_base + 91),
    )
    while cursor < len(values):
        if end_token is not None and values[cursor] == int(end_token):
            break
        group = values[cursor : cursor + 5]
        if len(group) < 5 or any(
            not lower <= token < upper for token, (lower, upper) in zip(group, ranges)
        ):
            break
        category_id = group[4] - cls_base
        class_id = COCO91_TO_COCO80.get(category_id)
        if class_id is not None:
            score_values = probs[cursor : cursor + 5] if probs is not None else ()
            boxes.append(
                {
                    "bbox": [
                        (group[0] - x1_base) / 1000.0,
                        (group[1] - y1_base) / 1000.0,
                        (group[2] - x2_base) / 1000.0,
                        (group[3] - y2_base) / 1000.0,
                    ],
                    "label": class_id,
                    "score": min(score_values) if score_values else 1.0,
                }
            )
        cursor += 5
    return boxes


def decode_grounding_tokens(
    token_ids: torch.Tensor | Iterable[int],
    *,
    x1_base: int,
    y1_base: int,
    x2_base: int,
    y2_base: int,
    label: str,
    start_token: Optional[int] = None,
    end_token: Optional[int] = None,
    step_probs: Optional[torch.Tensor | Iterable[float]] = None,
) -> list[dict]:
    """Decode coordinate-only grounding groups and attach the requested phrase."""
    values = _ids(token_ids)
    if start_token is not None and values and values[0] == int(start_token):
        values = values[1:]
    probs = _probabilities(step_probs)
    boxes = []
    cursor = 0
    ranges = (
        (x1_base, x1_base + 1000),
        (y1_base, y1_base + 1000),
        (x2_base, x2_base + 1000),
        (y2_base, y2_base + 1000),
    )
    while cursor < len(values):
        if end_token is not None and values[cursor] == int(end_token):
            break
        group = values[cursor : cursor + 4]
        if len(group) < 4 or any(
            not lower <= token < upper for token, (lower, upper) in zip(group, ranges)
        ):
            break
        score_values = probs[cursor : cursor + 4] if probs is not None else ()
        boxes.append(
            {
                "bbox": [
                    (group[0] - x1_base) / 1000.0,
                    (group[1] - y1_base) / 1000.0,
                    (group[2] - x2_base) / 1000.0,
                    (group[3] - y2_base) / 1000.0,
                ],
                "label": str(label),
                "score": min(score_values) if score_values else 1.0,
            }
        )
        cursor += 4
    return boxes


def detection_payload(
    items: Sequence[Mapping],
    original_size: tuple[int, int],
    *,
    conf: float = 0.0,
    iou: Optional[float] = None,
    max_det: int = 300,
    classes: Optional[Sequence[int]] = None,
) -> dict:
    """Scale normalized boxes to pixels and apply standard filters."""
    width, height = original_size
    allowed = None if classes is None else {int(value) for value in classes}
    boxes, normalized_boxes, scores, class_ids = [], [], [], []
    candidates = []

    def overlap(a, b):
        inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        intersection = inter_w * inter_h
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    for item in items:
        try:
            class_id = int(item["label"])
            score = float(item.get("score", 1.0))
            x1, y1, x2, y2 = (float(value) for value in item["bbox"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            class_id < 0
            or score < conf
            or (allowed is not None and class_id not in allowed)
        ):
            continue
        x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
        y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
        if x2 <= x1 or y2 <= y1:
            continue
        normalized = (x1, y1, x2, y2)
        candidates.append((score, class_id, normalized))

    # Generated order is not a confidence guarantee. Apply NMS and max_det in
    # descending confidence order, matching the standard detection contract.
    for score, class_id, normalized in sorted(
        candidates, key=lambda candidate: candidate[0], reverse=True
    ):
        if iou is not None and any(
            class_id == kept_class and overlap(normalized, kept_box) > iou
            for kept_box, kept_class in zip(normalized_boxes, class_ids)
        ):
            continue
        x1, y1, x2, y2 = normalized
        normalized_boxes.append(normalized)
        boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
        scores.append(score)
        class_ids.append(class_id)
        if len(boxes) >= max_det:
            break
    return {
        "boxes": boxes,
        "scores": scores,
        "classes": class_ids,
        "num_detections": len(boxes),
    }


def image_to_payload(
    image: Image.Image,
    target: str,
    original_size: tuple[int, int],
) -> dict:
    """Convert a generated VAE image to depth, normal, or edge payload."""
    width, height = original_size
    if image.size != (width, height):
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    if target == "depth":
        return {"depth": np.asarray(image.convert("L"), dtype=np.float32) / 255.0}
    if target in {"canny", "samedge"}:
        return {"edges": np.asarray(image.convert("L"), dtype=np.float32) / 255.0}
    if target == "normal":
        # MODUS was trained on the usual positive-z normal raster. LibreYOLO's
        # public camera-frame contract orients normals toward the camera, so the
        # complete vector (not only z) must be reversed at the boundary.
        normals = -(np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        valid = np.isfinite(normals).all(axis=-1, keepdims=True) & (norms > 1e-6)
        fallback = np.zeros_like(normals)
        fallback[..., 2] = -1.0
        normals = np.where(valid, normals / np.maximum(norms, 1e-6), fallback)
        return {"normal": normals.astype(np.float32)}
    raise ValueError(f"Target {target!r} is not a dense image modality.")


def input_to_image(value, modality: str) -> Image.Image:
    """Encode a public input payload as the RGB image representation MODUS saw."""
    if isinstance(value, Image.Image):
        if modality == "normal":
            # PIL/uint8 normal rasters use LibreYOLO's public ``(n + 1) / 2``
            # visualization. Reverse every component for MODUS's positive-z
            # training representation.
            public_raster = np.asarray(value.convert("RGB"), dtype=np.uint8)
            return Image.fromarray(255 - public_raster, mode="RGB")
        if modality not in {"rgb", "depth", "canny"}:
            raise ValueError(f"Cannot encode public modality {modality!r} as an image.")
        return value.convert("RGB")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if modality == "normal":
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("normal input must have shape [H,W,3].")
        if np.issubdtype(array.dtype, np.floating):
            normals = np.asarray(array, dtype=np.float32)
            if not np.isfinite(normals).all():
                raise ValueError("normal input contains NaN or infinite values.")
            if normals.size and (normals.min() < -1.0001 or normals.max() > 1.0001):
                raise ValueError("floating-point normal input must be in [-1, 1].")
            norms = np.linalg.norm(normals, axis=-1, keepdims=True)
            if np.any(norms <= 1e-6):
                raise ValueError("normal input contains a zero-length vector.")
            modus_normals = -(normals / norms)
            array = ((np.clip(modus_normals, -1.0, 1.0) + 1.0) * 127.5).round()
        else:
            if array.size and (array.min() < 0 or array.max() > 255):
                raise ValueError("integer normal rasters must be in [0, 255].")
            array = 255 - np.asarray(array, dtype=np.uint8)
        return Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    if modality in {"depth", "canny"}:
        if array.ndim == 3 and array.shape[-1] in (3, 4):
            if np.issubdtype(array.dtype, np.floating):
                if not np.isfinite(array).all():
                    raise ValueError(
                        f"{modality} input contains NaN or infinite values."
                    )
                if (
                    array.size
                    and float(array.min()) >= 0.0
                    and float(array.max()) <= 1.0
                ):
                    array = (array * 255.0).round()
                elif array.size and (
                    float(array.min()) < 0.0 or float(array.max()) > 255.0
                ):
                    raise ValueError(
                        f"floating-point {modality} rasters must be in [0, 1] "
                        "or [0, 255]."
                    )
            elif array.size and (array.min() < 0 or array.max() > 255):
                raise ValueError(f"integer {modality} rasters must be in [0, 255].")
            return Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB")
        if array.ndim != 2:
            raise ValueError(f"{modality} input must have shape [H,W].")
        if np.issubdtype(array.dtype, np.floating):
            array = np.asarray(array, dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError(f"{modality} input contains NaN or infinite values.")
            lo = float(array.min()) if array.size else 0.0
            hi = float(array.max()) if array.size else 0.0
            if modality == "canny" and (lo < 0.0 or hi > 1.0):
                raise ValueError("floating-point canny input must be in [0, 1].")
            if modality == "depth" and (lo < 0.0 or hi > 1.0):
                array = (array - lo) / (hi - lo) if hi > lo else np.zeros_like(array)
            array = (array * 255.0).round()
        elif array.size:
            lo, hi = int(array.min()), int(array.max())
            if modality == "canny" and (lo < 0 or hi > 255):
                raise ValueError("integer canny rasters must be in [0, 255].")
            if modality == "depth" and (lo < 0 or hi > 255):
                work = np.asarray(array, dtype=np.float32)
                array = (
                    (work - lo) * (255.0 / (hi - lo))
                    if hi > lo
                    else np.zeros_like(work)
                )
            elif hi <= 1:
                array = np.asarray(array, dtype=np.float32) * 255.0
        return Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").convert(
            "RGB"
        )
    if modality == "rgb":
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError("rgb input must have shape [H,W,3] or [H,W,4].")
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError("rgb input contains NaN or infinite values.")
            lo = float(array.min()) if array.size else 0.0
            hi = float(array.max()) if array.size else 0.0
            if lo < 0.0 or hi > 255.0:
                raise ValueError(
                    "floating-point rgb input must be in [0, 1] or [0, 255]."
                )
            if hi <= 1.0:
                array = (array * 255.0).round()
            else:
                array = array.round()
        elif array.size and (array.min() < 0 or array.max() > 255):
            raise ValueError("integer rgb input must be in [0, 255].")
        return Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB")
    raise ValueError(f"Cannot encode public modality {modality!r} as an image.")
