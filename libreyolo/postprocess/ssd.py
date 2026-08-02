# Derived from torchvision v0.26.0 SSD postprocessing and DefaultBoxGenerator.
# Upstream commit: 336d36e8db990a905498c73933e35231876e28bc
# License: BSD-3-Clause
"""SSD300 default-box decoding and class-wise NMS."""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torchvision.ops import batched_nms


_FEATURE_SIZES = (38, 19, 10, 5, 3, 1)
_ASPECT_RATIOS = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
_SCALES = (0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05)
_STEPS = (8, 16, 32, 64, 100, 300)
_BOX_CODER_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
_BBOX_XFORM_CLIP = math.log(1000.0 / 16)


@lru_cache(maxsize=4)
def _normalized_default_boxes(
    input_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the pinned generator's CPU ``cxcywh`` inventory once."""
    levels: list[torch.Tensor] = []
    for level, feature_size in enumerate(_FEATURE_SIZES):
        scale = _SCALES[level]
        next_scale = _SCALES[level + 1]
        pairs = [[scale, scale], [math.sqrt(scale * next_scale)] * 2]
        for aspect_ratio in _ASPECT_RATIOS[level]:
            root = math.sqrt(aspect_ratio)
            pairs.extend(
                [
                    [scale * root, scale / root],
                    [scale / root, scale * root],
                ]
            )

        # torchvision constructs these pairs as CPU float32 at generator init,
        # independent of the later feature dtype.
        wh_pairs = torch.as_tensor(pairs, dtype=torch.float32).clamp_(0, 1)
        cells_per_step = input_size / _STEPS[level]
        shifts = (torch.arange(feature_size) + 0.5) / cells_per_step
        shifts = shifts.to(dtype=dtype)
        shift_y, shift_x = torch.meshgrid(shifts, shifts, indexing="ij")
        centers = torch.stack((shift_x, shift_y), dim=-1).reshape(-1, 1, 2)
        centers = centers.expand(-1, len(pairs), -1).reshape(-1, 2)
        sizes = wh_pairs.repeat(feature_size * feature_size, 1)
        levels.append(torch.cat((centers, sizes), dim=-1))
    return torch.cat(levels, dim=0)


def _default_boxes(
    *,
    device: torch.device,
    dtype: torch.dtype,
    input_size: int = 300,
) -> torch.Tensor:
    """Generate SSD300's 8,732 default boxes in ``xyxy`` pixel space."""
    normalized = _normalized_default_boxes(input_size, dtype).to(device)
    xy_size = torch.tensor([input_size, input_size], device=device)
    half_sizes = normalized[:, 2:] * 0.5
    return torch.cat(
        (
            (normalized[:, :2] - half_sizes) * xy_size,
            (normalized[:, :2] + half_sizes) * xy_size,
        ),
        dim=-1,
    )


def _decode_boxes(
    regression: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """Apply SSD's weighted center/size box transform."""
    anchors = anchors.to(dtype=regression.dtype)
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    center_x = anchors[:, 0] + 0.5 * widths
    center_y = anchors[:, 1] + 0.5 * heights

    dx = regression[:, 0] / _BOX_CODER_WEIGHTS[0]
    dy = regression[:, 1] / _BOX_CODER_WEIGHTS[1]
    dw = (regression[:, 2] / _BOX_CODER_WEIGHTS[2]).clamp(max=_BBOX_XFORM_CLIP)
    dh = (regression[:, 3] / _BOX_CODER_WEIGHTS[3]).clamp(max=_BBOX_XFORM_CLIP)

    predicted_center_x = dx * widths + center_x
    predicted_center_y = dy * heights + center_y
    predicted_width = dw.exp() * widths
    predicted_height = dh.exp() * heights
    return torch.stack(
        (
            predicted_center_x - 0.5 * predicted_width,
            predicted_center_y - 0.5 * predicted_height,
            predicted_center_x + 0.5 * predicted_width,
            predicted_center_y + 0.5 * predicted_height,
        ),
        dim=-1,
    )


def _empty_result() -> dict[str, Any]:
    return {
        "num_detections": 0,
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int64),
    }


def postprocess(
    outputs: Any,
    conf_thres: float = 0.01,
    iou_thres: float = 0.45,
    original_size: Optional[Tuple[int, int]] = None,
    max_det: int = 200,
    class_map: Optional[Mapping[int, int]] = None,
    input_size: int = 300,
    topk_candidates: int = 400,
    **_unused,
) -> dict[str, Any]:
    """Decode one raw SSD head into LibreYOLO's detection dictionary."""
    max_det = min(max(0, int(max_det)), 200)
    if not isinstance(outputs, Mapping):
        raise TypeError("SSD postprocess expects a raw output mapping")
    regression = outputs["bbox_regression"]
    logits = outputs["cls_logits"]
    if regression.ndim == 3 and regression.shape[0] == 1:
        regression = regression[0]
        logits = logits[0]
    if regression.ndim != 2 or logits.ndim != 2:
        raise ValueError("SSD postprocess expects one image of raw head outputs")
    if regression.shape != (8732, 4) or logits.shape[0] != 8732:
        raise ValueError(
            "SSD300 raw outputs must have 8,732 anchors and four box offsets"
        )

    anchors = _default_boxes(
        device=regression.device,
        dtype=regression.dtype,
        input_size=input_size,
    )
    boxes = _decode_boxes(regression, anchors)
    boxes[:, 0::2].clamp_(0, input_size)
    boxes[:, 1::2].clamp_(0, input_size)
    probabilities = logits.softmax(dim=-1)

    if class_map is None:
        source_labels = range(1, logits.shape[-1])
    else:
        source_labels = sorted(
            label for label in class_map if 0 < label < logits.shape[-1]
        )

    image_boxes: list[torch.Tensor] = []
    image_scores: list[torch.Tensor] = []
    image_labels: list[torch.Tensor] = []
    for source_label in source_labels:
        scores = probabilities[:, source_label]
        keep = scores > conf_thres
        scores = scores[keep]
        selected_boxes = boxes[keep]
        candidate_count = min(int(scores.numel()), max(0, int(topk_candidates)))
        scores, indices = scores.topk(candidate_count)
        selected_boxes = selected_boxes[indices]
        target_label = (
            source_label - 1 if class_map is None else int(class_map[source_label])
        )
        image_boxes.append(selected_boxes)
        image_scores.append(scores)
        image_labels.append(
            torch.full(
                (candidate_count,),
                target_label,
                dtype=torch.int64,
                device=logits.device,
            )
        )

    boxes = torch.cat(image_boxes, dim=0)
    scores = torch.cat(image_scores, dim=0)
    labels = torch.cat(image_labels, dim=0)
    if boxes.numel() == 0 or max_det == 0:
        return _empty_result()

    finite = torch.isfinite(boxes).all(dim=-1) & torch.isfinite(scores)
    boxes = boxes[finite]
    scores = scores[finite]
    labels = labels[finite]
    if boxes.numel() == 0:
        return _empty_result()

    nms_boxes = boxes.float() if boxes.dtype == torch.float16 else boxes
    nms_scores = scores.float() if scores.dtype == torch.float16 else scores
    keep = batched_nms(nms_boxes, nms_scores, labels, iou_thres)
    keep = keep[:max_det]
    boxes = boxes[keep].clone()
    scores = scores[keep]
    labels = labels[keep]

    if original_size is not None and boxes.numel():
        original_width, original_height = original_size
        boxes[:, [0, 2]] *= original_width / input_size
        boxes[:, [1, 3]] *= original_height / input_size
        boxes[:, 0::2].clamp_(0, original_width)
        boxes[:, 1::2].clamp_(0, original_height)

    return {
        "num_detections": int(boxes.shape[0]),
        "boxes": boxes.detach().cpu().to(torch.float32).numpy(),
        "scores": scores.detach().cpu().to(torch.float32).numpy(),
        "classes": labels.detach().cpu().to(torch.int64).numpy(),
    }


__all__ = ["postprocess"]
