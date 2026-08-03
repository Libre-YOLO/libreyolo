"""FCOS box decoding and class-wise NMS.

Postprocessing semantics are derived from pytorch/vision v0.26.0 at commit
336d36e8db990a905498c73933e35231876e28bc (BSD-3-Clause).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torchvision.ops import batched_nms


def _single_image_tensor(
    value: Any,
    *,
    unbatched_dims: int,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == unbatched_dims + 1:
        if tensor.shape[0] != 1:
            raise ValueError(
                f"FCOS postprocess expects one image, got {tensor.shape[0]} in {name}"
            )
        tensor = tensor[0]
    if tensor.ndim != unbatched_dims:
        raise ValueError(
            f"FCOS {name} must have {unbatched_dims} or {unbatched_dims + 1} "
            f"dimensions, got {tuple(tensor.shape)}"
        )
    return tensor


def _empty() -> dict[str, Any]:
    return {
        "num_detections": 0,
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int64),
    }


def _map_labels(
    labels: torch.Tensor,
    class_map: Optional[Mapping[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if class_map is None:
        return labels, torch.ones_like(labels, dtype=torch.bool)
    mapped = torch.full_like(labels, -1)
    for source, target in class_map.items():
        mapped = torch.where(labels == source, mapped.new_tensor(target), mapped)
    return mapped, mapped >= 0


def _decode_boxes(regression: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    anchors = anchors.to(dtype=regression.dtype)
    center_x = 0.5 * (anchors[:, 0] + anchors[:, 2])
    center_y = 0.5 * (anchors[:, 1] + anchors[:, 3])
    width = anchors[:, 2] - anchors[:, 0]
    height = anchors[:, 3] - anchors[:, 1]
    distances = regression * torch.stack((width, height, width, height), dim=-1)
    return torch.stack(
        (
            center_x - distances[:, 0],
            center_y - distances[:, 1],
            center_x + distances[:, 2],
            center_y + distances[:, 3],
        ),
        dim=-1,
    )


def _input_size_value(input_size: int | tuple[int, int]) -> int:
    if isinstance(input_size, (tuple, list)):
        if len(input_size) != 2 or int(input_size[0]) != int(input_size[1]):
            raise ValueError(
                f"FCOS requires a scalar/square input size, got {input_size}"
            )
        return int(input_size[0])
    return int(input_size)


def postprocess(
    outputs: Any,
    conf_thres: float = 0.2,
    iou_thres: float = 0.6,
    original_size: Optional[Tuple[int, int]] = None,
    max_det: int = 100,
    class_map: Optional[Mapping[int, int]] = None,
    input_size: int | tuple[int, int] = 800,
    topk_candidates: int = 1000,
    detections_per_img: int = 100,
    **_unused,
) -> dict[str, Any]:
    """Decode one raw FCOS output into LibreYOLO's canonical detections."""
    if not isinstance(outputs, dict):
        raise TypeError("FCOS postprocess expects a raw output dictionary")
    if original_size is None:
        raise ValueError("original_size is required for FCOS postprocessing")

    logits = _single_image_tensor(
        outputs["cls_logits"], unbatched_dims=2, name="cls_logits"
    )
    regression = _single_image_tensor(
        outputs["bbox_regression"], unbatched_dims=2, name="bbox_regression"
    )
    centerness = _single_image_tensor(
        outputs["bbox_ctrness"], unbatched_dims=2, name="bbox_ctrness"
    )
    anchors = _single_image_tensor(outputs["anchors"], unbatched_dims=2, name="anchors")
    level_sizes = _single_image_tensor(
        outputs["level_sizes"], unbatched_dims=1, name="level_sizes"
    ).to(dtype=torch.int64)

    count = logits.shape[0]
    if not (
        regression.shape == (count, 4)
        and centerness.shape == (count, 1)
        and anchors.shape == (count, 4)
        and int(level_sizes.sum().item()) == count
    ):
        raise ValueError("FCOS raw output tensors have inconsistent anchor dimensions")

    from ..models.fcos.utils import resize_dimensions

    original_width, original_height = original_size
    resized_height, resized_width, _ = resize_dimensions(
        original_height,
        original_width,
        _input_size_value(input_size),
    )

    image_boxes: list[torch.Tensor] = []
    image_scores: list[torch.Tensor] = []
    image_labels: list[torch.Tensor] = []
    offset = 0
    num_classes = logits.shape[-1]
    for raw_level_size in level_sizes.tolist():
        level_size = int(raw_level_size)
        level_slice = slice(offset, offset + level_size)
        level_logits = logits[level_slice]
        level_regression = regression[level_slice]
        level_centerness = centerness[level_slice]
        level_anchors = anchors[level_slice]
        offset += level_size

        scores = torch.sqrt(
            torch.sigmoid(level_logits) * torch.sigmoid(level_centerness)
        ).flatten()
        kept = scores > float(conf_thres)
        candidate_indices = torch.where(kept)[0]
        scores = scores[kept]
        num_topk = min(int(candidate_indices.numel()), int(topk_candidates))
        if num_topk == 0:
            continue
        scores, ordering = scores.topk(num_topk)
        candidate_indices = candidate_indices[ordering]
        anchor_indices = torch.div(
            candidate_indices,
            num_classes,
            rounding_mode="floor",
        )
        labels = candidate_indices % num_classes
        boxes = _decode_boxes(
            level_regression[anchor_indices],
            level_anchors[anchor_indices],
        )
        boxes[:, 0::2].clamp_(0, resized_width)
        boxes[:, 1::2].clamp_(0, resized_height)
        image_boxes.append(boxes)
        image_scores.append(scores)
        image_labels.append(labels)

    if not image_boxes:
        return _empty()
    boxes = torch.cat(image_boxes)
    scores = torch.cat(image_scores)
    labels = torch.cat(image_labels)

    keep = batched_nms(boxes, scores, labels, float(iou_thres))
    limit = min(max(0, int(max_det)), max(0, int(detections_per_img)))
    keep = keep[:limit]
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    labels, mapped_mask = _map_labels(labels, class_map)
    boxes = boxes[mapped_mask]
    scores = scores[mapped_mask]
    labels = labels[mapped_mask]
    if not boxes.numel():
        return _empty()

    boxes = boxes.clone()
    boxes[:, 0::2] *= original_width / resized_width
    boxes[:, 1::2] *= original_height / resized_height
    boxes[:, 0::2].clamp_(0, original_width)
    boxes[:, 1::2].clamp_(0, original_height)

    boxes_array = boxes.detach().cpu().to(torch.float32).numpy()
    scores_array = scores.detach().cpu().to(torch.float32).numpy()
    classes_array = labels.detach().cpu().to(torch.int64).numpy()
    return {
        "num_detections": int(boxes_array.shape[0]),
        "boxes": boxes_array,
        "scores": scores_array,
        "classes": classes_array,
    }


__all__ = ["postprocess"]
