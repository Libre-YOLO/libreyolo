"""Mask R-CNN postprocessing for aligned two-stage instance masks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Tuple

import numpy as np
import torch

from .faster_rcnn import _map_labels


def _unpack(
    outputs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Normalize native and exported output containers."""
    if isinstance(outputs, list):
        if len(outputs) != 1:
            raise ValueError(
                "Mask R-CNN postprocess expects one image, "
                f"received {len(outputs)}"
            )
        outputs = outputs[0]

    masks = None
    if isinstance(outputs, dict):
        boxes = outputs["boxes"]
        scores = outputs["scores"]
        labels = outputs.get("labels", outputs.get("classes"))
        masks = outputs.get("masks")
        if labels is None:
            raise KeyError("Mask R-CNN output has no labels/classes field")
    elif isinstance(outputs, (tuple, list)) and len(outputs) in {3, 4}:
        boxes, scores, labels = outputs[:3]
        if len(outputs) == 4:
            masks = outputs[3]
    else:
        raise TypeError(
            "Mask R-CNN output must be a detection dict, a one-item list, "
            "or a (boxes, scores, labels[, masks]) tuple"
        )

    boxes = torch.as_tensor(boxes)
    scores = torch.as_tensor(scores)
    labels = torch.as_tensor(labels).to(dtype=torch.int64)
    if boxes.dim() == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if scores.dim() == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if labels.dim() == 2 and labels.shape[0] == 1:
        labels = labels[0]
    if masks is not None:
        masks = torch.as_tensor(masks)
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
    return boxes, scores, labels, masks


def postprocess(
    outputs: Any,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    original_size: Optional[Tuple[int, int]] = None,
    max_det: int = 300,
    class_map: Optional[Mapping[int, int]] = None,
    *,
    include_masks: bool = True,
    mask_threshold: float = 0.5,
    **_unused,
) -> dict:
    """Return boxes and optional boolean masks in contiguous class space."""
    del iou_thres
    boxes, scores, labels, masks = _unpack(outputs)
    if include_masks:
        if masks is None:
            raise KeyError("Mask R-CNN segment output has no masks field")
        if masks.shape[0] != boxes.shape[0]:
            raise ValueError("Mask R-CNN boxes and masks are not row-aligned")

    mapped_labels, mapped_mask = _map_labels(labels, class_map)
    keep = mapped_mask & (scores > conf_thres)
    boxes = boxes[keep]
    scores = scores[keep]
    mapped_labels = mapped_labels[keep]
    if masks is not None:
        masks = masks[keep]

    order = torch.argsort(scores, descending=True)[: max(0, int(max_det))]
    boxes = boxes[order].clone()
    scores = scores[order]
    mapped_labels = mapped_labels[order]
    if masks is not None:
        masks = masks[order]

    if original_size is not None and boxes.numel():
        width, height = original_size
        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)

    boxes_array = boxes.detach().cpu().to(torch.float32).numpy()
    scores_array = scores.detach().cpu().to(torch.float32).numpy()
    classes_array = mapped_labels.detach().cpu().to(torch.int64).numpy()
    if not boxes_array.size:
        boxes_array = np.zeros((0, 4), dtype=np.float32)
        scores_array = np.zeros((0,), dtype=np.float32)
        classes_array = np.zeros((0,), dtype=np.int64)

    result = {
        "num_detections": int(boxes_array.shape[0]),
        "boxes": boxes_array,
        "scores": scores_array,
        "classes": classes_array,
    }
    if include_masks:
        assert masks is not None
        mask_array = (masks >= mask_threshold).detach().cpu().numpy()
        if mask_array.shape[0] == 0 and original_size is not None:
            width, height = original_size
            mask_array = np.zeros((0, height, width), dtype=np.bool_)
        result["masks"] = mask_array.astype(np.bool_, copy=False)
    return result
