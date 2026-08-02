"""Faster R-CNN postprocessing for already-NMSed two-stage detections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Tuple

import numpy as np
import torch


def _unpack(outputs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize native and exported output containers to three tensors."""
    if isinstance(outputs, list):
        if len(outputs) != 1:
            raise ValueError(
                "Faster R-CNN postprocess expects one image, "
                f"received {len(outputs)}"
            )
        outputs = outputs[0]
    if isinstance(outputs, dict):
        boxes = outputs["boxes"]
        scores = outputs["scores"]
        labels = outputs.get("labels", outputs.get("classes"))
        if labels is None:
            raise KeyError("Faster R-CNN output has no labels/classes field")
    elif isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        boxes, scores, labels = outputs
    else:
        raise TypeError(
            "Faster R-CNN output must be a detection dict, a one-item list of "
            "detection dicts, or a (boxes, scores, labels) tuple"
        )

    boxes = torch.as_tensor(boxes)
    scores = torch.as_tensor(scores)
    labels = torch.as_tensor(labels)
    if boxes.dim() == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if scores.dim() == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if labels.dim() == 2 and labels.shape[0] == 1:
        labels = labels[0]
    return boxes, scores, labels.to(dtype=torch.int64)


def _map_labels(
    labels: torch.Tensor,
    class_map: Optional[Mapping[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if class_map is None:
        mapped = labels - 1
        return mapped, mapped >= 0

    mapped = torch.full_like(labels, -1)
    for source, target in class_map.items():
        mapped = torch.where(
            labels == source,
            torch.as_tensor(target, device=labels.device, dtype=labels.dtype),
            mapped,
        )
    return mapped, mapped >= 0


def postprocess(
    outputs: Any,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    original_size: Optional[Tuple[int, int]] = None,
    max_det: int = 300,
    class_map: Optional[Mapping[int, int]] = None,
    **_unused,
) -> dict:
    """Return LibreYOLO's canonical contiguous-class detection dictionary.

    Both RPN proposal NMS and class-wise RoI NMS already ran inside the model.
    Applying NMS here would suppress twice and make native/exported execution
    diverge, so ``iou_thres`` is accepted for API compatibility but does not
    alter the fixed upstream RoI NMS threshold (0.5).
    """
    del iou_thres
    boxes, scores, labels = _unpack(outputs)
    mapped_labels, mapped_mask = _map_labels(labels, class_map)
    keep = mapped_mask & (scores > conf_thres)
    boxes = boxes[keep]
    scores = scores[keep]
    mapped_labels = mapped_labels[keep]

    # Filter the sparse COCO labels before enforcing max_det so an unmapped
    # category id can never consume a valid detection's budget.
    order = torch.argsort(scores, descending=True)
    order = order[: max(0, int(max_det))]
    boxes = boxes[order].clone()
    scores = scores[order]
    mapped_labels = mapped_labels[order]

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
    return {
        "num_detections": int(boxes_array.shape[0]),
        "boxes": boxes_array,
        "scores": scores_array,
        "classes": classes_array,
    }
