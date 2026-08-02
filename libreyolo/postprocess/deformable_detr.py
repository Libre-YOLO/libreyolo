"""Deformable DETR sigmoid top-K decoding without NMS."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np
import torch


def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def postprocess(
    outputs,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    original_size: Optional[Tuple[int, int]] = None,
    max_det: int = 300,
    class_map: Optional[Mapping[int, int]] = None,
    **_unused,
):
    """Decode normalized set predictions into the canonical detection dict.

    The official 91-column COCO head uses sparse category ids. Unmapped ids
    are removed before top-K so they cannot consume the user's detection
    budget. Deformable DETR is a set predictor, so IoU/NMS is intentionally not
    applied.
    """
    del iou_thres
    logits = outputs["pred_logits"]
    normalized_boxes = outputs["pred_boxes"]
    if logits.dim() == 3:
        logits = logits[0]
        normalized_boxes = normalized_boxes[0]

    probabilities = logits.sigmoid()
    remapped_ids = None
    if class_map is not None:
        source_ids = sorted(class_map)
        source_columns = torch.as_tensor(
            source_ids, dtype=torch.long, device=probabilities.device
        )
        remapped_ids = torch.as_tensor(
            [class_map[index] for index in source_ids],
            dtype=torch.long,
            device=probabilities.device,
        )
        probabilities = probabilities[:, source_columns]

    num_classes = probabilities.shape[-1]
    scores, flat_indices = torch.topk(
        probabilities.reshape(-1), min(max_det, probabilities.numel())
    )
    query_indices = flat_indices // num_classes
    class_indices = flat_indices % num_classes
    if remapped_ids is not None:
        class_indices = remapped_ids[class_indices]

    boxes = _box_cxcywh_to_xyxy(normalized_boxes)[query_indices]
    keep = scores > conf_thres
    boxes = boxes[keep]
    scores = scores[keep]
    class_indices = class_indices[keep]

    if original_size is not None:
        original_width, original_height = original_size
        scale = torch.tensor(
            [original_width, original_height, original_width, original_height],
            dtype=boxes.dtype,
            device=boxes.device,
        )
        boxes = boxes * scale

    return {
        "num_detections": int(boxes.shape[0]),
        "boxes": boxes.detach().cpu().numpy()
        if boxes.numel()
        else np.zeros((0, 4), dtype=np.float32),
        "scores": scores.detach().cpu().numpy()
        if scores.numel()
        else np.zeros((0,), dtype=np.float32),
        "classes": class_indices.detach().cpu().numpy()
        if class_indices.numel()
        else np.zeros((0,), dtype=np.int64),
    }


__all__ = ["postprocess"]
