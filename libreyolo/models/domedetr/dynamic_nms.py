"""Density-adaptive NMS over the encoder's query proposals.

Ported from Dome-DETR (https://github.com/RicePasteM/Dome-DETR),
commit 2dde3bc1946a3e9fad9abd0612b59fc39bd6b861, Apache License 2.0.
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

Unlike ordinary NMS this uses a *per-box* IoU threshold: PAQI sets it from the
local DeFE density, so crowded regions suppress less aggressively than sparse
ones. Suppression is greedy in descending score order, per class.
"""

from __future__ import annotations

import torch


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union


def _per_class_dynamic_nms(boxes, scores, iou_thresholds):
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    thresholds = iou_thresholds[order]

    iou_matrix = _box_iou(boxes, boxes)

    num = boxes.shape[0]
    keep_flags = torch.ones(num, dtype=torch.bool, device=boxes.device)
    keep: list[int] = []
    for i in range(num):
        if not keep_flags[i]:
            continue
        keep.append(i)
        if i < num - 1:
            suppress = iou_matrix[i, (i + 1) :] >= thresholds[i]
            keep_flags[(i + 1) :] &= ~suppress
    return order[torch.tensor(keep, dtype=torch.long, device=boxes.device)]


def dynamic_nms(boxes, scores, classes, iou_thresholds) -> torch.Tensor:
    """Return the indices surviving per-class density-adaptive suppression."""
    keep_mask = torch.zeros_like(classes, dtype=torch.bool)
    for cls in classes.unique():
        cls_mask = classes == cls
        keep_cls = _per_class_dynamic_nms(
            boxes[cls_mask], scores[cls_mask], iou_thresholds[cls_mask]
        )
        cls_indices = torch.nonzero(cls_mask, as_tuple=True)[0]
        keep_mask[cls_indices[keep_cls]] = True
    return torch.nonzero(keep_mask, as_tuple=True)[0]
