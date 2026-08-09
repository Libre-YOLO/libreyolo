# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0. Derived from DETR's
# PostProcess at commit 29901c51d7fe8712168b8d0d64351170bc0f83e0.
# LibreYOLO adds confidence filtering, ranking, max_det, sparse-COCO mapping,
# NumPy output conversion, and its common inference signature.
"""Vanilla DETR softmax set-prediction postprocessing (no NMS)."""

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
    max_det: int = 100,
    class_map: Optional[Mapping[int, int]] = None,
    **_unused,
):
    """Decode a single DETR output into LibreYOLO's detection dictionary.

    Softmax is evaluated over every architectural class and the final
    no-object class. Each object query then contributes only its best valid
    class. Set prediction is already duplicate-aware, so ``iou_thres`` is
    accepted for API compatibility but no NMS is applied.
    """
    del iou_thres

    logits = outputs["pred_logits"]
    boxes_cxcywh = outputs["pred_boxes"]
    if logits.ndim == 3:
        logits = logits[0]
        boxes_cxcywh = boxes_cxcywh[0]
    if logits.ndim != 2 or boxes_cxcywh.ndim != 2:
        raise ValueError("DETR postprocess expects one (queries, classes) output")
    if logits.shape[0] != boxes_cxcywh.shape[0] or boxes_cxcywh.shape[-1] != 4:
        raise ValueError("DETR logits and boxes have incompatible shapes")

    # The final logit is DETR's explicit no-object class. It must participate
    # in softmax before being excluded, otherwise every object confidence is
    # inflated. Official COCO heads also contain 11 unused category-id slots;
    # slice those before per-query selection so an invalid id cannot consume a
    # max_det slot.
    probabilities = logits.softmax(dim=-1)
    object_probabilities = probabilities[:, :-1]
    mapped_ids = None
    if class_map is not None:
        source_ids = sorted(class_map)
        columns = torch.as_tensor(
            source_ids, dtype=torch.long, device=object_probabilities.device
        )
        mapped_ids = torch.as_tensor(
            [class_map[index] for index in source_ids],
            dtype=torch.long,
            device=object_probabilities.device,
        )
        object_probabilities = object_probabilities[:, columns]

    scores, classes = object_probabilities.max(dim=-1)
    if mapped_ids is not None:
        classes = mapped_ids[classes]

    budget = min(max(int(max_det), 0), int(scores.numel()))
    if budget:
        scores, query_indices = torch.topk(scores, budget)
        classes = classes[query_indices]
        boxes = _box_cxcywh_to_xyxy(boxes_cxcywh)[query_indices]
    else:
        scores = scores[:0]
        classes = classes[:0]
        boxes = boxes_cxcywh.new_zeros((0, 4))

    keep = scores > conf_thres
    scores = scores[keep]
    classes = classes[keep]
    boxes = boxes[keep]

    if original_size is not None:
        original_width, original_height = original_size
        scale = boxes.new_tensor(
            [original_width, original_height, original_width, original_height]
        )
        boxes = boxes * scale

    return {
        "num_detections": int(boxes.shape[0]),
        "boxes": (
            boxes.cpu().numpy() if boxes.numel() else np.zeros((0, 4), dtype=np.float32)
        ),
        "scores": (
            scores.cpu().numpy() if scores.numel() else np.zeros((0,), dtype=np.float32)
        ),
        "classes": (
            classes.cpu().numpy() if classes.numel() else np.zeros((0,), dtype=np.int64)
        ),
    }
