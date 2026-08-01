"""LW-DETR postprocessing (DETR-style top-K decode, no NMS).

Mirrors upstream's ``PostProcess`` (models/lwdetr.py): sigmoid the logits,
take the top ``num_select`` over every (query x class) pair, convert boxes from
cxcywh to xyxy, then rescale from ``[0, 1]`` to original-image pixels. Because
LW-DETR emits a set prediction, no IoU suppression is applied.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np
import torch


def postprocess(
    outputs,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    original_size: Optional[Tuple[int, int]] = None,
    max_det: int = 300,
    class_map: Optional[Mapping[int, int]] = None,
    **_unused,
):
    """Decode an LW-DETR output dict into a LibreYOLO detections dict.

    Args:
        outputs: ``{"pred_logits": (B, Q, nc), "pred_boxes": (B, Q, 4)}`` with
            boxes in cxcywh normalized to ``[0, 1]``.
        conf_thres: Score threshold applied after top-K.
        iou_thres: Unused — LW-DETR is NMS-free. Accepted for API parity.
        original_size: ``(width, height)`` of the source image; boxes are
            scaled into it when given.
        max_det: Top-K budget over (query x class) pairs.
        class_map: Optional class-index remap (COCO-91 ids to contiguous
            COCO-80 for the released checkpoints). Detections whose class is
            absent from the map are dropped, matching the fact that the 11
            unused COCO ids carry no annotations.

    Returns:
        dict with ``num_detections`` / ``boxes`` / ``scores`` / ``classes``.
    """
    del iou_thres  # LW-DETR is NMS-free; the set prediction is already ranked.

    # Lazy import: libreyolo.models eagerly imports every model class on package
    # init and model modules import from libreyolo.postprocess, so a
    # module-level import here would be circular.
    from ..models.lwdetr.box_ops import box_cxcywh_to_xyxy

    out_logits = outputs["pred_logits"]
    out_bbox = outputs["pred_boxes"]

    if out_logits.dim() == 3:
        out_logits = out_logits[0]
        out_bbox = out_bbox[0]

    num_classes = out_logits.shape[-1]
    prob = out_logits.sigmoid()

    topk_values, topk_indices = torch.topk(
        prob.view(-1), min(max_det, prob.numel())
    )
    scores = topk_values
    query_idx = topk_indices // num_classes
    class_idx = topk_indices % num_classes

    boxes = box_cxcywh_to_xyxy(out_bbox)[query_idx]

    keep = scores > conf_thres
    scores = scores[keep]
    class_idx = class_idx[keep]
    boxes = boxes[keep]

    if class_map is not None:
        mapped = torch.tensor(
            [class_map.get(int(c), -1) for c in class_idx.cpu()],
            dtype=class_idx.dtype,
            device=class_idx.device,
        )
        valid = mapped >= 0
        boxes = boxes[valid]
        scores = scores[valid]
        class_idx = mapped[valid]

    if original_size is not None:
        orig_w, orig_h = original_size
        scale = torch.tensor(
            [orig_w, orig_h, orig_w, orig_h], dtype=boxes.dtype, device=boxes.device
        )
        boxes = boxes * scale

    return {
        "num_detections": int(boxes.shape[0]),
        "boxes": boxes.cpu().numpy()
        if boxes.numel() > 0
        else np.zeros((0, 4), dtype=np.float32),
        "scores": scores.cpu().numpy()
        if scores.numel() > 0
        else np.zeros((0,), dtype=np.float32),
        "classes": class_idx.cpu().numpy()
        if class_idx.numel() > 0
        else np.zeros((0,), dtype=np.int64),
    }
