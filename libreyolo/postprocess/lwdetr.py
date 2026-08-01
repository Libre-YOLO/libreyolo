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
            COCO-80 for the released checkpoints). Columns absent from the map
            are excluded from selection entirely -- see below.

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

    prob = out_logits.sigmoid()

    class_ids = None
    if class_map is not None:
        # Drop the unmapped columns *before* top-K rather than filtering after.
        # The 11 COCO ids with no annotations are still columns of the 91-wide
        # head, so a post-hoc filter would let one of them consume a slot of the
        # max_det budget and silently return fewer detections than asked for,
        # with no replacement pulled up from the next valid candidate. Slicing
        # first makes the selection behave exactly like an 80-class head.
        source_ids = sorted(class_map)
        columns = torch.as_tensor(source_ids, dtype=torch.long, device=prob.device)
        class_ids = torch.as_tensor(
            [class_map[i] for i in source_ids], dtype=torch.long, device=prob.device
        )
        prob = prob[:, columns]

    num_classes = prob.shape[-1]
    topk_values, topk_indices = torch.topk(
        prob.reshape(-1), min(max_det, prob.numel())
    )
    scores = topk_values
    query_idx = topk_indices // num_classes
    class_idx = topk_indices % num_classes
    if class_ids is not None:
        class_idx = class_ids[class_idx]

    boxes = box_cxcywh_to_xyxy(out_bbox)[query_idx]

    keep = scores > conf_thres
    scores = scores[keep]
    class_idx = class_idx[keep]
    boxes = boxes[keep]

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
