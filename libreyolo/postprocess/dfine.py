"""D-FINE postprocessing (DETR-style top-K decode, no NMS).

Moved verbatim from ``libreyolo/models/dfine/utils.py``, which re-exports it
for backward compatibility. Also the single source for DEIM
(``postprocess.deim`` re-exports this function — the historical DEIM copy
was code-identical), DEIMv2, and RT-DETRv4 (inherits LibreDFINE).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _cleanup_masks(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if masks.numel() == 0:
        return masks
    _, h, w = masks.shape
    ys = torch.arange(h, device=masks.device)[None, :, None]
    xs = torch.arange(w, device=masks.device)[None, None, :]
    x1, y1, x2, y2 = boxes.T
    inside = (
        (xs >= x1[:, None, None])
        & (xs < x2[:, None, None])
        & (ys >= y1[:, None, None])
        & (ys < y2[:, None, None])
    )
    return masks * inside.to(dtype=masks.dtype)


def postprocess_seg(
    outputs,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    input_size=None,
    **_unused,
):
    """Decode D-FINE-seg output dict into detections plus instance masks.

    ``iou_thres`` is accepted for API parity but unused: D-FINE is a set
    predictor (top-k selection, no NMS), matching ``postprocess`` below.
    ``input_size`` routes the mask upsample through the model input
    resolution first, matching the exported-backend two-step resize path.
    """
    from ..models.dfine.box_ops import box_cxcywh_to_xyxy

    out_logits = outputs["pred_logits"]
    out_bbox = outputs["pred_boxes"]
    mask_pred = outputs.get("pred_masks")

    if out_logits.dim() == 3:
        out_logits = out_logits[0]
        out_bbox = out_bbox[0]
        if mask_pred is not None:
            mask_pred = mask_pred[0]

    num_classes = out_logits.shape[-1]
    prob = out_logits.sigmoid()
    topk_values, topk_indices = torch.topk(prob.view(-1), min(max_det, prob.numel()))
    scores = topk_values
    query_idx = topk_indices // num_classes
    class_idx = topk_indices % num_classes

    boxes_xyxy = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes_xyxy[query_idx]

    keep = scores > conf_thres
    scores = scores[keep]
    class_idx = class_idx[keep]
    boxes = boxes[keep]
    sel_query = query_idx[keep]

    masks = None
    if original_size is not None:
        orig_w, orig_h = original_size
        scale = torch.tensor(
            [orig_w, orig_h, orig_w, orig_h],
            dtype=boxes.dtype,
            device=boxes.device,
        )
        boxes = boxes * scale

        if mask_pred is not None and sel_query.numel() > 0:
            sel_masks = mask_pred[sel_query].float()
            if input_size is not None:
                if sel_masks.is_cuda:
                    sel_masks = sel_masks.to(dtype=torch.float16)
                if isinstance(input_size, (tuple, list)):
                    in_h, in_w = int(input_size[0]), int(input_size[-1])
                else:
                    in_h = in_w = int(input_size)
                sel_masks = F.interpolate(
                    sel_masks.unsqueeze(1),
                    size=(in_h, in_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            sel_masks = F.interpolate(
                sel_masks.unsqueeze(1),
                size=(int(orig_h), int(orig_w)),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            sel_masks = (sel_masks.clamp_(0, 1) >= 0.5).to(torch.uint8)
            masks = _cleanup_masks(sel_masks, boxes).bool().cpu()

    out = {
        "num_detections": int(boxes.shape[0]),
        "boxes": boxes.cpu().numpy()
        if boxes.numel() > 0
        else np.zeros((0, 4), dtype=np.float32),
        "scores": scores.cpu().numpy()
        if scores.numel() > 0
        else np.zeros((0,), dtype=np.float32),
        "classes": (
            class_idx.cpu().numpy()
            if class_idx.numel() > 0
            else np.zeros((0,), dtype=np.int64)
        ),
    }
    if masks is not None:
        out["masks"] = masks
    return out


def postprocess(
    outputs,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    **_unused,
):
    """Decode D-FINE output dict into a LibreYOLO detections dict.

    D-FINE outputs DETR-style ``{"pred_logits", "pred_boxes"}``. Post-processing:
    sigmoid → top-K across (query × class) → box-cxcywh→xyxy → scale to orig.
    No NMS is applied (set prediction already).

    Returns dict with ``num_detections`` / ``boxes`` / ``scores`` / ``classes``.
    """
    # Lazy import: libreyolo.models eagerly imports every model class on
    # package init, and model modules import from libreyolo.postprocess, so
    # a module-level import here would be circular (see package docstring).
    from ..models.dfine.box_ops import box_cxcywh_to_xyxy

    out_logits = outputs["pred_logits"]  # (B, Q, nc)
    out_bbox = outputs["pred_boxes"]  # (B, Q, 4) cxcywh in [0, 1]

    if out_logits.dim() == 3:
        out_logits = out_logits[0]
        out_bbox = out_bbox[0]

    num_classes = out_logits.shape[-1]
    prob = out_logits.sigmoid()

    # Top-K across all (queries × classes).
    topk_values, topk_indices = torch.topk(prob.view(-1), min(max_det, prob.numel()))
    scores = topk_values
    query_idx = topk_indices // num_classes
    class_idx = topk_indices % num_classes

    boxes_xyxy = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes_xyxy[query_idx]

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
        "classes": (
            class_idx.cpu().numpy()
            if class_idx.numel() > 0
            else np.zeros((0,), dtype=np.int64)
        ),
    }
