"""YOLOv7 anchor decode + shared NMS.

Reproduces upstream ``Anc2Box`` (MIT MultimediaTechLab/YOLO,
``yolo/utils/bounding_box_utils.py``): per head, the raw ``anchor*(5+nc)`` map
is decoded with the YOLOv5/v7 convention

    xy = (2*sigmoid(tx) - 0.5 + grid) * stride
    wh = (2*sigmoid(tw))^2 * anchor      # anchor in input pixels

objectness and class scores are sigmoid, final score = obj * cls. This differs
from the Darknet families' decode (``exp`` boxes), hence a separate module. NMS
reuses :func:`libreyolo.postprocess.common.postprocess_detections`.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from ..utils.general import cxcywh_to_xyxy
from .common import postprocess_detections


def decode_v7_head(
    output: torch.Tensor,
    anchors: Sequence[Tuple[float, float]],
    stride: int,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode one head ``(1, A*(5+nc), H, W)`` to ``(cxcywh_pixels, scores)``."""
    b, c, h, w = output.shape
    a = len(anchors)
    assert c == a * (5 + num_classes), (c, a, num_classes)
    device, dtype = output.device, output.dtype

    # (A, H, W, 5+nc), batch 0
    x = output.view(b, a, 5 + num_classes, h, w).permute(0, 1, 3, 4, 2).contiguous()[0]
    x = x.sigmoid()

    col = torch.arange(w, device=device, dtype=dtype).view(1, 1, w)
    row = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)
    anchor_t = torch.tensor(list(anchors), device=device, dtype=dtype)  # (A,2) w,h

    cx = (x[..., 0] * 2 - 0.5 + col) * stride
    cy = (x[..., 1] * 2 - 0.5 + row) * stride
    bw = (x[..., 2] * 2) ** 2 * anchor_t[:, 0].view(a, 1, 1)
    bh = (x[..., 3] * 2) ** 2 * anchor_t[:, 1].view(a, 1, 1)
    obj = x[..., 4]
    cls = x[..., 5:]

    boxes = torch.stack([cx, cy, bw, bh], dim=-1).reshape(-1, 4)
    scores = (obj.unsqueeze(-1) * cls).reshape(-1, num_classes)
    return boxes, scores


def postprocess(
    outputs: List[torch.Tensor],
    anchors: Sequence[Sequence[float]],
    strides: Sequence[int],
    num_classes: int,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 640,
    original_size: Tuple[int, int] | None = None,
    ratio: float = 1.0,
    max_det: int = 300,
) -> dict:
    """Decode all YOLOv7 heads, threshold, and run per-class NMS.

    ``anchors`` is one flat ``[w0,h0,w1,h1,w2,h2]`` list per head (as in v7.yaml).
    """
    all_boxes, all_scores = [], []
    for output, flat_anchor, stride in zip(outputs, anchors, strides):
        pairs = list(zip(flat_anchor[0::2], flat_anchor[1::2]))
        boxes, scores = decode_v7_head(output, pairs, stride, num_classes)
        all_boxes.append(boxes)
        all_scores.append(scores)

    boxes_cxcywh = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)

    max_scores, class_ids = torch.max(scores, dim=1)
    mask = max_scores > conf_thres
    if not mask.any():
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh[mask])
    return postprocess_detections(
        boxes=boxes_xyxy,
        scores=max_scores[mask],
        class_ids=class_ids[mask],
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        input_size=input_size,
        original_size=original_size,
        max_det=max_det,
        letterbox=True,
    )
