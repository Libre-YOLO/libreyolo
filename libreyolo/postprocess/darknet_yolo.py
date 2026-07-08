"""Anchor-box decode + shared NMS for the Darknet families (YOLOv2/v3/v4).

This is LibreYOLO's first *anchor-box* decoder — every other family is
anchor-free (YOLOX grid, YOLO9 anchor points) or query-based (the DETRs). One
parametric decoder covers all three Darknet head styles, driven by the
:class:`~libreyolo.models.darknet.net.DetectionSpec` metadata attached to each
head:

* ``[yolo]`` (v3/v4): anchors in **input pixels**, objectness + independent
  sigmoid class scores, optional YOLOv4 ``scale_x_y`` center expansion:
  ``bx = (sigmoid(tx)*s - (s-1)/2 + col) * stride``, ``bw = anchor_w * e^{tw}``.
* ``[region]`` (v2): anchors in **grid cells**, objectness + **softmax** class
  scores: ``bx = (sigmoid(tx) + col) * stride``, ``bw = anchor_w * e^{tw} *
  stride``.

The decode math is reproduced from the public-domain Darknet source
(``src/yolo_layer.c`` / ``src/region_layer.c``); no source is copied. NMS reuses
the shared :func:`libreyolo.postprocess.common.postprocess_detections`.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from ..utils.general import cxcywh_to_xyxy
from .common import postprocess_detections


def decode_head(
    output: torch.Tensor,
    spec,
    input_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode one head's raw feature map to ``(cxcywh_pixels, scores)``.

    Args:
        output: raw head tensor ``(B, A*(5+nc), H, W)``. ``B`` must be 1.
        spec: the head's :class:`DetectionSpec` (anchors, stride, kind, ...).
        input_size: square network input resolution in pixels.

    Returns:
        ``(boxes, scores)`` where ``boxes`` is ``(A*H*W, 4)`` center-x/center-y/
        width/height in input-image pixels and ``scores`` is ``(A*H*W, nc)``.
    """
    b, c, h, w = output.shape
    a = len(spec.anchors)
    nc = spec.num_classes
    assert c == a * (5 + nc), (
        f"head channel mismatch: got {c}, expected {a}*(5+{nc})={a * (5 + nc)}"
    )
    device, dtype = output.device, output.dtype

    # (B, A, 5+nc, H, W) -> (A, H, W, 5+nc), batch 0
    x = output.view(b, a, 5 + nc, h, w).permute(0, 1, 3, 4, 2).contiguous()[0]

    anchors = torch.tensor(spec.anchors, device=device, dtype=dtype)  # (A, 2) w,h

    # grid columns (x) and rows (y)
    col = torch.arange(w, device=device, dtype=dtype).view(1, 1, w)
    row = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)

    tx = torch.sigmoid(x[..., 0])
    ty = torch.sigmoid(x[..., 1])
    tw = x[..., 2]
    th = x[..., 3]
    obj = torch.sigmoid(x[..., 4])
    cls_logits = x[..., 5:]

    s = spec.scale_x_y
    if s != 1.0:
        tx = tx * s - (s - 1.0) / 2.0
        ty = ty * s - (s - 1.0) / 2.0

    cx = (tx + col) * spec.stride
    cy = (ty + row) * spec.stride

    aw = anchors[:, 0].view(a, 1, 1)
    ah = anchors[:, 1].view(a, 1, 1)
    if spec.kind == "region":
        bw = torch.exp(tw) * aw * spec.stride
        bh = torch.exp(th) * ah * spec.stride
        cls_prob = torch.softmax(cls_logits, dim=-1)
    else:  # yolo (v3/v4): anchors already in pixels
        bw = torch.exp(tw) * aw
        bh = torch.exp(th) * ah
        cls_prob = torch.sigmoid(cls_logits)

    boxes = torch.stack([cx, cy, bw, bh], dim=-1).reshape(-1, 4)
    scores = (obj.unsqueeze(-1) * cls_prob).reshape(-1, nc)
    return boxes, scores


def decode_detection(
    output: torch.Tensor,
    spec,
    input_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode the YOLOv1 ``[detection]`` head (dense FC vector) to boxes/scores.

    Unlike the anchor heads, YOLOv1 emits a flat ``(B, S*S*(C + N*(coords+1)))``
    vector from a fully-connected layer, laid out (per Darknet ``detection_layer``)
    as three contiguous blocks: class probabilities ``[S*S*C]``, per-box
    confidences ``[S*S*N]``, then box coordinates ``[S*S*N*coords]``. The class
    distribution is shared by the ``N`` boxes of a cell.

    Reproduces Darknet's ``get_detection_boxes``:

    * ``bx = (raw_x + col) / S``, ``by = (raw_y + row) / S`` (image-relative)
    * ``bw = raw_w**2`` (``sqrt`` head), ``bh = raw_h**2`` (else raw)
    * ``score[j] = confidence * class_prob[j]``

    Boxes are returned as center-x/center-y/width/height in **letterboxed input
    pixels** (image-relative ``[0, 1]`` scaled by ``input_size``), matching the
    anchor decoders so the shared NMS + letterbox-inverse path is reused.
    """
    b = output.shape[0]
    assert b == 1, "darknet detection decode expects batch size 1"
    s = spec.side
    n = spec.boxes_per_cell
    nc = spec.num_classes
    coords = spec.coords
    cells = s * s
    device, dtype = output.device, output.dtype

    preds = output.view(-1)
    class_block = preds[: cells * nc].view(cells, nc)  # (cells, C)
    conf_block = preds[cells * nc : cells * nc + cells * n].view(cells, n)  # (cells, N)
    box_block = preds[cells * nc + cells * n :].view(cells, n, coords)  # (cells, N, 4)

    if spec.softmax:
        class_block = torch.softmax(class_block, dim=-1)

    rows = (torch.arange(cells, device=device, dtype=dtype) // s).view(cells, 1)
    cols = (torch.arange(cells, device=device, dtype=dtype) % s).view(cells, 1)

    raw_x = box_block[..., 0]
    raw_y = box_block[..., 1]
    raw_w = box_block[..., 2]
    raw_h = box_block[..., 3]

    cx = (raw_x + cols) / s * input_size  # (cells, N)
    cy = (raw_y + rows) / s * input_size
    if spec.sqrt:
        bw = raw_w.pow(2) * input_size
        bh = raw_h.pow(2) * input_size
    else:
        bw = raw_w * input_size
        bh = raw_h * input_size

    boxes = torch.stack([cx, cy, bw, bh], dim=-1).reshape(-1, 4)  # (cells*N, 4)
    # score[cell, box, cls] = conf[cell, box] * class_prob[cell, cls]
    scores = (conf_block.unsqueeze(-1) * class_block.unsqueeze(1)).reshape(-1, nc)
    return boxes, scores


def _decode_detection_export(output: torch.Tensor, spec) -> torch.Tensor:
    """Traceable, batch-general YOLOv1 detection decode to ``(B, S*S*N, 4+nc)``.

    xyxy in input pixels + per-class ``conf * class_prob`` scores, matching the
    anchor-head export rows so the shared backend decode consumes one tensor.
    """
    s = spec.side
    n = spec.boxes_per_cell
    nc = spec.num_classes
    coords = spec.coords
    cells = s * s
    input_size = spec.stride * s
    bsz = output.shape[0]

    class_block = output[:, : cells * nc].view(bsz, cells, nc)
    conf_block = output[:, cells * nc : cells * nc + cells * n].view(bsz, cells, n)
    box_block = output[:, cells * nc + cells * n :].view(bsz, cells, n, coords)
    if spec.softmax:
        class_block = torch.softmax(class_block, dim=-1)

    grid = output.new_ones(cells).cumsum(0) - 1.0  # 0..cells-1
    rows_idx = (grid // s).view(1, cells, 1)
    cols_idx = (grid - (grid // s) * s).view(1, cells, 1)

    cx = (box_block[..., 0] + cols_idx) / s * input_size  # (B, cells, N)
    cy = (box_block[..., 1] + rows_idx) / s * input_size
    if spec.sqrt:
        bw = box_block[..., 2].pow(2) * input_size
        bh = box_block[..., 3].pow(2) * input_size
    else:
        bw = box_block[..., 2] * input_size
        bh = box_block[..., 3] * input_size

    x1 = (cx - bw / 2).unsqueeze(-1)
    y1 = (cy - bh / 2).unsqueeze(-1)
    x2 = (cx + bw / 2).unsqueeze(-1)
    y2 = (cy + bh / 2).unsqueeze(-1)
    scores = conf_block.unsqueeze(-1) * class_block.unsqueeze(2)  # (B, cells, N, nc)
    row = torch.cat([x1, y1, x2, y2, scores], dim=-1)  # (B, cells, N, 4+nc)
    return row.reshape(bsz, cells * n, 4 + nc)


def decode_export(
    outputs: List[torch.Tensor],
    specs: Sequence,
    anchor_tensors: List[torch.Tensor],
) -> torch.Tensor:
    """Batched, traceable decode of all heads to ``(B, 4+nc, N)``.

    Boxes are xyxy in input-image pixels; scores are per-class ``obj * cls``
    (sigmoid for yolo heads, softmax for region). This is the format the shared
    backend decode (`_parse_yolo9`) consumes, so an exported graph that ends in
    this tensor works across every runtime backend (ONNX/NCNN/TorchScript/...).

    ``anchor_tensors`` are ``(A, 2)`` buffers (device-tracked module state)
    supplied by the export wrapper — passing anchors as constants would bake the
    trace device into TorchScript and break CPU/GPU portability.
    """
    rows = []
    nc = specs[0].num_classes
    for output, spec, anchors in zip(outputs, specs, anchor_tensors):
        if spec.kind == "detection":  # YOLOv1 dense FC head
            rows.append(_decode_detection_export(output, spec))
            continue
        b, c, h, w = output.shape
        a = len(spec.anchors)
        x = output.view(b, a, 5 + nc, h, w).permute(0, 1, 3, 4, 2)  # B,A,H,W,5+nc

        # Grids from ``output`` (new_ones/cumsum) follow the input device at
        # runtime; anchors come in as device-tracked buffers for the same reason.
        col = (output.new_ones(w).cumsum(0) - 1.0).view(1, 1, 1, w)
        row = (output.new_ones(h).cumsum(0) - 1.0).view(1, 1, h, 1)
        aw = anchors[:, 0].view(1, a, 1, 1)
        ah = anchors[:, 1].view(1, a, 1, 1)

        tx = torch.sigmoid(x[..., 0])
        ty = torch.sigmoid(x[..., 1])
        obj = torch.sigmoid(x[..., 4])
        s = spec.scale_x_y
        if s != 1.0:
            tx = tx * s - (s - 1.0) / 2.0
            ty = ty * s - (s - 1.0) / 2.0
        cx = (tx + col) * spec.stride
        cy = (ty + row) * spec.stride
        if spec.kind == "region":
            bw = torch.exp(x[..., 2]) * aw * spec.stride
            bh = torch.exp(x[..., 3]) * ah * spec.stride
            clsp = torch.softmax(x[..., 5:], dim=-1)
        else:
            bw = torch.exp(x[..., 2]) * aw
            bh = torch.exp(x[..., 3]) * ah
            clsp = torch.sigmoid(x[..., 5:])
        x1 = (cx - bw / 2).unsqueeze(-1)
        y1 = (cy - bh / 2).unsqueeze(-1)
        x2 = (cx + bw / 2).unsqueeze(-1)
        y2 = (cy + bh / 2).unsqueeze(-1)
        score = obj.unsqueeze(-1) * clsp
        row_t = torch.cat([x1, y1, x2, y2, score], dim=-1)  # B,A,H,W,4+nc
        rows.append(row_t.reshape(b, a * h * w, 4 + nc))
    out = torch.cat(rows, dim=1)  # B, N, 4+nc
    return out.permute(0, 2, 1)  # B, 4+nc, N (yolo9-compatible)


def postprocess(
    outputs: List[torch.Tensor],
    specs: Sequence,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 416,
    original_size: Tuple[int, int] | None = None,
    ratio: float = 1.0,
    max_det: int = 300,
) -> dict:
    """Decode all Darknet heads and run confidence filtering + NMS.

    Args:
        outputs: raw head tensors in the order returned by ``DarknetNet.forward``.
        specs: matching :class:`DetectionSpec` sequence (same order/length).
        conf_thres/iou_thres/max_det: standard detection knobs.
        input_size: network input resolution (square).
        original_size: ``(width, height)`` of the source image for rescaling.
        ratio: letterbox scale factor applied during preprocessing
            (``r = min(input/orig_h, input/orig_w)``). Boxes are divided by it.
    """
    if len(outputs) != len(specs):
        raise ValueError(
            f"decode expects one spec per head: {len(outputs)} outputs vs "
            f"{len(specs)} specs"
        )

    all_boxes = []
    all_scores = []
    for output, spec in zip(outputs, specs):
        if spec.kind == "detection":  # YOLOv1 dense FC head
            boxes, scores = decode_detection(output, spec, input_size)
        else:  # anchor heads (v2 region / v3-v4 yolo)
            boxes, scores = decode_head(output, spec, input_size)
        all_boxes.append(boxes)
        all_scores.append(scores)

    boxes_cxcywh = torch.cat(all_boxes, dim=0)  # (N, 4)
    scores = torch.cat(all_scores, dim=0)  # (N, nc)

    max_scores, class_ids = torch.max(scores, dim=1)
    mask = max_scores > conf_thres
    if not mask.any():
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    boxes_xyxy = cxcywh_to_xyxy(boxes_cxcywh[mask])
    kept_scores = max_scores[mask]
    kept_classes = class_ids[mask]

    # v2/v3/v4 letterbox (aspect-preserving) their input, so the inverse is a
    # single divide by r = min(input/orig_h, input/orig_w). YOLOv1's FC head is
    # fed a plain square stretch instead (see preprocess_numpy_stretch), so its
    # inverse is independent x/y scaling (letterbox=False). Either way
    # postprocess_detections then clamps to the original bounds, drops degenerate
    # boxes, and runs per-class NMS.
    use_letterbox = not any(getattr(s, "kind", None) == "detection" for s in specs)
    return postprocess_detections(
        boxes=boxes_xyxy,
        scores=kept_scores,
        class_ids=kept_classes,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        input_size=input_size,
        original_size=original_size,
        max_det=max_det,
        letterbox=use_letterbox,
    )
