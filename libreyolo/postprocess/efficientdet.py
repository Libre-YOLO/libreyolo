# Copyright 2020 Google Research and Ross Wightman.
# Licensed under the Apache License, Version 2.0.
#
# Anchor generation and box decoding follow ``effdet`` 0.4.1 at commit
# c6dff775a36cea0bf9b76c58e59f936411c5ce01. LibreYOLO adds configurable
# filtering/NMS, sparse-COCO remapping, and its common result contract.
"""EfficientDet anchor generation, decode, and class-aware NMS."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torchvision.ops import batched_nms

from ..utils.coco import COCO91_TO_COCO80

_STRIDES = (8, 16, 32, 64, 128)
_NUM_SCALES = 3
_ASPECT_RATIOS = ((1.0, 1.0), (1.4, 0.7), (0.7, 1.4))
_ANCHOR_SCALE = 4.0
_ANCHORS_PER_LOCATION = _NUM_SCALES * len(_ASPECT_RATIOS)
_COCO90_CLASS_MAP = tuple(COCO91_TO_COCO80.get(index + 1, -1) for index in range(90))


def generate_anchors(
    input_size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build D0-D4 anchors in upstream ``(y1, x1, y2, x2)`` order."""
    size = int(input_size)
    if size <= 0:
        raise ValueError(f"input_size must be positive, got {input_size}")

    levels: list[torch.Tensor] = []
    # Generate in float64, like the NumPy reference, before its final float32
    # cast. This preserves the exact release-checkpoint anchor coordinates.
    for stride in _STRIDES:
        y = torch.arange(stride / 2, size, stride, dtype=torch.float64)
        x = torch.arange(stride / 2, size, stride, dtype=torch.float64)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        centers_y = yy.reshape(-1)
        centers_x = xx.reshape(-1)

        configurations: list[torch.Tensor] = []
        for scale_index in range(_NUM_SCALES):
            octave = 2.0 ** (scale_index / _NUM_SCALES)
            base = _ANCHOR_SCALE * stride * octave
            for aspect_x, aspect_y in _ASPECT_RATIOS:
                half_w = base * aspect_x / 2.0
                half_h = base * aspect_y / 2.0
                configurations.append(
                    torch.stack(
                        (
                            centers_y - half_h,
                            centers_x - half_w,
                            centers_y + half_h,
                            centers_x + half_w,
                        ),
                        dim=-1,
                    )
                )
        # Upstream concatenates anchors as (location, configuration, box).
        levels.append(torch.stack(configurations, dim=1).reshape(-1, 4))

    return torch.cat(levels, dim=0).to(device=device, dtype=dtype)


def _flatten_outputs(
    class_outputs: Sequence[torch.Tensor],
    box_outputs: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if len(class_outputs) != 5 or len(box_outputs) != 5:
        raise ValueError("EfficientDet expects five class and five box feature levels")
    batch = int(class_outputs[0].shape[0])
    class_channels = int(class_outputs[0].shape[1])
    if class_channels % _ANCHORS_PER_LOCATION:
        raise ValueError(
            "EfficientDet class channels are not divisible by nine anchors"
        )
    num_classes = class_channels // _ANCHORS_PER_LOCATION
    classes = torch.cat(
        [
            level.permute(0, 2, 3, 1).reshape(batch, -1, num_classes)
            for level in class_outputs
        ],
        dim=1,
    )
    boxes = torch.cat(
        [level.permute(0, 2, 3, 1).reshape(batch, -1, 4) for level in box_outputs],
        dim=1,
    )
    if classes.shape[1] != boxes.shape[1]:
        raise ValueError("EfficientDet class and box feature counts do not match")
    return classes, boxes, num_classes


def decode_candidates(
    output: tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]],
    *,
    input_size: int,
    max_candidates: int = 5000,
    sparse_coco: bool = True,
) -> torch.Tensor:
    """Decode top logits to ``(B, K, 6)`` xyxy/score/class candidates.

    Candidate selection intentionally happens on logits before sigmoid, as in
    the reference prediction bench. Invalid gaps in the 90-slot COCO head are
    marked with class ``-1`` and filtered by :func:`postprocess`.
    """
    class_outputs, box_outputs = output
    logits, box_regression, num_classes = _flatten_outputs(class_outputs, box_outputs)
    anchors = generate_anchors(
        input_size,
        device=box_regression.device,
        dtype=box_regression.dtype,
    )
    if anchors.shape[0] != box_regression.shape[1]:
        raise ValueError(
            f"EfficientDet produced {box_regression.shape[1]} boxes for "
            f"{anchors.shape[0]} anchors at imgsz={input_size}"
        )

    batch = int(logits.shape[0])
    budget = min(max(int(max_candidates), 0), int(logits.shape[1] * num_classes))
    if budget == 0:
        return logits.new_zeros((batch, 0, 6))
    top_logits, flat_indices = torch.topk(logits.reshape(batch, -1), budget, dim=1)
    anchor_indices = flat_indices // num_classes
    classes = flat_indices % num_classes
    gathered_regression = torch.gather(
        box_regression,
        1,
        anchor_indices.unsqueeze(-1).expand(-1, -1, 4),
    )
    gathered_anchors = anchors[anchor_indices]

    anchor_y = (gathered_anchors[..., 0] + gathered_anchors[..., 2]) * 0.5
    anchor_x = (gathered_anchors[..., 1] + gathered_anchors[..., 3]) * 0.5
    anchor_h = gathered_anchors[..., 2] - gathered_anchors[..., 0]
    anchor_w = gathered_anchors[..., 3] - gathered_anchors[..., 1]
    ty, tx, th, tw = gathered_regression.unbind(dim=-1)
    center_y = ty * anchor_h + anchor_y
    center_x = tx * anchor_w + anchor_x
    height = torch.exp(th) * anchor_h
    width = torch.exp(tw) * anchor_w
    decoded = torch.stack(
        (
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        ),
        dim=-1,
    )

    if sparse_coco:
        if num_classes != 90:
            raise ValueError(
                f"sparse COCO mapping requires 90 class slots, got {num_classes}"
            )
        class_map = classes.new_tensor(_COCO90_CLASS_MAP)
        classes = class_map[classes]

    return torch.cat(
        (
            decoded,
            top_logits.sigmoid().unsqueeze(-1),
            classes.to(decoded.dtype).unsqueeze(-1),
        ),
        dim=-1,
    )


def _empty_result() -> dict:
    return {
        "num_detections": 0,
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int64),
    }


def postprocess(
    output: tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 512,
    original_size: tuple[int, int] | None = None,
    ratio: float = 1.0,
    max_det: int = 100,
    max_candidates: int = 5000,
    sparse_coco: bool = True,
    **_unused,
) -> dict:
    """Decode one EfficientDet image to LibreYOLO's detection dictionary."""
    candidates = decode_candidates(
        output,
        input_size=input_size,
        max_candidates=max_candidates,
        sparse_coco=sparse_coco,
    )
    if candidates.shape[0] != 1:
        raise ValueError("EfficientDet native postprocess expects a single-image batch")
    candidates = candidates[0]
    keep = (candidates[:, 4] > conf_thres) & (candidates[:, 5] >= 0)
    if not torch.any(keep):
        return _empty_result()

    boxes = candidates[keep, :4].float()
    scores = candidates[keep, 4].float()
    classes = candidates[keep, 5].long()
    boxes[:, 0::2].clamp_(0, int(input_size))
    boxes[:, 1::2].clamp_(0, int(input_size))

    if original_size is not None:
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        original_width, original_height = original_size
        boxes = boxes / float(ratio)
        boxes[:, 0::2].clamp_(0, original_width)
        boxes[:, 1::2].clamp_(0, original_height)

    area_keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not torch.any(area_keep):
        return _empty_result()
    boxes = boxes[area_keep]
    scores = scores[area_keep]
    classes = classes[area_keep]

    selected = batched_nms(boxes, scores, classes, float(iou_thres))
    selected = selected[: max(int(max_det), 0)]
    boxes = boxes[selected].detach().cpu().numpy().astype(np.float32, copy=False)
    scores = scores[selected].detach().cpu().numpy().astype(np.float32, copy=False)
    classes = classes[selected].detach().cpu().numpy().astype(np.int64, copy=False)
    return {
        "num_detections": int(boxes.shape[0]),
        "boxes": boxes,
        "scores": scores,
        "classes": classes,
    }


__all__ = ["decode_candidates", "generate_anchors", "postprocess"]
