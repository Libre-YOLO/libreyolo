"""CenterNet heatmap decoding and affine box restoration (NMS-free)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _gather(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    dimensions = features.size(2)
    expanded = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), dimensions)
    return features.gather(1, expanded)


def _transpose_and_gather(
    features: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    features = features.permute(0, 2, 3, 1).contiguous()
    features = features.view(features.size(0), -1, features.size(3))
    return _gather(features, indices)


def _topk(
    scores: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, categories, height, width = scores.size()
    tracing = torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
    per_class_k = topk if tracing else min(topk, height * width)
    class_scores, class_indices = torch.topk(
        scores.view(batch, categories, -1), per_class_k
    )
    class_indices = class_indices % (height * width)
    class_ys = (class_indices // width).to(scores.dtype)
    class_xs = (class_indices % width).to(scores.dtype)

    global_k = topk if tracing else min(topk, categories * per_class_k)
    top_scores, top_indices = torch.topk(class_scores.view(batch, -1), global_k)
    top_classes = (top_indices // per_class_k).to(torch.int32)
    flattened_indices = _gather(class_indices.view(batch, -1, 1), top_indices).view(
        batch, global_k
    )
    top_ys = _gather(class_ys.view(batch, -1, 1), top_indices).view(batch, global_k)
    top_xs = _gather(class_xs.view(batch, -1, 1), top_indices).view(batch, global_k)
    return top_scores, flattened_indices, top_classes, top_ys, top_xs


def decode_centernet(
    heatmap: torch.Tensor,
    width_height: torch.Tensor,
    regression: torch.Tensor,
    *,
    topk: int = 100,
) -> torch.Tensor:
    """Decode raw stride-4 heads into ``[x1,y1,x2,y2,score,class]`` rows."""
    heatmap = heatmap.sigmoid()
    peaks = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    heatmap = heatmap * peaks.eq(heatmap).to(heatmap.dtype)
    scores, indices, classes, ys, xs = _topk(heatmap, topk)

    regression = _transpose_and_gather(regression, indices).view(
        heatmap.size(0), -1, 2
    )
    xs = xs.view(heatmap.size(0), -1, 1) + regression[:, :, 0:1]
    ys = ys.view(heatmap.size(0), -1, 1) + regression[:, :, 1:2]
    width_height = _transpose_and_gather(width_height, indices).view(
        heatmap.size(0), -1, 2
    )
    classes = classes.view(heatmap.size(0), -1, 1).to(heatmap.dtype)
    scores = scores.view(heatmap.size(0), -1, 1)
    boxes = torch.cat(
        (
            xs - width_height[..., 0:1] / 2,
            ys - width_height[..., 1:2] / 2,
            xs + width_height[..., 0:1] / 2,
            ys + width_height[..., 1:2] / 2,
        ),
        dim=2,
    )
    return torch.cat((boxes, scores, classes), dim=2)


def _decoded_rows(
    outputs: Any, topk: int
) -> tuple[torch.Tensor, tuple[int, int] | None]:
    if isinstance(outputs, dict):
        return (
            decode_centernet(
                outputs["hm"], outputs["wh"], outputs["reg"], topk=topk
            ),
            (int(outputs["hm"].shape[-2]), int(outputs["hm"].shape[-1])),
        )
    if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        return (
            decode_centernet(outputs[0], outputs[1], outputs[2], topk=topk),
            (int(outputs[0].shape[-2]), int(outputs[0].shape[-1])),
        )
    decoded = torch.as_tensor(outputs)
    if decoded.ndim != 3 or decoded.shape[-1] != 6:
        raise ValueError("CenterNet output must be raw heads or a (B, K, 6) tensor")
    return decoded, None


def postprocess(
    outputs: Any,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    original_size: tuple[int, int] | None = None,
    input_size: int | tuple[int, int] = 512,
    max_det: int = 100,
    topk: int = 100,
    **_unused,
) -> dict:
    """Filter CenterNet's ranked peaks and restore source-image coordinates."""
    del iou_thres
    rows, feature_size = _decoded_rows(outputs, topk=topk)
    if rows.shape[0] != 1:
        raise ValueError("CenterNet postprocess expects one batch item")
    rows = rows[0]
    boxes = rows[:, :4]
    scores = rows[:, 4]
    classes = rows[:, 5].to(torch.int64)
    if isinstance(input_size, (tuple, list)):
        input_height, input_width = int(input_size[0]), int(input_size[1])
    else:
        input_height = input_width = int(input_size)
    if feature_size is not None:
        feature_height, feature_width = feature_size
        boxes = boxes * boxes.new_tensor(
            [
                input_width / feature_width,
                input_height / feature_height,
                input_width / feature_width,
                input_height / feature_height,
            ]
        )

    budget = min(max(int(max_det), 0), int(scores.numel()))
    boxes = boxes[:budget]
    scores = scores[:budget]
    classes = classes[:budget]
    keep = scores > conf_thres
    boxes = boxes[keep].clone()
    scores = scores[keep]
    classes = classes[keep]

    if original_size is not None and boxes.numel():
        from ..models.centernet.utils import get_affine_transform, image_geometry

        center, scale = image_geometry(original_size, input_width)
        inverse = get_affine_transform(
            center, scale, (input_width, input_height), inverse=True
        )
        matrix = boxes.new_tensor(inverse)
        ones = torch.ones((boxes.shape[0], 1), dtype=boxes.dtype, device=boxes.device)
        first = torch.cat((boxes[:, :2], ones), dim=1) @ matrix.T
        second = torch.cat((boxes[:, 2:4], ones), dim=1) @ matrix.T
        boxes = torch.cat((first, second), dim=1)
        width, height = original_size
        boxes[:, [0, 2]].clamp_(0, width)
        boxes[:, [1, 3]].clamp_(0, height)

    boxes_array = boxes.detach().cpu().to(torch.float32).numpy()
    scores_array = scores.detach().cpu().to(torch.float32).numpy()
    classes_array = classes.detach().cpu().to(torch.int64).numpy()
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


__all__ = ["decode_centernet", "postprocess"]
