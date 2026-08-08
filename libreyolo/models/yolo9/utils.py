"""
Utility functions for YOLO9.

Provides preprocessing functions for YOLOv9 inference. Postprocessing lives
in ``libreyolo.postprocess.yolo9`` and is re-exported here for backward
compatibility.
"""

from __future__ import annotations

from ...utils.lazy import lazy_module


from ...preprocess.yolo9 import (  # noqa: F401  (moved; re-exported for backward compatibility)
    preprocess_image,
    preprocess_numpy,
)
from ...postprocess.yolo9 import (  # noqa: F401  (backward-compatible re-exports)
    ImageSize,
    _YOLO9_MAX_NMS_CANDIDATES,
    _YOLO9_OBB_MAX_NMS_CANDIDATES,
    _YOLO9_OBB_PREFILTER_CANDIDATES,
    _input_size_hw,
    _nms_keep_indices,
    _obb_prefilter_keep_indices,
    _rotated_nms_keep_indices,
    _xywhr_to_corners,
    _xywhr_to_xyxy,
    postprocess,
)


# torch is resolved on first use so this module stays importable in a
# torch-free ONNX deployment (discussions/711).
torch = lazy_module("torch")


def decode_boxes(
    box_preds: torch.Tensor, anchors: torch.Tensor, stride_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Decode box predictions to xyxy coordinates.

    Args:
        box_preds: Box predictions [l, t, r, b] distances from anchors (B, N, 4)
        anchors: Anchor points (N, 2)
        stride_tensor: Stride values (N, 1)

    Returns:
        Decoded boxes in xyxy format (B, N, 4)
    """
    anchors = anchors.unsqueeze(0)
    stride_tensor = stride_tensor.unsqueeze(0)

    # Decode: xyxy = [x - l, y - t, x + r, y + b] * stride
    x1 = (anchors[..., 0:1] - box_preds[..., 0:1]) * stride_tensor[..., 0:1]
    y1 = (anchors[..., 1:2] - box_preds[..., 1:2]) * stride_tensor[..., 0:1]
    x2 = (anchors[..., 0:1] + box_preds[..., 2:3]) * stride_tensor[..., 0:1]
    y2 = (anchors[..., 1:2] + box_preds[..., 3:4]) * stride_tensor[..., 0:1]

    decoded_boxes = torch.cat([x1, y1, x2, y2], dim=-1)
    return decoded_boxes
