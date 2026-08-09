"""YOLO-NAS preprocessing, postprocessing, and checkpoint helpers.

Postprocessing lives in ``libreyolo.postprocess.yolonas`` and is re-exported
here for backward compatibility (along with the resize-size constants it
shares with the preprocess side).
"""

from __future__ import annotations

from typing import Mapping, MutableMapping

from ...utils.lazy import lazy_module


from ...preprocess.yolonas import (  # noqa: F401  (moved; re-exported for backward compatibility)
    YOLO_NAS_POSE_PAD_VALUE,
    preprocess_image,
    preprocess_numpy,
    preprocess_pose_image,
)
from ...postprocess.yolonas import (  # noqa: F401  (backward-compatible re-exports)
    YOLO_NAS_PRE_NMS_TOP_K,
    YOLO_NAS_POSE_RESIZE_SIZE,
    YOLO_NAS_RESIZE_SIZE,
    _extract_decoded_predictions,
    _undo_letterbox_xy,
    _undo_letterbox_xyxy,
    postprocess,
    postprocess_pose,
)


# torch is resolved on first use so this module stays importable in a
# torch-free ONNX deployment (discussions/711).
torch = lazy_module("torch")



def unwrap_yolonas_checkpoint(
    checkpoint: Mapping | MutableMapping,
):
    """Extract the actual state dict from common YOLO-NAS checkpoint layouts.

    Official SuperGradients checkpoints typically store weights under ``net``,
    while training checkpoints may also contain ``ema_net``. Prefer EMA weights
    when present so downstream loading mirrors SG's own behavior.
    """
    if not isinstance(checkpoint, Mapping):
        return checkpoint

    for key in ("ema_net", "net", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value

    return checkpoint
