"""Inference-side preprocess for LibreYOLO RF-DETR.

Behavior matches upstream RF-DETR (https://github.com/roboflow/rf-detr) so weights
load and produce numerically equivalent detections.

Postprocessing lives in ``libreyolo.postprocess.rfdetr`` and is re-exported
here for backward compatibility.
"""


from ...postprocess.rfdetr import postprocess  # noqa: F401  (backward-compatible re-export)
from ...preprocess.rfdetr import (  # noqa: F401  (moved; re-exported for backward compatibility)
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_numpy,
)



