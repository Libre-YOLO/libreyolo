"""Shared train-time augmentation library.

Every model family's augmentation pipeline lives here (shared ops + per-family
recipe modules). The historical per-model ``transforms`` files are re-export
shims over this package.

The numpy/cv2 core (``boxes``, ``color``, ``geometry``, ``mosaic``,
``segments``, and the numpy recipe modules) is importable without torch.
The torchvision-v2 pipelines live in the ``detr`` submodule, which is
intentionally NOT imported here — import it explicitly
(``from libreyolo.data.augment import detr`` or via the model shims) so
numpy-only consumers never pay the torch import.
"""

from .boxes import adjust_box_anns, cxcywh2xyxy, xyxy2cxcywh
from .color import augment_hsv
from .copy_paste import copy_paste
from .geometry import (
    apply_affine_to_bboxes,
    get_affine_matrix,
    get_aug_params,
    mirror,
    preproc,
    random_affine,
)
from .mosaic import get_mosaic_coordinate
from .yolox import MosaicMixupDataset, TrainTransform, ValTransform

__all__ = [
    "adjust_box_anns",
    "cxcywh2xyxy",
    "xyxy2cxcywh",
    "augment_hsv",
    "copy_paste",
    "apply_affine_to_bboxes",
    "get_affine_matrix",
    "get_aug_params",
    "mirror",
    "preproc",
    "random_affine",
    "get_mosaic_coordinate",
    "MosaicMixupDataset",
    "TrainTransform",
    "ValTransform",
]
