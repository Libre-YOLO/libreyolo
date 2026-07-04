"""YOLOX-style data augmentation (compatibility shim).

The implementation moved to :mod:`libreyolo.data.augment`. This module
re-exports the full historical surface — including the stdlib/third-party
module attributes (``math``, ``random``, ``cv2``, ``np``) that existed here
before the move, so monkeypatching via this module's dotted path and
``from libreyolo.training.augment import X`` both keep working.
"""

import math  # noqa: F401  (historical module attribute)
import random  # noqa: F401  (historical module attribute)

import cv2  # noqa: F401  (historical module attribute)
import numpy as np  # noqa: F401  (historical module attribute)

from libreyolo.data.augment.boxes import (  # noqa: F401
    adjust_box_anns,
    cxcywh2xyxy,
    xyxy2cxcywh,
)
from libreyolo.data.augment.color import augment_hsv  # noqa: F401
from libreyolo.data.augment.geometry import (  # noqa: F401
    apply_affine_to_bboxes,
    get_affine_matrix,
    get_aug_params,
    mirror,
    preproc,
    random_affine,
)
from libreyolo.data.augment.mosaic import get_mosaic_coordinate  # noqa: F401
from libreyolo.data.augment.yolox import (  # noqa: F401
    MosaicMixupDataset,
    TrainTransform,
    ValTransform,
)
