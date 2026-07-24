"""D-FINE training transforms (compatibility shim + family subclasses).

The pipeline implementation moved to :mod:`libreyolo.data.augment.detr`
(merged with the formerly line-for-line DEIM twin). The D-FINE classes here
deliberately do NOT set ``wants_unresized_image`` — that attribute is
DEIM-specific and flips the dataset's ``pull_item`` input path; D-FINE keeps
receiving the pre-letterboxed frame, exactly as before the merge.

The module re-exports the full historical surface (including stdlib and
third-party module attributes) so existing imports and monkeypatch dotted
paths keep working. Also consumed by the RF-DETR and EC trainers.
"""

from __future__ import annotations

import random  # noqa: F401  (historical module attribute)
from typing import List, Optional, Sequence  # noqa: F401

import numpy as np  # noqa: F401  (historical module attribute)
import torch  # noqa: F401  (historical module attribute)
import torch.nn.functional as F  # noqa: F401
from torchvision import tv_tensors  # noqa: F401
from torchvision.transforms import v2 as tv2  # noqa: F401

from libreyolo.data.augment.detr import (  # noqa: F401
    DetrMultiScaleCollate,
    DetrPassThroughDataset,
    DetrTrainTransform,
    _generate_scales,
    _labels_at_index_2,
)
from libreyolo.data.augment.rfdetr import (  # noqa: F401
    RFDETRSegPassThroughDataset,
    RFDETRSegTransform,
)


class DFINETrainTransform(DetrTrainTransform):
    """D-FINE per-sample transform (no ``wants_unresized_image`` — see above)."""


class DFINEPassThroughDataset(DetrPassThroughDataset):
    """D-FINE identity dataset wrapper (no mosaic)."""

    _default_transform_cls = DFINETrainTransform


DFINEMultiScaleCollate = DetrMultiScaleCollate


class DFINESegTransform(RFDETRSegTransform):
    """D-FINE segment transform: square resize + masks, without ImageNet norm."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("imagenet_norm", False)
        super().__init__(*args, **kwargs)


DFINESegPassThroughDataset = RFDETRSegPassThroughDataset
