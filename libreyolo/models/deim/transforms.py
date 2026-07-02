"""DEIM training transforms (compatibility shim + family subclasses).

The pipeline implementation moved to :mod:`libreyolo.data.augment.detr`
(merged with the formerly line-for-line D-FINE twin). The DEIM classes here
are thin subclasses, NOT aliases: ``wants_unresized_image = True`` is a
DEIM-specific class attribute that ``YOLODataset``/``COCODataset.pull_item``
feature-sniff to hand DEIM the original-resolution image with un-letterboxed
labels. D-FINE's classes do not set it and keep the pre-letterboxed path.

The module re-exports the full historical surface (including stdlib and
third-party module attributes) so existing imports and monkeypatch dotted
paths keep working.
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


class DEIMTrainTransform(DetrTrainTransform):
    """DEIM per-sample transform (see module docstring for the class attr)."""

    # Sniffed by YOLODataset/COCODataset.pull_item — DEIM receives the
    # original-resolution image and labels divided back by the letterbox ratio.
    wants_unresized_image = True


class DEIMPassThroughDataset(DetrPassThroughDataset):
    """DEIM identity dataset wrapper (no mosaic)."""

    _default_transform_cls = DEIMTrainTransform


DEIMMultiScaleCollate = DetrMultiScaleCollate
