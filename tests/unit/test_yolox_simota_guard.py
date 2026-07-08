"""Regression: SimOTA must survive a GT with no candidate anchors.

A degenerate/off-grid box (mosaic clipping, bad label) makes
``get_geometry_constraint`` return an all-False anchor filter; without the
empty-assignment guard, ``simota_matching``'s ``torch.topk`` then crashes on an
empty cost row mid-epoch. The guard was first added to the YOLOv7 adaptation of
this code (``models/yolo7/loss.py``) and back-ported here to the source.
"""

import pytest
import torch

from libreyolo.models.yolox.nn import LibreYOLOXModel

pytestmark = pytest.mark.unit


def test_simota_survives_offgrid_gt():
    torch.manual_seed(0)
    m = LibreYOLOXModel(config="n", nb_classes=3)
    m.train()
    imgs = torch.rand(1, 3, 256, 256)
    # (cls, cx, cy, w, h) pixels — centre far outside the image/grid.
    targets = torch.zeros(1, 10, 5)
    targets[0, 0] = torch.tensor([0.0, 5000.0, 5000.0, 20.0, 20.0])

    out = m(imgs, targets)
    assert torch.isfinite(out["total_loss"]).all()
    assert out["total_loss"].item() > 0  # objectness BCE over all-negatives
