"""Task-appropriate trained-checkpoint smoke coverage for RT-DETRv2 OBB."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from libreyolo import LibreYOLO

from .conftest import RTDETRV2_OBB_PARAMS


pytestmark = [pytest.mark.e2e, pytest.mark.rtdetrv2]


@pytest.mark.parametrize("family,size,weights", RTDETRV2_OBB_PARAMS)
def test_rtdetrv2_obb_trained_checkpoint_smoke(family, size, weights):
    staged = Path(os.getenv("LIBREYOLO_RTDETRV2_OBB_WEIGHTS", weights))
    if not staged.is_file():
        pytest.skip(
            "stage a converted official checkpoint with LIBREYOLO_RTDETRV2_OBB_WEIGHTS"
        )

    model = LibreYOLO(str(staged), device="cuda")
    assert (model.FAMILY, model.size, model.task) == (family, size, "obb")

    image = Image.fromarray(np.full((480, 800, 3), 127, dtype=np.uint8))
    result = model.predict(image, conf=0.0, max_det=5)[0]

    assert result.obb is not None
    assert result.obb.data.shape[1] == 7
    assert result.boxes.data.shape[1] == 6
    assert len(result.obb) == len(result.boxes) <= 5
    assert result.orig_shape == (480, 800)
