"""Task-appropriate trained-checkpoint smoke coverage for YOLO-NAS-R (OBB)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from libreyolo import LibreYOLO

from .conftest import YOLONAS_OBB_PARAMS


pytestmark = [pytest.mark.e2e, pytest.mark.network, pytest.mark.yolonas]


@pytest.mark.parametrize("family,size,weights", YOLONAS_OBB_PARAMS)
def test_yolonas_obb_trained_checkpoint_smoke(family, size, weights):
    model = LibreYOLO(weights)
    assert (model.FAMILY, model.size, model.task) == (family, size, "obb")
    assert model.input_size == 1024
    assert model.names[0] == "plane"

    # Non-square on purpose: the OBB preprocessor pads bottom-right, and the
    # inverse transform has to land results back on the original canvas.
    image = Image.fromarray(np.full((480, 800, 3), 127, dtype=np.uint8))
    result = model.predict(image, conf=0.0, max_det=5)[0]

    assert result.obb is not None
    assert result.obb.data.shape[1] == 7
    assert result.boxes.data.shape[1] == 6
    assert len(result.obb) == len(result.boxes) <= 5
    assert result.orig_shape == (480, 800)

    xywhr = np.asarray(result.obb.xywhr)
    if len(xywhr):
        assert (xywhr[:, 0] >= 0).all() and (xywhr[:, 0] <= 800).all()
        assert (xywhr[:, 1] >= 0).all() and (xywhr[:, 1] <= 480).all()
        # Public contract: long side first, angle in [-pi/2, pi/2).
        assert (xywhr[:, 2] >= xywhr[:, 3]).all()
        assert (xywhr[:, 4] >= -np.pi / 2).all() and (xywhr[:, 4] < np.pi / 2).all()


@pytest.mark.parametrize("family,size,weights", YOLONAS_OBB_PARAMS)
def test_yolonas_obb_training_is_rejected(family, size, weights):
    model = LibreYOLO(weights)
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="coco128.yaml", epochs=1)
