"""End-to-end native preprocessing, prediction, and UI smoke tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.dinodetr]


@pytest.fixture(scope="module")
def native_model():
    from libreyolo import LibreDINODETR

    return LibreDINODETR(None, size="r50", device="cpu")


def test_validation_preprocessor_matches_inference_transform_exactly():
    from libreyolo.models.deformable_detr.utils import preprocess_numpy
    from libreyolo.validation.preprocessors import DeformableDETRValPreprocessor

    rng = np.random.default_rng(637)
    image_bgr = rng.integers(0, 256, (11, 7, 3), dtype=np.uint8)
    preprocessor = DeformableDETRValPreprocessor(img_size=(64, 64))
    actual, targets = preprocessor(
        image_bgr, np.zeros((0, 5), dtype=np.float32), (64, 64)
    )
    expected, ratio = preprocess_numpy(image_bgr[:, :, ::-1], input_size=64)
    np.testing.assert_array_equal(actual, expected)
    assert targets.shape[1] == 5
    assert ratio == 1.0


def test_public_predict_returns_results_for_rectangular_input(native_model):
    from libreyolo.utils.results import Results

    image = np.zeros((37, 53, 3), dtype=np.uint8)
    result = native_model.predict(image, imgsz=240, conf=0.999, max_det=5)
    assert isinstance(result, Results)
    assert result.orig_shape == (37, 53)
    assert result.boxes is not None
    assert len(result.boxes) <= 5


def test_ui_inference_with_cached_native_model(native_model, tmp_path):
    from libreyolo.ui.server import _UIState

    model_name = "dinodetr-r50"
    native_model.input_size = 240
    state = _UIState(device="cpu")
    state._input_dir = tmp_path / "inputs"
    state._input_dir.mkdir()
    state.run_dir = tmp_path / "run"
    state.run_dir.mkdir()
    state._models[model_name] = native_model

    image = Image.new("RGB", (53, 37), color=(32, 64, 96))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    response = state.infer(
        model_name,
        conf=0.999,
        filename="rectangular.png",
        data=encoded.getvalue(),
    )
    assert response["task"] == "detect"
    assert response["count"] >= 0
    assert response["rendered"].startswith("data:image/")
    assert Path(response["saved"]).exists()
