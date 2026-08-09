"""End-to-end preprocessing, prediction, and UI smoke tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.unit


def test_validation_preprocessor_matches_inference_transform_exactly():
    from libreyolo.models.deformable_detr.utils import preprocess_numpy
    from libreyolo.validation.preprocessors import DeformableDETRValPreprocessor

    rng = np.random.default_rng(2020)
    image_bgr = rng.integers(0, 256, (11, 7, 3), dtype=np.uint8)
    preprocessor = DeformableDETRValPreprocessor(img_size=(64, 64))
    actual, targets = preprocessor(
        image_bgr,
        np.zeros((0, 5), dtype=np.float32),
        (64, 64),
    )
    expected, ratio = preprocess_numpy(image_bgr[:, :, ::-1], input_size=64)

    np.testing.assert_array_equal(actual, expected)
    assert targets.shape[1] == 5
    assert ratio == 1.0
    assert preprocessor.custom_normalization is True
    assert preprocessor.uses_letterbox is False
    assert preprocessor.wants_unresized_image is True


def test_public_predict_pipeline_returns_results_for_rectangular_input():
    from libreyolo import LibreDeformableDETR
    from libreyolo.utils.results import Results

    model = LibreDeformableDETR(None, size="r50ss", device="cpu")
    image = np.zeros((37, 53, 3), dtype=np.uint8)
    result = model.predict(image, imgsz=64, conf=0.999, max_det=5)

    assert isinstance(result, Results)
    assert result.orig_shape == (37, 53)
    assert result.boxes is not None
    assert len(result.boxes) <= 5
    if len(result.boxes):
        boxes = result.boxes.xyxy.numpy()
        assert (boxes[:, 0::2] >= 0).all() and (boxes[:, 0::2] <= 53).all()
        assert (boxes[:, 1::2] >= 0).all() and (boxes[:, 1::2] <= 37).all()


def test_cli_and_ui_discover_all_five_variants():
    from libreyolo.cli.config import get_all_cli_names, resolve_model_name
    from libreyolo.ui.server import _resolve_download_url

    names = set(get_all_cli_names())
    for size in ("r50ss", "r50ssdc5", "r50", "r50refine", "r50twostage"):
        name = f"deformable_detr-{size}"
        filename = f"LibreDeformableDETR{size}.pt"
        assert name in names
        assert Path(resolve_model_name(name)).name == filename
        assert _resolve_download_url(name) == (
            f"https://huggingface.co/LibreYOLO/LibreDeformableDETR{size}/"
            f"resolve/main/{filename}"
        )


def test_ui_inference_smoke_with_cached_native_model(tmp_path):
    from libreyolo import LibreDeformableDETR
    from libreyolo.ui.server import _UIState

    model_name = "deformable_detr-r50ss"
    model = LibreDeformableDETR(None, size="r50ss", device="cpu")
    model.input_size = 64

    state = _UIState(device="cpu")
    state._input_dir = tmp_path / "inputs"
    state._input_dir.mkdir()
    state.run_dir = tmp_path / "run"
    state.run_dir.mkdir()
    state._models[model_name] = model

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
