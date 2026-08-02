"""Faster R-CNN family and validation-pipeline smoke tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


def test_family_metadata_and_filename_detection():
    from libreyolo import LibreFasterRCNN

    assert LibreFasterRCNN.FAMILY == "faster_rcnn"
    assert LibreFasterRCNN.FILENAME_PREFIX == "LibreFasterRCNN"
    assert LibreFasterRCNN.SUPPORTED_TASKS == ("detect",)
    assert LibreFasterRCNN.DEFAULT_TASK == "detect"
    assert LibreFasterRCNN.TRAIN_CONFIG is None
    assert LibreFasterRCNN.INPUT_SIZES == {"n": 320, "s": 800, "m": 800, "l": 800}
    for size in "nsml":
        assert (
            LibreFasterRCNN.detect_size_from_filename(
                f"LibreFasterRCNN{size}.pt"
            )
            == size
        )
    assert (
        LibreFasterRCNN.detect_size_from_filename(
            "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
        )
        == "l"
    )


def test_training_is_explicitly_unavailable():
    from libreyolo import LibreFasterRCNN

    model = LibreFasterRCNN(None, size="n", device="cpu")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="coco128.yaml")


def test_validation_preprocessor_matches_inference_pixels_and_scales_targets():
    from libreyolo.models.faster_rcnn.utils import preprocess_numpy
    from libreyolo.validation.preprocessors import FasterRCNNValPreprocessor

    rng = np.random.default_rng(0)
    image_bgr = rng.integers(0, 256, (7, 5, 3), dtype=np.uint8)
    targets = np.array([[1, 2, 3, 6, 4]], dtype=np.float32)
    preprocessor = FasterRCNNValPreprocessor(img_size=(320, 320))
    actual, scaled_targets = preprocessor(
        image_bgr, targets, input_size=(320, 320)
    )
    expected, _ = preprocess_numpy(image_bgr[:, :, ::-1])

    np.testing.assert_allclose(actual / 255.0, expected, rtol=0, atol=0)
    np.testing.assert_allclose(
        scaled_targets[0],
        np.array([64, 320 / 7 * 2, 192, 320 / 7 * 6, 4]),
        rtol=1e-6,
    )
    assert preprocessor.normalize is True
    assert preprocessor.custom_normalization is False
    assert preprocessor.uses_letterbox is False
    assert preprocessor.wants_unresized_image is True


def test_validator_slices_native_detection_list():
    from libreyolo.models.faster_rcnn.validator import FasterRCNNValidator

    validator = FasterRCNNValidator.__new__(FasterRCNNValidator)
    predictions = [
        {"boxes": torch.ones(2, 4), "scores": torch.ones(2), "labels": torch.ones(2)}
    ]
    assert validator._slice_batch_predictions(predictions, 0) is predictions[0]
