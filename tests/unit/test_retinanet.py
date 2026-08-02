"""RetinaNet family skeleton and registry tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torchvision.models.detection.transform import GeneralizedRCNNTransform

pytestmark = [pytest.mark.unit, pytest.mark.retinanet]


def test_family_metadata_and_filename_detection():
    from libreyolo import LibreRetinaNet

    assert LibreRetinaNet.FAMILY == "retinanet"
    assert LibreRetinaNet.FILENAME_PREFIX == "LibreRetinaNet"
    assert LibreRetinaNet.INPUT_SIZES == {"r50": 800, "r50v2": 800}
    assert LibreRetinaNet.SUPPORTED_TASKS == ("detect",)
    assert LibreRetinaNet.DEFAULT_TASK == "detect"
    assert LibreRetinaNet.TRAIN_CONFIG is None
    assert LibreRetinaNet.detect_size_from_filename("LibreRetinaNetr50.pt") == "r50"
    assert LibreRetinaNet.detect_size_from_filename("LibreRetinaNetr50v2.pt") == "r50v2"
    assert (
        LibreRetinaNet.detect_size_from_filename(
            "retinanet_resnet50_fpn_v2_coco-5905b1c5.pth"
        )
        == "r50v2"
    )


def test_skeleton_constructs_and_training_is_explicitly_unavailable():
    from libreyolo import LibreRetinaNet

    model = LibreRetinaNet(None, size="r50", device="cpu")
    assert model.family == "retinanet"
    assert model.task == "detect"
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="coco128.yaml")


@pytest.mark.parametrize("shape", [(73, 121), (121, 73)])
def test_preprocess_tensor_exactly_matches_upstream_transform(shape):
    from libreyolo.models.retinanet.utils import (
        IMAGE_MEAN,
        IMAGE_STD,
        preprocess_tensor,
    )

    generator = torch.Generator().manual_seed(sum(shape))
    image = torch.rand((3, *shape), generator=generator)
    transform = GeneralizedRCNNTransform(
        min_size=64,
        max_size=107,
        image_mean=list(IMAGE_MEAN),
        image_std=list(IMAGE_STD),
    )
    expected, _ = transform([image], None)
    actual = preprocess_tensor(image, input_size=64)
    assert torch.equal(actual, expected.tensors[0])


def test_validation_preprocessor_matches_inference_and_scales_targets():
    from libreyolo.models.retinanet.utils import preprocess_numpy, resize_scale
    from libreyolo.validation.preprocessors import RetinaNetValPreprocessor

    rng = np.random.default_rng(0)
    image_bgr = rng.integers(0, 256, (41, 67, 3), dtype=np.uint8)
    targets = np.array([[1, 2, 30, 35, 4]], dtype=np.float32)
    preprocessor = RetinaNetValPreprocessor(img_size=(64, 64))
    actual, scaled_targets = preprocessor(image_bgr, targets, input_size=(64, 64))
    expected, ratio = preprocess_numpy(image_bgr[:, :, ::-1], 64)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(scaled_targets[0, :4], targets[0, :4] * ratio)
    assert ratio == resize_scale((67, 41), 64)
    assert preprocessor.normalize is False
    assert preprocessor.custom_normalization is True
    assert preprocessor.uses_letterbox is True
    assert preprocessor.wants_unresized_image is True


def test_validator_forces_single_image_batches(monkeypatch):
    from libreyolo.models.retinanet.validator import RetinaNetValidator
    from libreyolo.validation.detection_validator import DetectionValidator

    monkeypatch.setattr(DetectionValidator, "_setup_dataloader", lambda _self: "loader")
    validator = RetinaNetValidator.__new__(RetinaNetValidator)
    validator.config = SimpleNamespace(batch_size=4)
    assert validator._setup_dataloader() == "loader"
    assert validator.config.batch_size == 1
