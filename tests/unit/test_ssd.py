"""SSD family shell, naming, and maturity tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_family_metadata_and_filename_detection():
    from libreyolo import LibreSSD

    assert LibreSSD.FAMILY == "ssd"
    assert LibreSSD.FILENAME_PREFIX == "LibreSSD"
    assert LibreSSD.INPUT_SIZES == {"300": 300}
    assert LibreSSD.SUPPORTED_TASKS == ("detect",)
    assert LibreSSD.DEFAULT_TASK == "detect"
    assert LibreSSD.TRAIN_CONFIG is None
    assert LibreSSD.detect_size_from_filename("LibreSSD300.pt") == "300"
    assert LibreSSD.detect_size_from_filename("LibreSSD30.pt") is None


def test_family_constructs_and_training_is_explicitly_unavailable():
    from libreyolo import LibreSSD

    model = LibreSSD(None, size="300", device="cpu")
    assert model.family == "ssd"
    assert model.model.num_classes == 91
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="coco128.yaml")


def test_raw_head_forward_shapes():
    import torch

    from libreyolo.models.ssd.nn import LibreSSDModel

    model = LibreSSDModel(num_classes=91).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 300, 300))

    assert output["bbox_regression"].shape == (1, 8732, 4)
    assert output["cls_logits"].shape == (1, 8732, 91)


def test_state_dict_layout_matches_reference_architecture():
    from torchvision.models.detection import ssd300_vgg16

    from libreyolo.models.ssd.nn import LibreSSDModel

    reference = ssd300_vgg16(weights=None, weights_backbone=None)
    ours = LibreSSDModel(num_classes=91)
    reference_shapes = {
        key: tuple(value.shape) for key, value in reference.state_dict().items()
    }
    our_shapes = {key: tuple(value.shape) for key, value in ours.state_dict().items()}
    assert our_shapes == reference_shapes


def test_preprocess_matches_reference_fixed_resize_exactly():
    import numpy as np
    import torch
    from torchvision.models.detection.transform import GeneralizedRCNNTransform

    from libreyolo.models.ssd.utils import preprocess_numpy

    image = np.arange(73 * 119 * 3, dtype=np.uint8).reshape(73, 119, 3)
    source = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    transform = GeneralizedRCNNTransform(
        300,
        300,
        [0.48235, 0.45882, 0.40784],
        [1.0 / 255.0] * 3,
        size_divisible=1,
        fixed_size=(300, 300),
    )
    reference = transform([source])[0].tensors[0]
    actual, ratio = preprocess_numpy(image)

    assert ratio == 1.0
    assert torch.equal(torch.from_numpy(actual), reference)


def test_validation_preprocessor_matches_native_input_and_geometry():
    import numpy as np

    from libreyolo.models.ssd.utils import preprocess_numpy
    from libreyolo.validation.preprocessors import SSDValPreprocessor

    bgr = np.arange(73 * 119 * 3, dtype=np.uint8).reshape(73, 119, 3)
    targets = np.array([[10.0, 11.0, 60.0, 50.0, 4.0]], dtype=np.float32)
    preprocessor = SSDValPreprocessor(img_size=(300, 300), max_labels=4)

    image, padded_targets = preprocessor(bgr, targets, (300, 300))
    reference, _ = preprocess_numpy(bgr[:, :, ::-1].copy(), 300)

    assert np.array_equal(image, reference)
    assert preprocessor.custom_normalization is True
    assert preprocessor.uses_letterbox is False
    assert preprocessor.wants_unresized_image is True
    assert np.allclose(
        padded_targets[0],
        [10.0 * 300 / 119, 11.0 * 300 / 73, 60.0 * 300 / 119, 50.0 * 300 / 73, 4.0],
    )
    assert not padded_targets[1:].any()
