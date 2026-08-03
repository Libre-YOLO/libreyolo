"""CenterNet registration, preprocessing, decoding, and validation contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo import LibreCenterNet
from libreyolo.models.centernet.utils import preprocess_bgr, preprocess_numpy
from libreyolo.models.registry import group_of
from libreyolo.postprocess.centernet import decode_centernet, postprocess
from libreyolo.validation.preprocessors import CenterNetValPreprocessor

pytestmark = pytest.mark.unit


def test_family_metadata_and_inference_only_contract():
    assert LibreCenterNet.FAMILY == "centernet"
    assert LibreCenterNet.FILENAME_PREFIX == "LibreCenterNet"
    assert LibreCenterNet.INPUT_SIZES == {"resdcn18": 512, "dla34": 512}
    assert LibreCenterNet.SUPPORTED_TASKS == ("detect",)
    assert LibreCenterNet.TRAIN_CONFIG is None
    assert group_of("centernet") == "g3"
    with pytest.raises(NotImplementedError, match="inference-only"):
        LibreCenterNet.train(object())


def test_official_filename_and_prefixed_state_recognition():
    state = {
        "module.conv1.weight": torch.zeros(64, 3, 7, 7),
        "module.deconv_layers.0.conv_offset_mask.weight": torch.zeros(27, 512, 3, 3),
        "module.hm.2.weight": torch.zeros(80, 64, 1, 1),
        "module.wh.2.weight": torch.zeros(2, 64, 1, 1),
        "module.reg.2.weight": torch.zeros(2, 64, 1, 1),
    }
    converted = LibreCenterNet.convert_upstream_state_dict(state)
    assert converted is not None
    assert LibreCenterNet.can_load(converted)
    assert LibreCenterNet.detect_size(state) == "resdcn18"
    assert LibreCenterNet.detect_nb_classes(state) == 80
    assert (
        LibreCenterNet.detect_size_from_filename("ctdet_coco_resdcn18.pth")
        == "resdcn18"
    )
    assert LibreCenterNet.detect_size_from_filename("ctdet_coco_dla_2x.pth") == "dla34"


def test_rgb_and_bgr_preprocessing_are_the_same_pinned_color_contract():
    rng = np.random.default_rng(637)
    bgr = rng.integers(0, 256, size=(47, 83, 3), dtype=np.uint8)
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    direct, direct_ratio = preprocess_bgr(bgr, input_size=128)
    public, public_ratio = preprocess_numpy(rgb, input_size=128)
    np.testing.assert_array_equal(public, direct)
    assert public_ratio == direct_ratio == 128 / 83


def test_decode_and_affine_postprocess_restore_one_known_peak():
    heatmap = torch.full((1, 2, 32, 32), -20.0)
    width_height = torch.zeros((1, 2, 32, 32))
    regression = torch.zeros((1, 2, 32, 32))
    heatmap[0, 1, 16, 10] = 20.0
    width_height[0, :, 16, 10] = torch.tensor([4.0, 6.0])
    regression[0, :, 16, 10] = torch.tensor([0.25, 0.5])

    decoded = decode_centernet(heatmap, width_height, regression, topk=10)
    assert decoded.shape == (1, 10, 6)
    result = postprocess(
        {"hm": heatmap, "wh": width_height, "reg": regression},
        conf_thres=0.5,
        original_size=(100, 50),
        input_size=128,
        max_det=10,
        topk=10,
    )
    assert result["num_detections"] == 1
    np.testing.assert_allclose(
        result["boxes"][0],
        np.array([25.78125, 17.1875, 38.28125, 35.9375], dtype=np.float32),
        rtol=0,
        atol=1e-5,
    )
    assert result["classes"].tolist() == [1]
    assert result["scores"][0] > 0.999


def test_validation_preprocessor_is_custom_normalized_but_not_letterboxed():
    preprocessor = CenterNetValPreprocessor((128, 128), max_labels=4)
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    targets = np.array([[10, 5, 30, 25, 1]], dtype=np.float32)
    processed, scaled = preprocessor(image, targets, (128, 128))
    assert processed.shape == (3, 128, 128)
    assert preprocessor.custom_normalization is True
    assert preprocessor.uses_letterbox is False
    assert preprocessor.wants_unresized_image is True
    np.testing.assert_allclose(scaled[0], [12.8, 12.8, 38.4, 64.0, 1.0])
