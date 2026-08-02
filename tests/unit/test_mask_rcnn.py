"""Mask R-CNN family and validation-pipeline smoke tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

pytestmark = pytest.mark.unit


class _TinyMaskModel(nn.Module):
    def __init__(self, size, num_classes, *, return_masks):
        super().__init__()
        self.size = size
        self.num_classes = num_classes
        self.roi_heads = nn.Identity()
        self.roi_heads.return_masks = return_masks


def _patch_tiny_model(monkeypatch):
    import libreyolo.models.mask_rcnn.model as model_module

    monkeypatch.setattr(model_module, "LibreMaskRCNNModel", _TinyMaskModel)


def test_training_is_explicitly_unavailable(monkeypatch):
    from libreyolo import LibreMaskRCNN

    _patch_tiny_model(monkeypatch)
    model = LibreMaskRCNN(None, size="r50", device="cpu")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="coco128.yaml")


def test_task_dispatch_controls_mask_branch(monkeypatch):
    from libreyolo import LibreMaskRCNN
    from libreyolo.models.faster_rcnn.validator import FasterRCNNValidator
    from libreyolo.models.mask_rcnn.validator import MaskRCNNValidator

    _patch_tiny_model(monkeypatch)
    segment = LibreMaskRCNN(None, size="r50", task="segment", device="cpu")
    assert segment.model.roi_heads.return_masks is True
    assert segment.validator_class is MaskRCNNValidator
    del segment

    detect = LibreMaskRCNN(None, size="r50", task="detect", device="cpu")
    assert detect.model.roi_heads.return_masks is False
    assert detect.validator_class is FasterRCNNValidator


def test_validation_preprocessor_matches_inference_pixels_and_scales_targets():
    from libreyolo.models.mask_rcnn.utils import preprocess_numpy
    from libreyolo.validation.preprocessors import FasterRCNNValPreprocessor

    rng = np.random.default_rng(0)
    image_bgr = rng.integers(0, 256, (7, 5, 3), dtype=np.uint8)
    targets = np.array([[1, 2, 3, 6, 4]], dtype=np.float32)
    preprocessor = FasterRCNNValPreprocessor(img_size=(800, 800))
    actual, scaled_targets = preprocessor(
        image_bgr,
        targets,
        input_size=(800, 800),
    )
    expected, _ = preprocess_numpy(image_bgr[:, :, ::-1])

    np.testing.assert_allclose(actual / 255.0, expected, rtol=0, atol=0)
    np.testing.assert_allclose(
        scaled_targets[0],
        np.array([160, 800 / 7 * 2, 480, 800 / 7 * 6, 4]),
        rtol=1e-6,
    )
    assert preprocessor.normalize is True
    assert preprocessor.custom_normalization is False
    assert preprocessor.uses_letterbox is False
    assert preprocessor.wants_unresized_image is True


def test_validator_slices_native_mask_detection_list():
    from libreyolo.models.mask_rcnn.validator import MaskRCNNValidator

    validator = MaskRCNNValidator.__new__(MaskRCNNValidator)
    predictions = [
        {
            "boxes": torch.ones(2, 4),
            "scores": torch.ones(2),
            "labels": torch.ones(2),
            "masks": torch.ones(2, 1, 5, 7),
        }
    ]
    assert validator._slice_batch_predictions(predictions, 0) is predictions[0]
