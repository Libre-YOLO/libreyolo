"""EfficientDet one-output export and backend contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from libreyolo.backends.base import BaseBackend
from libreyolo.models.efficientdet.nn import EfficientDetExportWrapper
from libreyolo.postprocess.efficientdet import decode_candidates

pytestmark = pytest.mark.unit


class _RawFixture(nn.Module):
    num_classes = 90

    def __init__(self, input_size: int = 128) -> None:
        super().__init__()
        self.input_size = input_size
        for index, stride in enumerate((8, 16, 32, 64, 128)):
            cells = input_size // stride
            classes = torch.full((1, 810, cells, cells), -100.0)
            boxes = torch.zeros((1, 36, cells, cells))
            if index == 0:
                classes[0, 0, 0, 0] = 10.0
                classes[0, 11, 0, 0] = 9.5
                classes[0, 12, 0, 0] = 9.0
            self.register_buffer(f"classes_{index}", classes)
            self.register_buffer(f"boxes_{index}", boxes)

    def forward(self, images: torch.Tensor):
        batch = images.shape[0]
        dependency = images.mean() * 0.0
        classes = [
            getattr(self, f"classes_{index}").expand(batch, -1, -1, -1)
            + dependency
            for index in range(5)
        ]
        boxes = [
            getattr(self, f"boxes_{index}").expand(batch, -1, -1, -1)
            + dependency
            for index in range(5)
        ]
        return classes, boxes


class _BackendFixture(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            model_path="fixture",
            nb_classes=80,
            device="cpu",
            imgsz=128,
            model_family="efficientdet",
            model_size="d0",
            names={index: f"class_{index}" for index in range(80)},
            task="detect",
            supported_tasks=("detect",),
        )

    def _run_inference(self, blob: np.ndarray) -> list:
        raise NotImplementedError


def test_export_wrapper_matches_native_candidate_decode():
    raw_model = _RawFixture()
    wrapper = EfficientDetExportWrapper(
        raw_model, input_size=128, max_candidates=3, sparse_coco=True
    )
    images = torch.rand(1, 3, 128, 128)
    expected = decode_candidates(
        raw_model(images), input_size=128, max_candidates=3, sparse_coco=True
    )
    actual = wrapper(images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (1, 3, 6)


def test_export_wrapper_traces_as_one_tensor():
    wrapper = EfficientDetExportWrapper(
        _RawFixture(), input_size=128, max_candidates=3, sparse_coco=True
    ).eval()
    images = torch.rand(1, 3, 128, 128)
    traced = torch.jit.trace(wrapper, images, check_trace=False)
    output = traced(images)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 6)


def test_backend_parser_scales_and_uses_class_aware_nms():
    backend = _BackendFixture()
    output = np.array(
        [
            [
                [20.0, 10.0, 100.0, 70.0, 0.95, 0.0],
                [20.0, 10.0, 100.0, 70.0, 0.90, 1.0],
                [20.0, 10.0, 100.0, 70.0, 0.99, -1.0],
                [0.0, 0.0, 10.0, 10.0, 0.10, 2.0],
            ]
        ],
        dtype=np.float32,
    )
    boxes, scores, classes, masks = backend._parse_outputs(
        [output], 128, (64, 32), conf=0.25, ratio=2.0
    )
    result = backend._build_result(
        boxes,
        scores,
        classes,
        orig_shape=(32, 64),
        image_path=None,
        iou=0.5,
        classes=None,
        max_det=100,
    )

    assert masks is None
    assert len(result.boxes) == 2
    np.testing.assert_allclose(
        result.boxes.xyxy.numpy(),
        [[10.0, 5.0, 50.0, 32.0], [10.0, 5.0, 50.0, 32.0]],
    )
    np.testing.assert_array_equal(result.boxes.cls.numpy(), [0.0, 1.0])


def test_backend_uses_efficientdet_validation_preprocessor():
    backend = _BackendFixture()
    preprocessor = backend._get_val_preprocessor()
    assert type(preprocessor).__name__ == "EfficientDetValPreprocessor"
