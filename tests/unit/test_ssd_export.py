"""SSD packed export and backend-adapter contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from libreyolo.backends.base import BaseBackend
from libreyolo.models.ssd.nn import SSDExportWrapper
from libreyolo.models.ssd.utils import preprocess_image
from libreyolo.postprocess.ssd import _decode_boxes, _default_boxes, postprocess
from libreyolo.utils.coco import COCO91_CATEGORY_IDS, COCO91_TO_COCO80

pytestmark = pytest.mark.unit


class _RawHead(nn.Module):
    num_classes = 91

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = images.shape[0]
        regression = torch.arange(8732 * 4, dtype=images.dtype)
        regression = regression.reshape(1, 8732, 4).expand(batch, -1, -1)
        logits = torch.arange(8732 * 91, dtype=images.dtype)
        logits = logits.reshape(1, 8732, 91).expand(batch, -1, -1)
        return {"bbox_regression": regression, "cls_logits": logits}


class _FixedRawHead(nn.Module):
    num_classes = 91

    def __init__(self, outputs: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.outputs = outputs

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        del images
        return self.outputs


class _SSDBackend(BaseBackend):
    def __init__(self) -> None:
        super().__init__(
            model_path="dummy.onnx",
            nb_classes=80,
            device="cpu",
            imgsz=300,
            model_family="ssd",
            model_size="300",
            names={index: f"class_{index}" for index in range(80)},
            task="detect",
            supported_tasks=("detect",),
        )

    def _run_inference(self, blob: np.ndarray) -> list:
        raise NotImplementedError


def _raw_outputs() -> dict[str, torch.Tensor]:
    logits = torch.full((1, 8732, 91), -20.0)
    logits[..., 0] = 0.0
    logits[0, 100, 13] = 20.0
    return {
        "bbox_regression": torch.zeros(1, 8732, 4),
        "cls_logits": logits,
    }


def test_export_wrapper_packs_decoded_boxes_and_mapped_probabilities():
    wrapper = SSDExportWrapper(_RawHead())
    images = torch.zeros(2, 3, 300, 300)
    raw = wrapper.model(images)
    packed = wrapper(images)

    assert packed.shape == (2, 84, 8732)
    anchors = _default_boxes(device=torch.device("cpu"), dtype=torch.float32)
    expected_boxes = _decode_boxes(raw["bbox_regression"][0], anchors).clamp(0, 300)
    expected_scores = raw["cls_logits"].softmax(dim=-1)
    expected_scores = expected_scores[..., list(COCO91_CATEGORY_IDS)]
    torch.testing.assert_close(packed[0, :4].T, expected_boxes)
    torch.testing.assert_close(packed[:, 4:].transpose(1, 2), expected_scores)


def test_backend_parser_reconstructs_native_ssd_postprocess():
    backend = _SSDBackend()
    raw = _raw_outputs()
    packed = SSDExportWrapper(_FixedRawHead(raw))(torch.zeros(1, 3, 300, 300))
    actual = backend._postprocess(
        packed,
        conf_thres=0.5,
        iou_thres=0.45,
        original_size=(600, 150),
        input_size=300,
        max_det=10,
    )
    expected = postprocess(
        raw,
        conf_thres=0.5,
        iou_thres=0.45,
        original_size=(600, 150),
        max_det=10,
        class_map=COCO91_TO_COCO80,
    )

    np.testing.assert_array_equal(actual["boxes"].numpy(), expected["boxes"])
    np.testing.assert_array_equal(actual["scores"].numpy(), expected["scores"])
    np.testing.assert_array_equal(actual["classes"].numpy(), expected["classes"])


def test_backend_uses_ssd_fixed_resize_preprocessing():
    backend = _SSDBackend()
    pixels = np.arange(73 * 119 * 3, dtype=np.uint8).reshape(73, 119, 3)
    image = Image.fromarray(pixels, mode="RGB")
    actual = backend._preprocess(image, 300, "auto")[0]
    expected = preprocess_image(image)[0]
    assert torch.equal(actual, expected)


def test_export_rejects_non_native_canvas_and_embedded_nms():
    from libreyolo import LibreSSD

    model = LibreSSD(None, device="cpu")
    with pytest.raises(ValueError, match="requires imgsz=300"):
        model.export(imgsz=320)
    with pytest.raises(NotImplementedError, match="nms=True"):
        model.export(imgsz=300, nms=True)
