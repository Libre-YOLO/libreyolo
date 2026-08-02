"""Deformable DETR export and exported-backend contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


def _stub_backend(*, nb_classes: int = 80):
    from libreyolo.backends.base import BaseBackend

    class StubBackend(BaseBackend):
        def _run_inference(self, blob: np.ndarray) -> list:
            del blob
            return []

    return StubBackend(
        model_path="fixture.onnx",
        nb_classes=nb_classes,
        device="cpu",
        imgsz=64,
        model_family="deformable_detr",
        model_size="r50ss",
        names={index: f"class_{index}" for index in range(nb_classes)},
        task="detect",
        supported_tasks=("detect",),
        default_task="detect",
    )


def test_export_wrapper_returns_only_final_detection_tensors():
    from libreyolo.models.deformable_detr.nn import DeformableDETRExportWrapper

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return {
                "pred_logits": x.mean(dim=(-2, -1)),
                "pred_boxes": x.amax(dim=(-2, -1)),
                "aux_outputs": [{"ignored": x}],
            }

    image = torch.rand(1, 3, 8, 8)
    logits, boxes = DeformableDETRExportWrapper(FakeModel())(image)
    torch.testing.assert_close(logits, image.mean(dim=(-2, -1)))
    torch.testing.assert_close(boxes, image.amax(dim=(-2, -1)))


def test_backend_preprocess_and_validation_adapter_use_family_transform():
    from libreyolo.models.deformable_detr.utils import preprocess_numpy
    from libreyolo.validation.preprocessors import DeformableDETRValPreprocessor

    backend = _stub_backend()
    image = np.random.default_rng(2020).integers(0, 256, (11, 7, 3), dtype=np.uint8)
    tensor, original, original_size = backend._preprocess_deformable_detr(
        Image.fromarray(image), 64, "rgb"
    )
    expected, _ = preprocess_numpy(image, 64)

    np.testing.assert_array_equal(tensor.numpy()[0], expected)
    assert original.size == (7, 11)
    assert original_size == (7, 11)
    assert isinstance(
        backend._get_val_preprocessor(img_size=64), DeformableDETRValPreprocessor
    )


def test_backend_removes_unused_coco_columns_before_topk():
    from libreyolo.backends.base import _is_nms_free_family
    from libreyolo.utils.coco import COCO91_CATEGORY_IDS

    backend = _stub_backend()
    logits = np.full((1, 8, 91), -10.0, dtype=np.float32)
    boxes = np.full((1, 8, 4), 0.5, dtype=np.float32)
    unused = sorted(set(range(91)) - set(COCO91_CATEGORY_IDS))
    logits[:, :, unused] = 10.0
    for query, category_id in enumerate(COCO91_CATEGORY_IDS[:5]):
        logits[0, query, category_id] = 5.0

    parsed_boxes, scores, classes = backend._parse_deformable_detr(
        [logits, boxes], 100, 80, conf=0.0, max_det=5
    )

    assert _is_nms_free_family("deformable_detr") is True
    assert parsed_boxes.shape == (5, 4)
    assert set(classes.tolist()) == {0, 1, 2, 3, 4}
    np.testing.assert_allclose(scores, 1.0 / (1.0 + np.exp(-5.0)), atol=1e-6)


def test_export_support_is_onnx_validated_and_ncnn_blocked():
    from libreyolo.export.onnx import (
        _requires_onnx_opset17,
        _uses_dfine_style_export_wrapper,
    )
    from libreyolo.export.support import get_support

    assert _uses_dfine_style_export_wrapper("deformable_detr") is True
    assert _requires_onnx_opset17("deformable_detr") is True
    assert get_support("deformable_detr", "detect", "onnx").tier == "validated"
    assert get_support("deformable_detr", "detect", "ncnn").tier == "blocked"


def test_two_stage_onnx_export_uses_safe_cpu_trace(tmp_path):
    from libreyolo import LibreDeformableDETR
    from libreyolo.export.exporter import OnnxExporter

    model = LibreDeformableDETR(None, size="r50twostage", nb_classes=3, device="cpu")
    exporter = OnnxExporter(model)
    with pytest.warns(RuntimeWarning, match="traced on CPU"):
        _, device, _ = exporter._resolve_params(
            str(tmp_path / "two-stage.onnx"), 128, "cuda", False, False
        )
    assert device == torch.device("cpu")
    with pytest.raises(NotImplementedError, match="validated in FP32 only"):
        exporter._resolve_params(
            str(tmp_path / "two-stage-fp16.onnx"), 128, "cuda", True, False
        )


def test_onnx_roundtrip_preserves_raw_and_public_predictions(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from libreyolo import LibreDeformableDETR, LibreYOLO
    from libreyolo.export.exporter import OnnxExporter

    torch.manual_seed(2020)
    model = LibreDeformableDETR(None, size="r50ss", nb_classes=3, device="cpu")
    model.model.eval()
    tensor = torch.rand(1, 3, 64, 64)
    exporter = OnnxExporter(model)
    with (
        exporter._model_context("cpu", False, False, 1, (64, 64)) as (wrapped, _),
        torch.inference_mode(),
    ):
        expected = tuple(output.numpy() for output in wrapped(tensor))

    artifact = model.export(
        format="onnx",
        imgsz=64,
        dynamic=False,
        simplify=False,
        output_path=str(tmp_path / "LibreDeformableDETRr50ss.onnx"),
    )
    graph = onnx.load(artifact)
    onnx.checker.check_model(graph)
    assert [item.name for item in graph.graph.input] == ["images"]
    assert [item.name for item in graph.graph.output] == [
        "pred_logits",
        "pred_boxes",
    ]
    assert max(item.version for item in graph.opset_import) >= 16
    metadata = {item.key: item.value for item in graph.metadata_props}
    assert metadata["model_family"] == "deformable_detr"
    assert metadata["model_size"] == "r50ss"
    assert metadata["imgsz"] == "64"

    backend = LibreYOLO(artifact, device="cpu")
    actual = backend._run_inference(tensor.numpy())
    assert len(actual) == len(expected) == 2
    for converted, native in zip(actual, expected):
        np.testing.assert_allclose(converted, native, rtol=2e-4, atol=2e-5)

    image = np.random.default_rng(51).integers(0, 256, (37, 53, 3), dtype=np.uint8)
    native_result = model.predict(image, imgsz=64, conf=0.0, max_det=20)
    onnx_result = backend.predict(image, conf=0.0, max_det=20)
    native_boxes = native_result.boxes.data.numpy()
    onnx_boxes = onnx_result.boxes.data.numpy()
    assert onnx_boxes.shape == native_boxes.shape
    np.testing.assert_allclose(
        onnx_boxes[:, :5], native_boxes[:, :5], rtol=2e-3, atol=1e-3
    )
    np.testing.assert_array_equal(onnx_boxes[:, 5], native_boxes[:, 5])


def test_ncnn_export_is_rejected_before_conversion(tmp_path):
    from libreyolo import LibreDeformableDETR

    model = LibreDeformableDETR(None, size="r50ss", nb_classes=3, device="cpu")
    with pytest.raises(NotImplementedError, match="NCNN export is not supported"):
        model.export(
            format="ncnn",
            imgsz=64,
            output_path=str(tmp_path / "deformable_detr_ncnn"),
        )
