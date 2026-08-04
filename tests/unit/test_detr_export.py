"""Export and exported-backend parity tests for vanilla DETR."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


def _backend_stub():
    from libreyolo.backends.base import BaseBackend

    class StubBackend(BaseBackend):
        def _run_inference(self, blob):
            del blob
            return []

    return object.__new__(StubBackend)


def test_detr_backend_parser_matches_native_postprocess():
    from libreyolo.backends.base import _is_nms_free_family
    from libreyolo.postprocess.detr import postprocess
    from libreyolo.utils.coco import COCO91_TO_COCO80

    rng = np.random.default_rng(7)
    logits = rng.normal(size=(1, 12, 92)).astype(np.float32)
    boxes = rng.uniform(0.1, 0.9, size=(1, 12, 4)).astype(np.float32)
    native = postprocess(
        {
            "pred_logits": torch.from_numpy(logits),
            "pred_boxes": torch.from_numpy(boxes),
        },
        conf_thres=0.0,
        original_size=(320, 180),
        max_det=9,
        class_map=COCO91_TO_COCO80,
    )

    backend = _backend_stub()
    backend.nb_classes = 80
    parsed_boxes, parsed_scores, parsed_classes = backend._parse_detr(
        [logits, boxes], 320, 180, 0.0, max_det=9
    )

    assert _is_nms_free_family("detr") is True
    np.testing.assert_allclose(parsed_boxes, native["boxes"], rtol=0, atol=2e-5)
    np.testing.assert_allclose(parsed_scores, native["scores"], rtol=0, atol=2e-7)
    np.testing.assert_array_equal(parsed_classes, native["classes"])


def test_detr_backend_uses_family_validation_preprocessor():
    from libreyolo.validation.preprocessors import DETRValPreprocessor

    backend = _backend_stub()
    backend.model_family = "detr"
    backend.model_size = "r50"
    backend.imgsz = 64
    assert isinstance(backend._get_val_preprocessor(), DETRValPreprocessor)


@pytest.mark.parametrize("export_format", ("onnx", "torchscript"))
def test_detr_export_raw_and_predict_parity(tmp_path, export_format):
    if export_format == "onnx":
        onnx = pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    from libreyolo import LibreDETR, LibreYOLO
    from libreyolo.models.detr.nn import DETRExportWrapper

    torch.manual_seed(3)
    model = LibreDETR(None, size="r50", nb_classes=3, device="cpu")
    model.model.eval()
    tensor = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        expected = DETRExportWrapper(model.model)(tensor)

    suffix = ".onnx" if export_format == "onnx" else ".torchscript"
    artifact = model.export(
        format=export_format,
        imgsz=64,
        dynamic=False,
        simplify=False,
        device="cpu",
        output_path=str(tmp_path / f"LibreDETRr50{suffix}"),
    )
    backend = LibreYOLO(artifact, device="cpu")
    actual = backend._run_inference(tensor.numpy())

    assert backend.model_family == "detr"
    assert backend.model_size == "r50"
    assert backend.nb_classes == 3
    assert backend.imgsz == 64
    assert len(actual) == 2
    for converted, native in zip(actual, expected):
        tolerance = 2e-5 if export_format == "onnx" else 0.0
        np.testing.assert_allclose(
            converted,
            native.detach().cpu().numpy(),
            rtol=1e-5,
            atol=tolerance,
        )

    if export_format == "onnx":
        graph = onnx.load(artifact)
        assert [value.name for value in graph.graph.input] == ["images"]
        assert [value.name for value in graph.graph.output] == [
            "pred_logits",
            "pred_boxes",
        ]
        assert (
            next(item.version for item in graph.opset_import if item.domain == "") == 17
        )
        metadata = {item.key: item.value for item in graph.metadata_props}
        assert metadata["model_family"] == "detr"
        assert metadata["size"] == "r50"
        assert metadata["imgsz"] == "64"

    image = np.random.default_rng(11).integers(0, 256, size=(48, 80, 3), dtype=np.uint8)
    native_result = model.predict(image, imgsz=64, conf=0.0, max_det=10)
    exported_result = backend.predict(image, conf=0.0, max_det=10)
    native_boxes = native_result.boxes.data.cpu().numpy()
    exported_boxes = exported_result.boxes.data.cpu().numpy()
    assert exported_boxes.shape == native_boxes.shape
    np.testing.assert_array_equal(exported_boxes[:, 5], native_boxes[:, 5])
    np.testing.assert_allclose(
        exported_boxes[:, :5],
        native_boxes[:, :5],
        rtol=2e-4,
        atol=2e-3,
    )


def test_detr_export_support_contract():
    from libreyolo import LibreDETR
    from libreyolo.export.exporter import OnnxExporter
    from libreyolo.export.support import get_support

    assert get_support("detr", "detect", "onnx").tier == "validated"
    assert get_support("detr", "detect", "torchscript").tier == "validated"
    ncnn = get_support("detr", "detect", "ncnn")
    assert ncnn.tier == "blocked"
    assert "NCNN" in ncnn.reason

    model = LibreDETR(None, size="r50", nb_classes=3, device="cpu")
    with pytest.raises(NotImplementedError, match="fixed square"):
        OnnxExporter(model)._resolve_params(
            output_path=None,
            imgsz=(64, 96),
            device="cpu",
            half=False,
            int8=False,
        )
