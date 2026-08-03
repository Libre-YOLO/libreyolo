"""DeiT ONNX and TorchScript export parity contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("export_format", ("onnx", "torchscript", "openvino"))
def test_deit_export_raw_and_predict_parity(tmp_path, export_format):
    if export_format == "onnx":
        onnx = pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    elif export_format == "openvino":
        pytest.importorskip("openvino")

    from libreyolo import LibreDeiT, LibreYOLO

    torch.manual_seed(17)
    model = LibreDeiT(size="t", nb_classes=5, device="cpu")
    model.model.eval()
    tensor = torch.rand(1, 3, 224, 224)
    with torch.inference_mode():
        expected = model.model(tensor).cpu().numpy()

    suffix = {
        "onnx": ".onnx",
        "torchscript": ".torchscript",
        "openvino": "_openvino",
    }[export_format]
    artifact = model.export(
        format=export_format,
        imgsz=224,
        dynamic=False,
        simplify=False,
        device="cpu",
        output_path=str(tmp_path / f"LibreDeiTt-cls{suffix}"),
    )
    backend = LibreYOLO(artifact, device="cpu")
    actual = np.asarray(backend._run_inference(tensor.numpy())[0])

    assert backend.model_family == "deit"
    assert backend.model_size == "t"
    assert backend.task == "classify"
    assert backend.nb_classes == 5
    assert backend.imgsz == 224
    if export_format == "openvino":
        raw_cosine = torch.nn.functional.cosine_similarity(
            torch.from_numpy(actual), torch.from_numpy(expected)
        )
        assert float(raw_cosine) > 0.999
        assert int(actual.argmax()) == int(expected.argmax())
    else:
        tolerance = 2e-4 if export_format == "onnx" else 0.0
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=tolerance)

    if export_format == "onnx":
        graph = onnx.load(artifact)
        assert [value.name for value in graph.graph.input] == ["images"]
        assert [value.name for value in graph.graph.output] == ["output"]
        assert (
            next(item.version for item in graph.opset_import if item.domain == "") == 17
        )
        metadata = {item.key: item.value for item in graph.metadata_props}
        assert metadata["model_family"] == "deit"
        assert metadata["task"] == "classify"
        assert metadata["size"] == "t"
        assert metadata["imgsz"] == "224"

    image = np.random.default_rng(23).integers(
        0, 256, size=(240, 280, 3), dtype=np.uint8
    )
    native_probs = model.predict(image).probs.data
    exported_probs = backend.predict(image).probs.data
    cosine = torch.nn.functional.cosine_similarity(
        native_probs[None], exported_probs[None]
    )
    threshold = 0.999 if export_format == "openvino" else 0.999999
    assert float(cosine) > threshold
    assert int(native_probs.argmax()) == int(exported_probs.argmax())


def test_deit_export_uses_opset17():
    from libreyolo.export.onnx import _requires_onnx_opset17
    from libreyolo.export.support import get_support

    assert _requires_onnx_opset17("deit") is True
    assert get_support("deit", "classify", "onnx").tier == "validated"
    assert get_support("deit", "classify", "torchscript").tier == "validated"
    assert get_support("deit", "classify", "openvino").tier == "validated"
    assert get_support("deit", "classify", "tensorrt").tier == "validated"
