"""Deterministic raw-output parity for YOLO9 OBB exports."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("format", ("onnx", "torchscript"))
def test_yolo9_obb_raw_parity(tmp_path, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    import libreyolo
    from libreyolo.export.exporter import OnnxExporter
    from libreyolo.models.yolo9.model import LibreYOLO9

    torch.manual_seed(0)
    imgsz = 320
    model = LibreYOLO9(None, size="t", nb_classes=3, device="cpu", task="obb")
    model.model.eval()
    tensor = torch.rand(1, 3, imgsz, imgsz)

    exporter = OnnxExporter(model)
    with exporter._model_context("cpu", False, False, 1, (imgsz, imgsz)) as (
        wrapped,
        _,
    ):
        with torch.no_grad():
            native = wrapped(tensor)
    if isinstance(native, torch.Tensor):
        native = (native,)

    # The OBB prediction tensor carries boxes, one angle row, and class scores.
    assert native[0].shape == (1, 4 + 1 + 3, native[0].shape[-1])

    artifact = model.export(
        format=format,
        imgsz=imgsz,
        dynamic=False,
        simplify=False,
        output_path=str(tmp_path / f"LibreYOLO9t-obb.{format}"),
    )
    actual = libreyolo.LibreYOLO(artifact, device="cpu")._run_inference(tensor.numpy())

    assert len(actual) == len(native)
    rtol, atol = (2e-3, 2e-2) if format == "onnx" else (1e-3, 1e-3)
    for actual_output, native_output in zip(actual, native):
        expected = native_output.detach().cpu().numpy()
        if format == "onnx":
            element_match = np.isclose(actual_output, expected, rtol=rtol, atol=atol)
            assert float(element_match.mean()) > 0.95
            continue
        np.testing.assert_allclose(
            actual_output,
            expected,
            rtol=rtol,
            atol=atol,
        )
