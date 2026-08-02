"""Official-checkpoint ONNX and TorchScript parity for both CenterNet variants."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from .conftest import require_test_weights

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.supported_backend,
    pytest.mark.centernet,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.slow,
]

CASES = (
    ("resdcn18", "LibreCenterNetresdcn18.pt"),
    ("dla34", "LibreCenterNetdla34.pt"),
)
FORMATS = (
    pytest.param("onnx", marks=pytest.mark.onnx),
    pytest.param("torchscript", marks=pytest.mark.torchscript),
)


def _weights_path(filename: str) -> str:
    staged = os.environ.get("CENTERNET_CONVERTED_CKPT_DIR")
    if staged and (Path(staged) / filename).exists():
        return str(Path(staged) / filename)
    return require_test_weights(filename, expected_family="centernet")


@pytest.mark.parametrize(("size", "filename"), CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("export_format", FORMATS)
def test_official_checkpoint_exported_predict_parity(
    tmp_path, sample_image, size, filename, export_format
):
    if export_format == "onnx":
        onnx = pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    from libreyolo import LibreCenterNet, LibreYOLO
    from libreyolo.export.exporter import BaseExporter

    model = LibreCenterNet(_weights_path(filename), size=size, device="cpu")
    input_tensor, _, _, _ = model._preprocess(sample_image, input_size=512)
    exporter = BaseExporter.create(export_format, model)
    with (
        exporter._model_context("cpu", False, False, 1, (512, 512)) as (
            wrapped,
            _,
        ),
        torch.no_grad(),
    ):
        expected_raw = wrapped(input_tensor).detach().cpu().numpy()

    suffix = ".onnx" if export_format == "onnx" else ".torchscript"
    artifact = model.export(
        format=export_format,
        imgsz=512,
        dynamic=False,
        simplify=False,
        device="cpu",
        output_path=str(tmp_path / f"LibreCenterNet{size}{suffix}"),
    )
    if export_format == "onnx":
        graph = onnx.load(artifact)
        onnx.checker.check_model(graph)
        assert [value.name for value in graph.graph.input] == ["images"]
        assert [value.name for value in graph.graph.output] == ["output"]

    backend = LibreYOLO(artifact, device="cpu")
    assert backend.model_family == "centernet"
    assert backend.model_size == size
    assert backend.imgsz == 512
    actual_raw = backend._run_inference(input_tensor.numpy())
    assert len(actual_raw) == 1
    np.testing.assert_allclose(
        actual_raw[0], expected_raw, rtol=2e-5, atol=1e-3
    )

    native = model.predict(sample_image, conf=0.25, max_det=100).boxes.data.numpy()
    exported = backend.predict(sample_image, conf=0.25, max_det=100).boxes.data.numpy()
    assert len(native) > 0
    assert exported.shape == native.shape
    np.testing.assert_array_equal(exported[:, 5], native[:, 5])
    np.testing.assert_allclose(exported[:, :4], native[:, :4], rtol=1e-5, atol=1e-2)
    np.testing.assert_allclose(exported[:, 4], native[:, 4], rtol=1e-5, atol=1e-4)
