"""NCNN raw-output parity for YOLO-NAS detection and pose."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("task", "size"), [("detect", "s"), ("pose", "n")])
@pytest.mark.parametrize("format", ["onnx", "torchscript", "ncnn"])
def test_yolonas_raw_parity(tmp_path, task, size, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    if format == "ncnn" and (
        importlib.util.find_spec("pnnx") is None
        or importlib.util.find_spec("ncnn") is None
    ):
        pytest.skip("PNNX and NCNN are required")

    from libreyolo import LibreYOLO, LibreYOLONAS

    torch.manual_seed(0)
    model = LibreYOLONAS(
        None,
        size=size,
        nb_classes=3,
        task=task,
        device="cpu",
    )
    # This test isolates raw graph/backend parity on a tiny synthetic tensor.
    # Production YOLO-NAS export keeps the public >=640 preprocessing contract.
    model.INPUT_SIZE_MIN = 32
    model.model.eval()
    tensor = torch.rand(1, 3, 96, 96)
    traced = torch.jit.trace(model.model, tensor)
    with torch.no_grad():
        native = traced(tensor)
    if isinstance(native, torch.Tensor):
        native = (native,)

    artifact = model.export(
        format=format,
        imgsz=96,
        dynamic=False,
        simplify=False,
        output_path=str(tmp_path / f"yolonas-{task}.{format}"),
    )
    actual = LibreYOLO(artifact, device="cpu")._run_inference(tensor.numpy())

    assert len(actual) == len(native)
    for actual_output, native_output in zip(actual, native):
        np.testing.assert_allclose(
            actual_output,
            native_output.detach().cpu().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )
