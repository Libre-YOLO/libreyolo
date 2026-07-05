"""Export round-trip parity for the Darknet families + YOLOv7.

The decode is baked into the export graph (a single ``(B, 4+nc, N)`` tensor), so
these check that an exported ONNX / TorchScript graph reproduces the native
export-wrapper output bit-for-bit (within runtime fp tolerance). PR-gate safe:
random weights, tiny variants, ONNX guarded by importorskip.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from libreyolo import LibreYOLO2, LibreYOLO3, LibreYOLO4, LibreYOLO7
from libreyolo.models.darknet.export import DarknetExportWrapper
from libreyolo.models.yolo7.export import YOLO7ExportWrapper

pytestmark = pytest.mark.unit


def _wrapper(model):
    if model.FAMILY == "yolo7":
        return YOLO7ExportWrapper(model.model).eval()
    return DarknetExportWrapper(model.model).eval()


CASES = [
    (LibreYOLO2, "t", 416),
    (LibreYOLO3, "t", 416),
    (LibreYOLO4, "t", 416),
    (LibreYOLO7, "b", 640),
]


@pytest.mark.parametrize("cls,size,imgsz", CASES)
def test_export_wrapper_output_contract(cls, size, imgsz):
    model = cls(size=size, device="cpu")
    model.model.eval()
    x = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        out = _wrapper(model)(x)
    # (B, 4+nc, N): boxes + per-class scores, N anchors flattened over heads
    assert out.shape[0] == 1
    assert out.shape[1] == 4 + model.nb_classes
    assert out.shape[2] > 0
    # scores are probabilities in [0, 1]
    scores = out[:, 4:, :]
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0


@pytest.mark.parametrize("cls,size,imgsz", CASES)
def test_onnx_graph_matches_native_wrapper(cls, size, imgsz, tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    model = cls(size=size, device="cpu")
    model.model.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        native = _wrapper(model)(x).numpy()

    onnx_path = model.export(format="onnx")
    try:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        onnx_out = sess.run(None, {iname: x.numpy()})[0]
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

    assert onnx_out.shape == native.shape
    assert np.abs(native - onnx_out).max() < 1e-3


@pytest.mark.parametrize("cls,size,imgsz", [CASES[1], CASES[3]])
def test_torchscript_graph_matches_native_wrapper(cls, size, imgsz):
    model = cls(size=size, device="cpu")
    model.model.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        native = _wrapper(model)(x)

    ts_path = model.export(format="torchscript")
    try:
        ts = torch.jit.load(ts_path, map_location="cpu").eval()
        with torch.no_grad():
            ts_out = ts(x)
    finally:
        if os.path.exists(ts_path):
            os.remove(ts_path)

    assert torch.allclose(native, ts_out, atol=1e-4)
