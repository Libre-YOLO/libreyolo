"""PICODET export tests: ONNX + TorchScript shape and numerics."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.picodet]


def test_export_onnx_round_trip_matches_eager():
    """Exported ONNX model must produce numerically identical detections to
    the eager-mode model on the same input.

    This catches export-time bugs that don't surface in shape-only tests:
    silent dtype casts, opset-version regressions, and tracer specialization
    on dynamic axes.
    """
    onnx = pytest.importorskip("onnx")  # noqa: F841
    pytest.importorskip("onnxruntime")

    from libreyolo import LibrePICODET, LibreYOLO

    eager = LibrePICODET(size="s", nb_classes=80, device="cpu")

    # Run eager forward in export mode
    eager.model.head.export = True
    eager.model.eval()
    rng = np.random.default_rng(0)
    img_np = rng.standard_normal((1, 3, 320, 320), dtype=np.float32)
    with torch.no_grad():
        eager_out = eager.model(torch.from_numpy(img_np)).numpy()
    eager.model.head.export = False  # leave it as we found it

    # Round-trip through ONNX
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "picodet_s.onnx")
        path = eager.export(format="onnx", imgsz=320, half=False, output_path=out)
        assert os.path.exists(path)

        m = LibreYOLO(path)
        # Hit the underlying session directly to compare raw outputs:
        sess = m.session  # type: ignore[attr-defined]
        ort_out = sess.run(None, {sess.get_inputs()[0].name: img_np})[0]

    # Bit-equivalent check is too strict (ONNX may upgrade ops, fuse, etc.).
    # Use a tight tolerance instead.
    np.testing.assert_allclose(ort_out, eager_out, rtol=1e-4, atol=1e-4)


def test_export_torchscript_runs():
    from libreyolo import LibrePICODET

    torch.manual_seed(0)
    m = LibrePICODET(size="s", nb_classes=80, device="cpu")
    m.model.eval()
    x = torch.randn(1, 3, 320, 320)
    m.model.head.export = True
    with torch.no_grad():
        native = m.model(x)
    m.model.head.export = False
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "picodet_s.torchscript")
        path = m.export(format="torchscript", imgsz=320, half=False, output_path=out)
        assert os.path.exists(path)

        ts = torch.jit.load(path)
        with torch.no_grad():
            y = ts(x)
        # Single fused output: (1, N, 4 + nc) = (1, 2125, 84) at imgsz=320.
        assert y.shape == (1, 2125, 84), f"unexpected shape {y.shape}"
        torch.testing.assert_close(y, native, rtol=1e-4, atol=1e-4)


def test_export_ncnn_raw_parity(tmp_path):
    pytest.importorskip("pnnx")
    pytest.importorskip("ncnn")
    from libreyolo import LibrePICODET, LibreYOLO

    torch.manual_seed(0)
    model = LibrePICODET(size="s", nb_classes=3, device="cpu")
    model.model.eval()
    tensor = torch.rand(1, 3, 96, 96)
    model.model.head.export = True
    with torch.no_grad():
        native = model.model(tensor).numpy()
    model.model.head.export = False

    artifact = model.export(
        format="ncnn",
        imgsz=96,
        dynamic=False,
        output_path=str(tmp_path / "picodet.ncnn"),
    )
    backend = LibreYOLO(artifact, device="cpu")
    actual = backend._run_inference(tensor.numpy())[0]
    np.testing.assert_allclose(actual, native, rtol=1e-3, atol=1e-3)
