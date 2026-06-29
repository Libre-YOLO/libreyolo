"""LibreMobileNetV4 ONNX export: single logits output, numerics match eager."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.onnx]


def test_export_onnx_round_trip_matches_eager():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    from libreyolo import LibreMobileNetV4

    eager = LibreMobileNetV4(size="s", nb_classes=1000, device="cpu")
    eager.model.eval()
    rng = np.random.default_rng(0)
    img = rng.standard_normal((1, 3, 224, 224), dtype=np.float32)
    with torch.no_grad():
        eager_out = eager.model(torch.from_numpy(img)).numpy()
    assert eager_out.shape == (1, 1000)

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "mnv4s.onnx")
        path = eager.export(format="onnx", imgsz=224, half=False, output_path=out)
        assert os.path.exists(path)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        assert len(sess.get_outputs()) == 1  # single logits tensor
        ort_out = sess.run(None, {sess.get_inputs()[0].name: img})[0]

    assert ort_out.shape == (1, 1000)
    np.testing.assert_allclose(ort_out, eager_out, rtol=1e-4, atol=1e-4)
