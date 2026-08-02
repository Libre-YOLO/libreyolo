"""LibreSwin CPU export and public-runtime parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.swin]


@pytest.mark.parametrize("format", ["onnx", "torchscript"])
def test_swin_cpu_export_predict_parity(tmp_path, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    from libreyolo import LibreSwin, LibreYOLO

    torch.manual_seed(6)
    model = LibreSwin(size="t", nb_classes=5, device="cpu")
    image = np.random.default_rng(6).integers(
        0, 256, size=(249, 271, 3), dtype=np.uint8
    )
    expected = model.predict(image).probs.data.cpu()
    suffix = ".onnx" if format == "onnx" else ".torchscript"
    artifact = model.export(
        format=format,
        output_path=str(tmp_path / f"swin-t{suffix}"),
        imgsz=224,
        dynamic=False,
        half=False,
        simplify=False,
    )

    runtime = LibreYOLO(artifact, device="cpu")
    actual = runtime.predict(image).probs.data.cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert runtime.model_family == "swin"
    assert runtime.task == "classify"
    assert runtime.imgsz == 224
    assert int(actual.argmax()) == int(expected.argmax())
