"""Small-artifact VGG ONNX and TorchScript parity for the PR gate."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.unit, pytest.mark.vgg]


def _compact_vgg():
    from libreyolo import LibreVGG

    model = LibreVGG(size="16", nb_classes=10, device="cpu")
    # VGG's published 4096-wide classifier makes a random-weight artifact over
    # 500 MB. Retain the complete feature graph and upstream classifier module
    # indices while narrowing only the synthetic PR-gate head. The full trained
    # graph has a separate external-data acceptance test.
    model.model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 16),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(16, 16),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(16, 10),
    )
    model.model.eval()
    return model


@pytest.mark.onnx
def test_onnx_round_trip_matches_eager(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    model = _compact_vgg()
    rng = np.random.default_rng(0)
    sample = rng.standard_normal((1, 3, 224, 224), dtype=np.float32)
    with torch.inference_mode():
        expected = model.model(torch.from_numpy(sample)).numpy()

    output = tmp_path / "vgg16.onnx"
    exported = model.export(
        format="onnx",
        imgsz=224,
        dynamic=False,
        simplify=False,
        half=False,
        output_path=str(output),
    )
    session = ort.InferenceSession(exported, providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
    assert actual.shape == expected.shape == (1, 10)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.torchscript
def test_torchscript_round_trip_matches_eager(tmp_path):
    model = _compact_vgg()
    sample = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        expected = model.model(sample)

    output = tmp_path / "vgg16.torchscript"
    exported = model.export(
        format="torchscript",
        imgsz=224,
        half=False,
        output_path=str(output),
    )
    runtime = torch.jit.load(exported, map_location="cpu").eval()
    with torch.inference_mode():
        actual = runtime(sample)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_export_rejects_non_native_resolution(tmp_path):
    model = _compact_vgg()
    output = tmp_path / "vgg16.torchscript"
    with pytest.raises(ValueError, match="fixed native resolution 224x224"):
        model.export(
            format="torchscript",
            imgsz=256,
            output_path=str(output),
        )
    assert not output.exists()
