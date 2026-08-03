"""LibreViT ONNX export and runtime parity coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.vit, pytest.mark.onnx]


def test_random_tiny_onnx_round_trip_matches_eager(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    from libreyolo import LibreViT

    eager = LibreViT(size="ti", nb_classes=11, device="cpu")
    eager.model.eval()
    image = np.random.default_rng(2020).standard_normal(
        (1, 3, 224, 224), dtype=np.float32
    )
    with torch.inference_mode():
        expected = eager.model(torch.from_numpy(image)).numpy()

    output_path = tmp_path / "LibreViTti-cls.onnx"
    exported = eager.export(
        format="onnx", imgsz=224, half=False, output_path=str(output_path)
    )
    session = ort.InferenceSession(exported, providers=["CPUExecutionProvider"])
    assert [output.name for output in session.get_outputs()] == ["output"]
    actual = session.run(None, {session.get_inputs()[0].name: image})[0]

    assert actual.shape == expected.shape == (1, 11)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.external_data
@pytest.mark.network
def test_pretrained_tiny_onnx_backend_matches_native_predict(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from libreyolo import LibreYOLO
    from libreyolo.backends.onnx import OnnxBackend

    local = Path("weights/LibreViTti-cls.pt")
    weights = str(local) if local.exists() else "LibreViTti-cls.pt"
    native = LibreYOLO(weights, device="cpu")
    pixels = np.arange(277 * 301 * 3, dtype=np.uint32).reshape(277, 301, 3)
    image = Image.fromarray(pixels.astype(np.uint8))

    input_tensor, _, _, _ = native._preprocess(image)
    with torch.inference_mode():
        expected_logits = native.model(input_tensor).numpy()

    output_path = tmp_path / "LibreViTti-cls.onnx"
    exported = native.export(
        format="onnx", imgsz=224, half=False, output_path=str(output_path)
    )
    backend = OnnxBackend(exported, device="cpu")
    actual_logits = backend.session.run(
        None, {backend.input_name: input_tensor.numpy()}
    )[0]
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-4, atol=1e-4)

    native_result = native.predict(image)
    exported_result = backend.predict(image)
    torch.testing.assert_close(
        exported_result.probs.data,
        native_result.probs.data,
        rtol=1e-4,
        atol=1e-5,
    )
    assert exported_result.probs.top1 == native_result.probs.top1
