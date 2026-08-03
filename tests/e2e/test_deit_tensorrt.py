"""Official-checkpoint TensorRT FP16 parity for DeiT classification."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from .conftest import require_test_weights, requires_tensorrt


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.external_data,
    pytest.mark.export_backend,
    pytest.mark.supported_backend,
    pytest.mark.tensorrt,
    pytest.mark.trt,
    pytest.mark.deit,
    pytest.mark.slow,
]


@requires_tensorrt
def test_deit_tensorrt_fp16_raw_and_predict_parity(tmp_path):
    from libreyolo import LibreYOLO, SAMPLE_IMAGE

    weights = require_test_weights("LibreDeiTt-cls.pt")
    native = LibreYOLO(str(weights), device="cuda")
    native.model.eval()

    torch.manual_seed(41)
    tensor = torch.rand(1, 3, 224, 224, device="cuda")
    with torch.inference_mode():
        expected = native.model(tensor).float().cpu().numpy()

    artifact = native.export(
        format="tensorrt",
        imgsz=224,
        dynamic=False,
        half=True,
        simplify=False,
        workspace=0.25,
        output_path=str(tmp_path / "LibreDeiTt-cls-fp16.engine"),
    )
    backend = LibreYOLO(artifact, device="cuda")
    actual = np.asarray(backend._run_inference(tensor.cpu().numpy())[0])

    assert np.isfinite(actual).all()
    raw_cosine = torch.nn.functional.cosine_similarity(
        torch.from_numpy(actual), torch.from_numpy(expected)
    )
    assert float(raw_cosine) > 0.999

    native_result = native.predict(SAMPLE_IMAGE)
    exported_result = backend.predict(SAMPLE_IMAGE)
    native_probs = native_result.probs.data.cpu()
    exported_probs = exported_result.probs.data.cpu()
    probs_cosine = torch.nn.functional.cosine_similarity(
        native_probs[None], exported_probs[None]
    )
    assert float(probs_cosine) > 0.999
    assert exported_result.probs.top1 == native_result.probs.top1
    assert exported_result.probs.top5 == native_result.probs.top5
