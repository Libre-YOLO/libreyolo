"""End-to-end Paddle export and CPU inference parity for YOLO9."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from .conftest import load_model, run_export_compare_test


def _has_validated_paddle_stack() -> bool:
    try:
        return (
            importlib.metadata.version("paddlepaddle") == "2.6.2"
            and importlib.metadata.version("x2paddle") == "1.6.0"
            and tuple(
                int(part) for part in importlib.metadata.version("onnx").split(".")[:2]
            )
            <= (1, 17)
        )
    except importlib.metadata.PackageNotFoundError:
        return False


requires_paddle = pytest.mark.skipif(
    not _has_validated_paddle_stack(),
    reason="validated Paddle export stack is not installed",
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.experimental_backend,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.paddle,
    pytest.mark.yolo9,
    pytest.mark.slow,
]


@requires_paddle
def test_yolo9_paddle_export_and_cpu_parity(sample_image, tmp_path):
    exported_path, pt_results, paddle_results = run_export_compare_test(
        "yolo9",
        "t",
        sample_image,
        tmp_path,
        format="paddle",
        export_kwargs={
            "imgsz": 640,
            "batch": 1,
            "dynamic": False,
            "half": False,
            "simplify": True,
        },
        match_threshold=0.95,
        device="cpu",
    )

    artifact = Path(exported_path)
    assert artifact.is_dir()
    assert (artifact / "model.pdmodel").is_file()
    assert (artifact / "model.pdiparams").is_file()
    assert not list(artifact.glob("*.py")), "converter source leaked into artifact"

    metadata = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert metadata["model_family"] == "yolo9"
    assert metadata["task"] == "detect"
    assert metadata["precision"] == "fp32"
    assert metadata["dynamic"] is False
    assert metadata["imgsz"] == 640
    assert metadata["imgsz_h"] == 640
    assert metadata["imgsz_w"] == 640

    assert len(pt_results) > 0
    assert len(paddle_results) > 0

    from libreyolo import LibreYOLO
    from libreyolo.export.exporter import PaddleExporter

    native = load_model("yolo9", "t", device="cpu")
    paddle = LibreYOLO(exported_path, device="cpu")
    exporter = PaddleExporter(native)
    with (
        exporter._model_context(torch.device("cpu"), False, False, 1, (640, 640)) as (
            export_model,
            probe,
        ),
        torch.inference_mode(),
    ):
        expected = export_model(probe).detach().cpu().numpy()
        actual = paddle._run_inference(probe.cpu().numpy())[0]

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=3e-3)
