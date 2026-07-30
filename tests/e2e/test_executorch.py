"""Real ExecuTorch XNNPACK export/runtime checks for detection flagships."""

from __future__ import annotations

import importlib.resources
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.e2e, pytest.mark.executorch]


def _require_executorch(monkeypatch):
    pytest.importorskip("executorch")
    pytest.importorskip("executorch.runtime")
    if shutil.which("flatc") is not None:
        return

    # ExecuTorch Windows wheels bundle flatc but 1.2 does not add its
    # directory to PATH. Keep this host-tool workaround local to the real
    # toolchain test; production emits an actionable error.
    package_root = importlib.resources.files("executorch")
    bundled = Path(str(package_root.joinpath("data", "bin", "flatc.exe")))
    if not bundled.exists():
        pytest.skip("ExecuTorch lowering requires flatc on PATH")
    monkeypatch.setenv("PATH", f"{bundled.parent}{os.pathsep}{os.environ['PATH']}")


@pytest.mark.parametrize(
    ("family", "imgsz"),
    [("yolo9", 64), ("rfdetr", 384)],
)
def test_detection_flagship_raw_parity_and_predict(
    tmp_path, monkeypatch, family, imgsz
):
    _require_executorch(monkeypatch)

    from libreyolo import LibreRFDETR, LibreYOLO, LibreYOLO9
    from libreyolo.export.exporter import ExecuTorchExporter
    from libreyolo.utils.results import Results

    torch.manual_seed(0)
    if family == "yolo9":
        model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    else:
        pytest.importorskip("transformers")
        model = LibreRFDETR(
            {}, size="n", nb_classes=2, device="cpu", task="detect"
        )
    model.model.eval()
    original_training = model.model.training
    original_export = getattr(
        getattr(model.model, "head", None), "export", None
    )

    rng = np.random.default_rng(0)
    first = torch.from_numpy(
        rng.standard_normal((1, 3, imgsz, imgsz), dtype=np.float32)
    )
    second = (
        torch.full_like(first, 100.0)
        if family == "yolo9"
        else torch.from_numpy(
            np.random.default_rng(1).standard_normal(
                (1, 3, imgsz, imgsz), dtype=np.float32
            )
        )
    )

    exporter = ExecuTorchExporter(model)
    with exporter._model_context(
        torch.device("cpu"), False, False, 1, (imgsz, imgsz)
    ) as (prepared, _), torch.no_grad():
        expected = prepared(first)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)

    artifact = model.export(
        "executorch",
        output_path=str(tmp_path / f"{family}.pte"),
        imgsz=imgsz,
        batch=1,
        dynamic=False,
    )
    assert model.model.training is original_training
    if original_export is not None:
        assert model.model.head.export is original_export

    backend = LibreYOLO(artifact)
    actual = backend._run_inference(first.numpy())
    changed = backend._run_inference(second.numpy())
    assert len(actual) == len(expected)

    parity_error = 0.0
    for actual_output, expected_output in zip(actual, expected):
        expected_array = expected_output.detach().cpu().numpy()
        np.testing.assert_allclose(
            actual_output, expected_array, rtol=1e-3, atol=2e-4
        )
        parity_error = max(
            parity_error,
            float(np.max(np.abs(actual_output - expected_array))),
        )

    sensitivity = max(
        float(np.max(np.abs(first_output - second_output)))
        for first_output, second_output in zip(actual, changed)
    )
    assert sensitivity > max(parity_error * 100, 1e-4)

    image = np.random.default_rng(2).integers(
        0, 256, (imgsz, imgsz, 3), dtype=np.uint8
    )
    result = backend.predict(image)
    assert isinstance(result, Results)
    assert result.boxes is not None


def test_failed_export_restores_yolo9_state(tmp_path, monkeypatch):
    _require_executorch(monkeypatch)

    from libreyolo import LibreYOLO9

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    model.model.train()
    original_training = model.model.training
    original_export = model.model.head.export

    def fail_export(*args, **kwargs):
        raise RuntimeError("simulated lowering failure")

    monkeypatch.setattr(
        "libreyolo.export.executorch.export_executorch", fail_export
    )
    output = tmp_path / "failed.pte"
    with pytest.raises(RuntimeError, match="simulated"):
        model.export(
            "executorch",
            output_path=str(output),
            imgsz=64,
            batch=1,
            dynamic=False,
        )

    assert model.model.training is original_training
    assert model.model.head.export is original_export
    assert not output.exists()
    assert not Path(f"{output}.json").exists()
