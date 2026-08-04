"""Real MNN conversion, fresh-load runtime, and detection parity checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import (
    match_detections,
    requires_mnn,
    requires_rfdetr,
    run_export_compare_test,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.supported_backend,
    pytest.mark.mnn,
]

_FLAGSHIPS = [
    pytest.param("yolo9", "t", marks=pytest.mark.yolo9),
    pytest.param(
        "rfdetr",
        "n",
        marks=(pytest.mark.rfdetr, requires_rfdetr),
    ),
]


@requires_mnn
@pytest.mark.external_data
@pytest.mark.parametrize(("family", "size"), _FLAGSHIPS)
def test_mnn_flagship_export_runtime_and_detection_parity(
    family, size, sample_image, tmp_path
):
    exported_path, native_results, runtime_results = run_export_compare_test(
        family,
        size,
        sample_image,
        tmp_path,
        format="mnn",
        export_kwargs={"dynamic": False, "simplify": True},
        match_threshold=0.8,
        device="cpu",
    )

    artifact = Path(exported_path)
    sidecar = Path(f"{artifact}.json")
    assert artifact.is_file() and artifact.stat().st_size > 0
    assert sidecar.is_file()
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["model_family"] == family
    assert metadata["size"] == size
    assert metadata["task"] == "detect"
    assert metadata["format"] == "mnn"
    assert metadata["dynamic"] is False
    assert metadata["precision"] == "fp32"
    assert metadata["mnn_backend"] == "cpu"
    assert metadata["mnn_input_names"]
    assert metadata["mnn_output_names"]
    assert metadata["mnn_input_shape"][0] == metadata["mnn_batch"] == 1
    assert len(native_results) > 0
    assert len(runtime_results) == len(native_results)
    match_rate, matched, total = match_detections(native_results, runtime_results)
    assert match_rate >= 0.8, f"matched={matched}/{total}, rate={match_rate:.2%}"
