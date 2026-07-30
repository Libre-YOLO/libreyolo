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


def _detections(result) -> np.ndarray:
    """Return postprocessed detection rows for list and scalar Results APIs."""
    if isinstance(result, list):
        result = result[0]
    return result.boxes.data.detach().cpu().numpy()


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0


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


@pytest.mark.experimental_backend
@pytest.mark.parametrize(
    ("family", "class_name", "size"),
    [
        ("teed", "LibreTEED", "t"),
        ("dexined", "LibreDexiNed", "b"),
    ],
)
def test_edge_map_runtime_parity(
    tmp_path, monkeypatch, family, class_name, size
):
    """Prove edge-map conversion and parity without restricted checkpoints."""
    _require_executorch(monkeypatch)

    import libreyolo
    from libreyolo import LibreYOLO

    torch.manual_seed(7)
    model_class = getattr(libreyolo, class_name)
    model = model_class(None, size=size, device="cpu")
    first = np.random.default_rng(7).integers(
        0, 256, (40, 64, 3), dtype=np.uint8
    )
    second = np.random.default_rng(8).integers(
        0, 256, (40, 64, 3), dtype=np.uint8
    )
    expected = model.predict(first, imgsz=64).edges.data.numpy()

    artifact = model.export(
        "executorch",
        output_path=str(tmp_path / f"{family}.pte"),
        imgsz=64,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)
    actual = runtime.predict(first).edges.data.numpy()
    changed = runtime.predict(second).edges.data.numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-4)
    assert float(np.max(np.abs(actual - changed))) > 1e-4


@pytest.mark.experimental_backend
@pytest.mark.parametrize("family", ["yolonas", "yolo9_p2"])
def test_additional_detection_raw_parity(tmp_path, monkeypatch, family):
    """Cover detector families lacking redistributable trained parity data."""
    _require_executorch(monkeypatch)

    from libreyolo import LibreYOLO, LibreYOLO9P2, LibreYOLONAS
    from libreyolo.export.exporter import ExecuTorchExporter

    torch.manual_seed(11)
    if family == "yolonas":
        model = LibreYOLONAS(None, size="s", nb_classes=2, device="cpu")
    else:
        model = LibreYOLO9P2(
            None, size="t", nb_classes=2, device="cpu"
        )

    first = torch.from_numpy(
        np.random.default_rng(11).standard_normal(
            (1, 3, 64, 64), dtype=np.float32
        )
    )
    second = (
        torch.full_like(first, 100.0)
        if family == "yolo9_p2"
        else torch.from_numpy(
            np.random.default_rng(12).standard_normal(
                (1, 3, 64, 64), dtype=np.float32
            )
        )
    )
    exporter = ExecuTorchExporter(model)
    with exporter._model_context(
        torch.device("cpu"), False, False, 1, (64, 64)
    ) as (prepared, _), torch.no_grad():
        expected = prepared(first)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)

    artifact = model.export(
        "executorch",
        output_path=str(tmp_path / f"{family}.pte"),
        imgsz=64,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)
    actual = runtime._run_inference(first.numpy())
    changed = runtime._run_inference(second.numpy())

    assert len(expected) == len(actual)
    parity_error = 0.0
    for expected_output, actual_output in zip(expected, actual):
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

    image = np.random.default_rng(13).integers(
        0, 256, (64, 64, 3), dtype=np.uint8
    )
    result = runtime.predict(image, conf=0.0, max_det=20)
    assert result.boxes is not None


@pytest.mark.external_data
@pytest.mark.flagship_nightly
@pytest.mark.parametrize(
    ("family", "weights_env", "imgsz"),
    [
        ("yolo9", "LIBREYOLO_EXECUTORCH_YOLO9_WEIGHTS", 640),
        ("rfdetr", "LIBREYOLO_EXECUTORCH_RFDETR_WEIGHTS", 384),
        ("yolox", "LIBREYOLO_EXECUTORCH_YOLOX_WEIGHTS", 416),
        ("picodet", "LIBREYOLO_EXECUTORCH_PICODET_WEIGHTS", 320),
        ("yolo9_e2e", "LIBREYOLO_EXECUTORCH_YOLO9_E2E_WEIGHTS", 640),
        ("ec", "LIBREYOLO_EXECUTORCH_EC_WEIGHTS", 640),
        ("rtdetr", "LIBREYOLO_EXECUTORCH_RTDETR_WEIGHTS", 640),
        ("rtdetrv2", "LIBREYOLO_EXECUTORCH_RTDETRV2_WEIGHTS", 640),
        ("rtdetrv4", "LIBREYOLO_EXECUTORCH_RTDETRV4_WEIGHTS", 640),
        ("yolo1", "LIBREYOLO_EXECUTORCH_YOLO1_WEIGHTS", 448),
        ("yolo2", "LIBREYOLO_EXECUTORCH_YOLO2_WEIGHTS", 416),
        ("yolo3", "LIBREYOLO_EXECUTORCH_YOLO3_WEIGHTS", 416),
        ("yolo4", "LIBREYOLO_EXECUTORCH_YOLO4_WEIGHTS", 416),
        ("yolo7", "LIBREYOLO_EXECUTORCH_YOLO7_WEIGHTS", 640),
    ],
)
def test_trained_detection_parity(
    tmp_path, monkeypatch, family, weights_env, imgsz
):
    """Match trained native and ExecuTorch post-NMS detections on real images."""
    _require_executorch(monkeypatch)

    from libreyolo import LibreYOLO

    weights_value = os.environ.get(weights_env)
    image_values = os.environ.get("LIBREYOLO_EXECUTORCH_IMAGES", "").splitlines()
    if not weights_value or len(image_values) < 2:
        pytest.skip(
            f"set {weights_env} and LIBREYOLO_EXECUTORCH_IMAGES "
            "to a newline-separated list of at least two images"
        )

    weights = Path(weights_value)
    images = [Path(value) for value in image_values if value.strip()]
    if not weights.is_file() or any(not image.is_file() for image in images):
        pytest.skip("staged trained-checkpoint parity inputs are unavailable")

    native = LibreYOLO(str(weights), device="cpu")
    artifact = native.export(
        "executorch",
        output_path=str(tmp_path / f"{family}.pte"),
        imgsz=imgsz,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)

    for image in images:
        expected = _detections(
            native.predict(str(image), conf=0.25, iou=0.6)
        )
        actual = _detections(
            runtime.predict(str(image), conf=0.25, iou=0.6)
        )
        assert len(expected) > 0
        assert len(actual) == len(expected)

        remaining = set(range(len(actual)))
        for expected_row in expected:
            same_class = [
                index
                for index in remaining
                if int(actual[index, 5]) == int(expected_row[5])
            ]
            assert same_class
            match = max(
                same_class,
                key=lambda index: _box_iou(expected_row, actual[index]),
            )
            remaining.remove(match)
            assert _box_iou(expected_row, actual[match]) >= 0.95
            assert abs(float(expected_row[4] - actual[match, 4])) <= 0.01


@pytest.mark.external_data
@pytest.mark.flagship_nightly
@pytest.mark.parametrize(
    ("family", "weights_env"),
    [
        ("mobilenetv4", "LIBREYOLO_EXECUTORCH_MOBILENETV4_WEIGHTS"),
        ("efficientnetv2", "LIBREYOLO_EXECUTORCH_EFFICIENTNETV2_WEIGHTS"),
        ("resnet", "LIBREYOLO_EXECUTORCH_RESNET_WEIGHTS"),
    ],
)
def test_trained_classification_parity(
    tmp_path, monkeypatch, family, weights_env
):
    """Match trained native and ExecuTorch logits and top-1 predictions."""
    _require_executorch(monkeypatch)

    from libreyolo import LibreYOLO

    weights_value = os.environ.get(weights_env)
    image_values = os.environ.get("LIBREYOLO_EXECUTORCH_IMAGES", "").splitlines()
    if not weights_value or len(image_values) < 2:
        pytest.skip(
            f"set {weights_env} and LIBREYOLO_EXECUTORCH_IMAGES "
            "to a newline-separated list of at least two images"
        )

    weights = Path(weights_value)
    images = [Path(value) for value in image_values if value.strip()]
    if not weights.is_file() or any(not image.is_file() for image in images):
        pytest.skip("staged trained-checkpoint parity inputs are unavailable")

    native = LibreYOLO(str(weights), device="cpu")
    artifact = native.export(
        "executorch",
        output_path=str(tmp_path / f"{family}.pte"),
        imgsz=224,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)

    for image in images:
        native_result = native.predict(str(image))
        runtime_result = runtime.predict(str(image))
        expected = native_result.probs.data.detach().cpu().numpy()
        actual = runtime_result.probs.data.detach().cpu().numpy()
        cosine = float(
            np.dot(expected, actual)
            / (np.linalg.norm(expected) * np.linalg.norm(actual))
        )
        assert cosine >= 0.999
        assert int(np.argmax(actual)) == int(np.argmax(expected))


@pytest.mark.external_data
@pytest.mark.flagship_nightly
def test_trained_pidnet_semantic_parity(tmp_path, monkeypatch):
    """Match trained PIDNet semantic maps after public postprocessing."""
    _require_executorch(monkeypatch)

    from libreyolo import LibreYOLO

    weights_value = os.environ.get("LIBREYOLO_EXECUTORCH_PIDNET_WEIGHTS")
    image_values = os.environ.get("LIBREYOLO_EXECUTORCH_IMAGES", "").splitlines()
    if not weights_value or len(image_values) < 2:
        pytest.skip(
            "set LIBREYOLO_EXECUTORCH_PIDNET_WEIGHTS and "
            "LIBREYOLO_EXECUTORCH_IMAGES to at least two images"
        )

    weights = Path(weights_value)
    images = [Path(value) for value in image_values if value.strip()]
    if not weights.is_file() or any(not image.is_file() for image in images):
        pytest.skip("staged trained-checkpoint parity inputs are unavailable")

    native = LibreYOLO(str(weights), device="cpu")
    artifact = native.export(
        "executorch",
        output_path=str(tmp_path / "pidnet.pte"),
        imgsz=1024,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)

    for image in images:
        expected = (
            native.predict(str(image)).semantic_mask.data.detach().cpu().numpy()
        )
        actual = (
            runtime.predict(str(image)).semantic_mask.data.detach().cpu().numpy()
        )
        assert expected.shape == actual.shape
        assert float(np.mean(expected == actual)) >= 0.95


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
