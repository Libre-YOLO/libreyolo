"""Unit tests for surface-normal decoding and validation."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from libreyolo.data.normal_dataset import NormalDataset, load_normal_png
from libreyolo.validation import NormalValidator, ValidationConfig
from libreyolo.validation.normal_validator import angular_errors

pytestmark = pytest.mark.unit

IMGSZ = 16
FRONT = np.array([0.0, 0.0, -1.0], dtype=np.float32)
RIGHT = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _write_image(path: Path, width: int = IMGSZ, height: int = IMGSZ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(30, 60, 90)).save(path)


def _encode_normals(normals: np.ndarray) -> np.ndarray:
    return np.rint((normals + 1.0) * 0.5 * 65535.0).astype(np.uint16)


def _write_normals(path: Path, normals: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_rgb = _encode_normals(normals)
    assert cv2.imwrite(str(path), encoded_rgb[..., ::-1])


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), mask.astype(np.uint8))


def _decoded_vector(vector: np.ndarray) -> torch.Tensor:
    encoded = _encode_normals(vector)
    decoded = encoded.astype(np.float64) / 65535.0 * 2.0 - 1.0
    decoded /= np.linalg.norm(decoded)
    return torch.tensor(decoded, dtype=torch.float32)


def _make_dataset_yaml(
    root: Path,
    *,
    vector: np.ndarray = FRONT,
    mask: np.ndarray | None = None,
) -> Path:
    _write_image(root / "images" / "val" / "wall.jpg")
    normals = np.broadcast_to(vector, (IMGSZ, IMGSZ, 3)).copy()
    _write_normals(root / "normals" / "val" / "wall.png", normals)
    if mask is not None:
        _write_mask(root / "masks" / "val" / "wall.png", mask)
    yaml_path = root / "data.yaml"
    yaml_path.write_text(f"path: {root.as_posix()}\nval: images/val\n")
    return yaml_path


class _StubNormalModel:
    size = "n"
    names = {0: "normal"}
    normal_resize_mode = "stretch"

    def __init__(self, vector: torch.Tensor | None = None):
        self.model = nn.Identity()
        self.vector = vector if vector is not None else _decoded_vector(FRONT)

    def _get_model_name(self) -> str:
        return "stub"

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        return (
            self.vector.to(images.device)
            .view(1, 3, 1, 1)
            .expand(batch, 3, height, width)
        )


def _run_validator(tmp_path: Path, model: _StubNormalModel | None = None):
    yaml_path = _make_dataset_yaml(tmp_path)
    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,
        batch_size=1,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )
    return NormalValidator(model or _StubNormalModel(), config).run()


def test_angular_errors_cover_known_angles_and_invalid_prediction():
    targets = torch.from_numpy(
        np.asarray(
            [[FRONT, FRONT, FRONT, [0.0, 0.0, 0.0]]],
            dtype=np.float64,
        )
    )
    predictions = torch.from_numpy(
        np.asarray(
            [[FRONT, RIGHT, -FRONT, [1.0, 0.0, 0.0]]],
            dtype=np.float64,
        )
    )
    predictions[0, 2] = 0.0

    errors = angular_errors(predictions, targets)

    assert errors.tolist() == pytest.approx([0.0, 90.0, 180.0])


def test_uint16_rgb_decode_preserves_channel_convention(tmp_path):
    path = tmp_path / "normal.png"
    normals = np.array([[RIGHT, FRONT]], dtype=np.float32)
    _write_normals(path, normals)

    decoded = load_normal_png(path)

    assert decoded.dtype == np.float32
    assert decoded.shape == (1, 2, 3)
    assert decoded[0, 0].tolist() == pytest.approx(RIGHT, abs=3e-5)
    assert decoded[0, 1].tolist() == pytest.approx(FRONT, abs=3e-5)
    assert np.linalg.norm(decoded, axis=-1).ravel().tolist() == pytest.approx(
        [1.0, 1.0]
    )


def test_loader_rejects_non_uint16_normal_png(tmp_path):
    path = tmp_path / "normal.png"
    assert cv2.imwrite(str(path), np.zeros((2, 2, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="expected uint16"):
        load_normal_png(path)


def test_dataset_horizontal_flip_negates_x_component(tmp_path, monkeypatch):
    yaml_path = _make_dataset_yaml(tmp_path, vector=RIGHT)
    from libreyolo.data.normal_dataset import resolve_normal_data

    dataset = NormalDataset(
        resolve_normal_data(yaml_path),
        split="val",
        imgsz=IMGSZ,
        augment=True,
        resize_mode="stretch",
    )
    monkeypatch.setattr("libreyolo.data.normal_dataset.random.random", lambda: 0.0)

    _, target, _, _ = dataset[0]

    assert float(target[0].mean()) == pytest.approx(-1.0, abs=3e-5)
    assert torch.linalg.vector_norm(target, dim=0).mean().item() == pytest.approx(1.0)


def test_perfect_predictions_score_perfect_threshold_metrics(tmp_path):
    metrics = _run_validator(tmp_path)

    assert metrics["metrics/mean_angular_error"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["metrics/median_angular_error"] == pytest.approx(0.0, abs=1e-3)
    assert metrics["metrics/within_11_25"] == pytest.approx(100.0)
    assert metrics["metrics/within_22_5"] == pytest.approx(100.0)
    assert metrics["metrics/within_30"] == pytest.approx(100.0)
    assert metrics["fitness"] == pytest.approx(1.0)


def test_opposite_predictions_score_180_degrees(tmp_path):
    metrics = _run_validator(tmp_path, _StubNormalModel(-_decoded_vector(FRONT)))

    assert metrics["metrics/mean_angular_error"] == pytest.approx(180.0, abs=1e-4)
    assert metrics["metrics/median_angular_error"] == pytest.approx(180.0)
    assert metrics["metrics/within_30"] == 0.0


def test_optional_mask_excludes_invalid_ground_truth_pixels(tmp_path):
    mask = np.zeros((IMGSZ, IMGSZ), dtype=np.uint8)
    mask[:, : IMGSZ // 2] = 255
    yaml_path = _make_dataset_yaml(tmp_path, mask=mask)

    class _HalfGarbageModel(_StubNormalModel):
        def _forward(self, images: torch.Tensor) -> torch.Tensor:
            prediction = super()._forward(images).clone()
            prediction[..., IMGSZ // 2 :] = 0.0
            return prediction

    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )
    metrics = NormalValidator(_HalfGarbageModel(), config).run()

    assert metrics["metrics/mean_angular_error"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["metrics/within_11_25"] == pytest.approx(100.0)


def test_all_invalid_mask_raises(tmp_path):
    yaml_path = _make_dataset_yaml(
        tmp_path,
        mask=np.zeros((IMGSZ, IMGSZ), dtype=np.uint8),
    )
    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )

    with pytest.raises(ValueError, match="no valid ground-truth pixels"):
        NormalValidator(_StubNormalModel(), config).run()


def test_imgsz_divisor_mismatch_raises(tmp_path):
    yaml_path = _make_dataset_yaml(tmp_path)
    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )
    model = _StubNormalModel()
    model.normal_imgsz_divisor = 14

    with pytest.raises(ValueError, match="divisible by 14"):
        NormalValidator(model, config).run()
