"""Unit tests for edge data decoding and BSDS-style validation."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from libreyolo.data.edge_dataset import EdgeDataset, load_edge_map, resolve_edge_data
from libreyolo.validation import EdgeValidator, ValidationConfig
from libreyolo.validation.edge_validator import (
    edge_f_measure,
    match_edge_pixels,
    thin_edge_probabilities,
)

pytestmark = pytest.mark.unit

IMGSZ = 16


def _vertical_edge(column: int = 8) -> np.ndarray:
    edge = np.zeros((IMGSZ, IMGSZ), dtype=np.uint8)
    edge[:, column] = 255
    return edge


def _write_dataset(root: Path, edge: np.ndarray | None = None) -> Path:
    image_path = root / "images" / "val" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (IMGSZ, IMGSZ), color=(20, 40, 60)).save(image_path)
    edge_path = root / "edges" / "val" / "sample.png"
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(edge_path), edge if edge is not None else _vertical_edge())
    yaml_path = root / "data.yaml"
    yaml_path.write_text(f"path: {root.as_posix()}\nval: images/val\n")
    return yaml_path


class _StubEdgeModel:
    size = "t"
    names = {0: "edge"}
    edge_resize_mode = "stretch"
    edge_imgsz_divisor = 4

    def __init__(self, output: np.ndarray | None = None):
        self.model = nn.Identity()
        self.output = torch.from_numpy(
            (output if output is not None else _vertical_edge()).astype(np.float32)
            / 255.0
        )

    def _get_model_name(self) -> str:
        return "stub-edge"

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        return self.output.to(images.device)[None, None].expand(batch, 1, -1, -1)


def test_load_edge_map_normalizes_integer_and_supports_inversion(tmp_path):
    path = tmp_path / "edge.png"
    assert cv2.imwrite(str(path), np.array([[0, 128, 255]], dtype=np.uint8))

    decoded = load_edge_map(path)
    inverted = load_edge_map(path, invert=True)

    assert decoded.dtype == np.float32
    assert decoded.tolist()[0] == pytest.approx([0.0, 128 / 255.0, 1.0])
    assert inverted.tolist()[0] == pytest.approx([1.0, 127 / 255.0, 0.0])


def test_dataset_pairs_same_stem_and_returns_validity_mask(tmp_path):
    yaml_path = _write_dataset(tmp_path)
    dataset = EdgeDataset(
        resolve_edge_data(yaml_path),
        split="val",
        imgsz=IMGSZ,
        resize_mode="stretch",
    )

    image, target, info, image_id = dataset[0]

    assert image.shape == (3, IMGSZ, IMGSZ)
    assert target["edges"].shape == (1, IMGSZ, IMGSZ)
    assert target["valid"].all()
    assert info["orig_shape"] == (IMGSZ, IMGSZ)
    assert image_id == 0


def test_thinning_retains_a_one_pixel_ridge():
    edge = torch.from_numpy(_vertical_edge().astype(np.float32) / 255.0)

    thinned = thin_edge_probabilities(edge)

    torch.testing.assert_close(thinned, edge)


def test_correspondence_is_one_to_one_and_respects_normalized_tolerance():
    target = np.zeros((100, 100), dtype=bool)
    target[50, 50] = True
    prediction = np.zeros_like(target)
    prediction[50, 50:52] = True

    matches, predicted, targets = match_edge_pixels(prediction, target, max_dist=0.0075)
    assert (matches, predicted, targets) == (1, 2, 1)
    assert edge_f_measure(matches, predicted, targets) == pytest.approx(2 / 3)

    shifted = np.zeros_like(target)
    shifted[50, 51] = True
    assert match_edge_pixels(shifted, target, max_dist=0.0075)[0] == 1
    assert match_edge_pixels(shifted, target, max_dist=0.001)[0] == 0


def test_perfect_prediction_scores_perfect_ods_and_ois(tmp_path):
    yaml_path = _write_dataset(tmp_path)
    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,
        batch_size=1,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
        edge_thresholds=(0.25, 0.5, 0.75),
    )

    metrics = EdgeValidator(_StubEdgeModel(), config).run()

    assert metrics["metrics/ODS"] == pytest.approx(1.0)
    assert metrics["metrics/OIS"] == pytest.approx(1.0)
    assert metrics["fitness"] == pytest.approx(1.0)


def test_ods_aggregates_one_threshold_while_ois_optimizes_each_image(tmp_path):
    config = ValidationConfig(
        data=str(tmp_path / "unused.yaml"),
        imgsz=IMGSZ,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
        edge_thresholds=(0.25, 0.75),
    )
    validator = EdgeValidator(_StubEdgeModel(), config)
    validator._init_metrics()

    targets = torch.zeros((2, 1, IMGSZ, IMGSZ), dtype=torch.float32)
    targets[:, :, 4, 4] = 1.0
    predictions = torch.zeros_like(targets)
    predictions[0, 0, 4, 4] = 0.9
    predictions[0, 0, 12, 12] = 0.6
    predictions[1, 0, 4, 4] = 0.4
    validator._update_metrics(
        predictions,
        {
            "edges": targets,
            "valid": torch.ones_like(targets, dtype=torch.bool),
        },
        img_info=None,
    )

    metrics = validator._compute_metrics()

    assert metrics["metrics/ODS"] == pytest.approx(0.8)
    assert metrics["metrics/OIS"] == pytest.approx(1.0)
    assert metrics["metrics/best_threshold"] == pytest.approx(0.25)


def test_imgsz_divisor_mismatch_raises(tmp_path):
    yaml_path = _write_dataset(tmp_path)
    config = ValidationConfig(
        data=str(yaml_path),
        imgsz=IMGSZ,
        device="cpu",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "runs"),
    )
    model = _StubEdgeModel()
    model.edge_imgsz_divisor = 10

    with pytest.raises(ValueError, match="divisible by 10"):
        EdgeValidator(model, config).run()
