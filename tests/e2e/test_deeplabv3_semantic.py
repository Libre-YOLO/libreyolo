"""Public-checkpoint semantic prediction and validation for DeepLabv3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo import LibreYOLO

from .conftest import (
    DEEPLABV3_SEMANTIC_PARAMS,
    GENERAL_NIGHTLY_SEMANTIC_PARAMS,
    cuda_cleanup,
    require_test_weights,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.deeplabv3,
    pytest.mark.external_data,
    pytest.mark.network,
]


@pytest.mark.parametrize("family,size,weights", DEEPLABV3_SEMANTIC_PARAMS)
def test_public_checkpoint_predicts_semantic_mask(family, size, weights, sample_image):
    weights = require_test_weights(weights, expected_family=family)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreYOLO(weights, device=device)
    try:
        result = model.predict(sample_image, imgsz=520)
        with Image.open(sample_image) as image:
            expected_shape = (image.height, image.width)
        assert result.boxes is None
        assert result.semantic_mask is not None
        assert tuple(result.semantic_mask.data.shape) == expected_shape
        assert result.semantic_mask.data.unique().numel() >= 2
        assert result.names == model.names
        assert model.FAMILY == family
        assert model.size == size
    finally:
        del model
        cuda_cleanup()


@pytest.mark.parametrize(
    "family,size,weights",
    GENERAL_NIGHTLY_SEMANTIC_PARAMS,
)
def test_general_nightly_semantic_inference_is_stable(
    family, size, weights, sample_image
):
    weights = require_test_weights(weights, expected_family=family)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreYOLO(weights, device=device)
    try:
        first = model.predict(sample_image, imgsz=520).semantic_mask.data.cpu()
        second = model.predict(sample_image, imgsz=520).semantic_mask.data.cpu()
        assert first.shape == second.shape
        assert torch.equal(first, second)
    finally:
        del model
        cuda_cleanup()


def _make_semantic_dataset(root: Path) -> Path:
    names = {
        0: "__background__",
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "pottedplant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tvmonitor",
    }
    image_dir = root / "images" / "val"
    mask_dir = root / "masks" / "val"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for index in range(2):
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        image[:, :48] = (35 + index * 10, 55, 75)
        image[:, 48:] = (150, 120 + index * 10, 90)
        Image.fromarray(image).save(image_dir / f"sample{index}.png")
        mask = np.zeros((64, 96), dtype=np.uint8)
        mask[:, 48:] = 15
        Image.fromarray(mask).save(mask_dir / f"sample{index}.png")
    data = {
        "path": str(root),
        "train": "images/val",
        "val": "images/val",
        "masks_dir": "masks",
        "nc": 21,
        "names": names,
    }
    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return yaml_path


@pytest.mark.smoke
def test_public_mv3_tiny_semantic_validation(tmp_path):
    weights = require_test_weights(
        "LibreDeepLabv3mv3-sem.pt",
        expected_family="deeplabv3",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreYOLO(weights, device=device)
    try:
        metrics = model.val(
            data=str(_make_semantic_dataset(tmp_path)),
            batch=1,
            imgsz=520,
            workers=0,
            device=device,
            verbose=False,
        )
        assert 0.0 <= metrics["metrics/mIoU"] <= 1.0
        assert 0.0 <= metrics["metrics/pixel_accuracy"] <= 1.0
    finally:
        del model
        cuda_cleanup()
