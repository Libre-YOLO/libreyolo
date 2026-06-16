"""End-to-end smoke tests for LibreFOMO."""

from pathlib import Path

import pytest
import torch

from libreyolo import LibreYOLO


pytestmark = [pytest.mark.e2e, pytest.mark.fomo]


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_load_cloud_checkpoint(size: str) -> None:
    """Download and load the cloud weights, check metadata, and run a forward pass."""
    model_name = f"LibreFOMO{size}.pt"

    model = LibreYOLO(model_name, device="cpu")

    assert model.size == size
    assert model.task == "point"
    assert model.nb_classes == 1

    imgsz = model.model.imgsz
    dummy_input = torch.zeros(1, 3, imgsz, imgsz)

    with torch.no_grad():
        out = model._forward(dummy_input)

    expected_hw = imgsz // 8
    assert out.shape == (1, 2, expected_hw, expected_hw), f"Unexpected output shape: {out.shape}"
