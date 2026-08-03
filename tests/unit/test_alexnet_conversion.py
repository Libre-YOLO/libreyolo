"""LibreAlexNet metadata wrapping and runtime auto-conversion."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.alexnet]


def _alexnet_signature(nc: int = 7) -> dict[str, torch.Tensor]:
    return {
        "features.0.weight": torch.empty((64, 3, 11, 11), device="meta"),
        "classifier.1.weight": torch.empty((4096, 256 * 6 * 6), device="meta"),
        "classifier.4.weight": torch.empty((4096, 4096), device="meta"),
        "classifier.6.weight": torch.empty((nc, 4096), device="meta"),
    }


def test_runtime_autoconversion_wraps_classification_metadata(tmp_path: Path):
    from libreyolo.models.autoconvert import autoconvert_upstream_checkpoint
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    source = tmp_path / "alexnet.pth"
    torch.save(_alexnet_signature(), source)

    output = autoconvert_upstream_checkpoint(str(source))
    assert output is not None
    output_path = Path(output)
    assert output_path.name == "alexnet-LibreAlexNetb-cls.pt"

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(checkpoint, strict=False) == []
    assert checkpoint["model_family"] == "alexnet"
    assert checkpoint["size"] == "b"
    assert checkpoint["task"] == "classify"
    assert checkpoint["nc"] == 7
    assert checkpoint["imgsz"] == 224
