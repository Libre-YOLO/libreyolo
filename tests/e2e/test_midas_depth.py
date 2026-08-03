"""Task-appropriate MiDaS clean-download and depth-result smoke."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from libreyolo import LibreYOLO
from libreyolo.utils.serialization import validate_checkpoint_metadata

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.midas,
]


def test_midas_small_clean_download_predicts_depth(tmp_path: Path, sample_image):
    """Use the upstream route; MiDaS must never enter detect-only MODEL_CATALOG."""
    checkpoint_path = tmp_path / "LibreMiDaSs-depth.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LibreYOLO(str(checkpoint_path), device=device)
    result = model.predict(sample_image)

    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert checkpoint["model_family"] == "midas"
    assert checkpoint["size"] == "s"
    assert result.boxes is None
    assert result.depth_map is not None
    assert tuple(result.depth_map.data.shape) == result.orig_shape
    assert torch.isfinite(result.depth_map.data).all()
