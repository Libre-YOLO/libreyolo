"""Nightly load contract for the current flagship release checkpoints."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from libreyolo import LibreYOLO

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.flagship_nightly,
]

RELEASE_CHECKPOINTS = [
    ("LibreYOLO9t.pt", "yolo9", "detect"),
    ("LibreRFDETRn.pt", "rfdetr", "detect"),
    ("LibreRFDETRn-seg.pt", "rfdetr", "segment"),
    ("LibreRFDETRn-obb.pt", "rfdetr", "obb"),
]


def _resolve_release_checkpoint(filename: str) -> Path:
    staged_dir = os.environ.get("LIBREYOLO_RELEASE_CHECKPOINT_DIR")
    if staged_dir:
        path = Path(staged_dir) / filename
        if not path.is_file():
            pytest.fail(f"Required release checkpoint is not staged: {path}")
        return path

    # The nightly runner persists ``weights/`` in its shared cache. A missing
    # canonical filename follows the normal public auto-download route; a
    # cached file is reused without network or filesystem replacement.
    return Path("weights") / filename


@pytest.mark.parametrize(
    ("filename", "family", "task"),
    RELEASE_CHECKPOINTS,
)
def test_current_flagship_release_checkpoint_loads(
    filename,
    family,
    task,
):
    """Published flagship artifacts must load without manual conversion."""
    checkpoint = _resolve_release_checkpoint(filename)

    model = LibreYOLO(str(checkpoint), device="cpu")

    assert model.FAMILY == family
    assert model.task == task
