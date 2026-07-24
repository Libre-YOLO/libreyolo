"""LibrePAGE gaze-target inference check.

Runs when a converted checkpoint is staged locally (or auto-downloadable)
and skips cleanly otherwise, mirroring the L2CS gaze e2e pattern. Uses BYO
head boxes so the check does not depend on any face-detector backend.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image

# Per-family marker (not ``general_nightly``): targeted family jobs /
# ``pytest -m page`` include this check without making weight absence a
# hard nightly gate.
pytestmark = [pytest.mark.e2e, pytest.mark.page]

_PAGE_WEIGHTS = "LibrePAGEs-gazetarget.pt"


def _staged_page_weights() -> str | None:
    for candidate in (Path(_PAGE_WEIGHTS), Path("weights") / _PAGE_WEIGHTS):
        if candidate.exists():
            return str(candidate)
    return None


def test_page_gazetarget_inference_is_stable():
    weights = _staged_page_weights()
    if weights is None:
        pytest.skip(
            f"LibrePAGE weights not staged locally; place {_PAGE_WEIGHTS} "
            "in ./ or ./weights to run this gaze-target check."
        )
    pytest.importorskip("transformers")

    from libreyolo import LibreYOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreYOLO(weights, device=device)
    assert model.family == "page"
    assert model.task == "gazetarget"

    image = Image.new("RGB", (320, 240), color=(96, 96, 96))
    kwargs = {"head_boxes": [(40, 30, 120, 130)]}
    first = model(image, **kwargs)
    second = model(image, **kwargs)

    assert first.gazetarget is not None, "page did not return gazetarget output"
    assert len(first.gazetarget) == 1
    assert len(first.boxes) == 1
    assert first.gazetarget.heatmaps is not None
    assert tuple(first.gazetarget.heatmaps.shape[-2:]) == (64, 64)
    inout = float(first.gazetarget.inout[0])
    assert 0.0 <= inout <= 1.0
    x, y = (float(v) for v in first.gazetarget.xy[0])
    assert 0.0 <= x <= 320.0 and 0.0 <= y <= 240.0

    # Determinism: identical input + boxes give identical outputs.
    assert torch.allclose(first.gazetarget.data, second.gazetarget.data)

    row = first.summary()[0]
    assert "gaze_target" in row
