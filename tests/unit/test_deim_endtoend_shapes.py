"""Eval-time dynamic-size tests for DEIM.

The DEIM wrapper builds its model with a concrete ``eval_spatial_size``, so
eval-mode forwards must rebuild the precomputed encoder pos-embed and decoder
anchors whenever the runtime input size differs from the native one.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.deim.nn import LibreDEIMModel

pytestmark = pytest.mark.unit


def _make(eval_spatial_size):
    return LibreDEIMModel(
        config="s", nb_classes=2, eval_spatial_size=eval_spatial_size
    ).eval()


def test_eval_forward_accepts_runtime_size_different_from_eval_spatial_size():
    """Eval must rebuild fixed-size encoder/decoder constants for runtime imgsz."""
    model = _make((640, 640))

    for hw in ((320, 320), (800, 800)):
        x = torch.randn(1, 3, *hw)
        with torch.no_grad():
            out = model(x)
        assert out["pred_boxes"].shape[-1] == 4


def test_eval_rectangular_size_with_native_token_count_matches_dynamic():
    """A rect grid can share the native buffer's token count (16x25 == 20x20).

    Constants must be rebuilt from the grid shape, not the token count, or
    the encoder/decoder silently apply wrong-grid embeddings and anchors.
    """
    fixed = _make((640, 640))
    dynamic = _make(None)
    dynamic.load_state_dict(fixed.state_dict(), strict=False)

    x = torch.randn(1, 3, 512, 800)
    with torch.no_grad():
        out_fixed = fixed(x)
        out_dynamic = dynamic(x)
    for key in ("pred_logits", "pred_boxes"):
        torch.testing.assert_close(
            out_fixed[key], out_dynamic[key], rtol=1e-4, atol=1e-5
        )
