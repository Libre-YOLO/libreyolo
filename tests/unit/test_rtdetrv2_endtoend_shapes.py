"""Eval-time dynamic-size tests for RT-DETRv2 (and the shared RT-DETR encoder).

The RT-DETRv2 wrapper builds its model with a concrete ``eval_spatial_size``,
so eval-mode forwards must rebuild the precomputed encoder pos-embed and
decoder anchors whenever the runtime input size differs from the native one.
RT-DETR v1 passes no ``eval_spatial_size`` and must stay dynamic-by-default.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.rtdetr.nn import RTDETRModel
from libreyolo.models.rtdetrv2.nn import RTDETRv2Model

pytestmark = pytest.mark.unit


def _make_v2(eval_spatial_size):
    return RTDETRv2Model(
        num_classes=2, eval_spatial_size=eval_spatial_size
    ).eval()


def test_eval_forward_accepts_runtime_size_different_from_eval_spatial_size():
    """Eval must rebuild fixed-size encoder/decoder constants for runtime imgsz."""
    model = _make_v2((640, 640))

    for hw in ((320, 320), (800, 800)):
        x = torch.randn(1, 3, *hw)
        with torch.no_grad():
            out = model(x)
        assert out["pred_logits"].shape == (1, 300, 2)
        assert out["pred_boxes"].shape == (1, 300, 4)


def test_eval_rectangular_size_with_native_token_count_matches_dynamic():
    """A rect grid can share the native buffer's token count (16x25 == 20x20).

    Constants must be rebuilt from the grid shape, not the token count, or
    the encoder/decoder silently apply wrong-grid embeddings and anchors.
    """
    fixed = _make_v2((640, 640))
    dynamic = _make_v2(None)
    dynamic.load_state_dict(fixed.state_dict(), strict=False)

    x = torch.randn(1, 3, 512, 800)
    with torch.no_grad():
        out_fixed = fixed(x)
        out_dynamic = dynamic(x)
    for key in ("pred_logits", "pred_boxes"):
        torch.testing.assert_close(
            out_fixed[key], out_dynamic[key], rtol=1e-4, atol=1e-5
        )


def test_rtdetr_v1_default_stays_dynamic_with_no_extra_buffers():
    """v1 never sets eval_spatial_size; behavior and state-dict must not change."""
    model = RTDETRModel(num_classes=2).eval()

    assert model.decoder.eval_spatial_size is None
    assert not any(
        key.endswith(("anchors", "valid_mask")) for key in model.state_dict()
    )

    for hw in ((320, 320), (512, 800)):
        x = torch.randn(1, 3, *hw)
        with torch.no_grad():
            out = model(x)
        assert out["pred_logits"].shape[0] == 1
        assert out["pred_boxes"].shape[-1] == 4
