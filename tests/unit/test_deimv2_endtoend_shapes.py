"""Eval-time dynamic-size tests for DEIMv2.

DEIMv2 configs bake a per-size native ``eval_spatial_size`` (320 for atto),
so eval-mode forwards must rebuild the precomputed encoder pos-embed and
decoder anchors whenever the runtime input size differs from the native one.
The validator letterboxes to the requested imgsz and calls the model
directly, so this path must not rely on any preprocess-level guard.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.deimv2.nn import LibreDEIMv2Model

pytestmark = pytest.mark.unit


def test_eval_forward_accepts_runtime_size_different_from_eval_spatial_size():
    """Eval must rebuild fixed-size encoder/decoder constants for runtime imgsz."""
    model = LibreDEIMv2Model(config="atto", nb_classes=2).eval()

    for hw in ((320, 320), (640, 640)):
        x = torch.randn(1, 3, *hw)
        with torch.no_grad():
            out = model(x)
        assert out["pred_boxes"].shape[-1] == 4


def test_eval_rectangular_size_with_native_token_count_matches_dynamic():
    """A rect grid can share the native buffer's token count (5x20 == 10x10).

    Constants must be rebuilt from the grid shape, not the token count, or
    the encoder/decoder silently apply wrong-grid embeddings and anchors.
    Native atto size is 320 (stride-32 grid 10x10 = 100 tokens); 160x640
    gives a 5x20 grid whose 100 tokens collide with the native count
    exactly, while 384x512 (12x16 = 192) exercises the plain non-native
    rebuild. Sizes must stay divisible by 32.
    """
    fixed = LibreDEIMv2Model(config="atto", nb_classes=2).eval()
    dynamic = LibreDEIMv2Model(config="atto", nb_classes=2).eval()
    dynamic.load_state_dict(fixed.state_dict(), strict=False)
    dynamic.encoder.eval_spatial_size = None
    dynamic.decoder.eval_spatial_size = None

    for hw in ((160, 640), (384, 512)):
        x = torch.randn(1, 3, *hw)
        with torch.no_grad():
            out_fixed = fixed(x)
            out_dynamic = dynamic(x)
        for key in ("pred_logits", "pred_boxes"):
            torch.testing.assert_close(
                out_fixed[key], out_dynamic[key], rtol=1e-4, atol=1e-5
            )
