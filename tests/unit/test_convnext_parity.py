"""LibreConvNeXt <-> timm exact inference parity (acceptance gate).

Gated as external_data/network because it pulls timm's pretrained weights.
Asserts the native model loads timm's state_dict strictly and produces
bit-identical logits (max_abs_diff == 0) in eval mode.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.external_data, pytest.mark.network]

TAGS = {
    "t": "convnext_tiny.fb_in1k",
    "s": "convnext_small.fb_in1k",
    "b": "convnext_base.fb_in1k",
}
RES = {"t": 224, "s": 224, "b": 224}


@pytest.mark.parametrize("size", ["t", "s", "b"])
def test_timm_parity(size):
    timm = pytest.importorskip("timm")
    from libreyolo.models.convnext.nn import ConvNeXt

    tm = timm.create_model(TAGS[size], pretrained=True).eval()
    ours = ConvNeXt(size=size, num_classes=1000)
    result = ours.load_state_dict(tm.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    ours.eval()

    x = torch.randn(2, 3, RES[size], RES[size])
    with torch.no_grad():
        diff = (tm(x) - ours(x)).abs().max().item()
    assert diff == 0.0, f"{size}: max_abs_diff={diff}"
