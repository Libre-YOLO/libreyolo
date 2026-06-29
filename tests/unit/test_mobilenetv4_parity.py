"""LibreMobileNetV4 <-> timm exact inference parity (acceptance gate).

Gated as external_data/network because it pulls timm's pretrained weights.
Asserts the native model loads timm's state_dict strictly and produces
bit-identical logits (max_abs_diff == 0) in eval mode.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.external_data, pytest.mark.network]

TAGS = {
    "s": "mobilenetv4_conv_small.e2400_r224_in1k",
    "m": "mobilenetv4_conv_medium.e500_r224_in1k",
    "l": "mobilenetv4_conv_large.e500_r256_in1k",
}
RES = {"s": 224, "m": 224, "l": 256}


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_timm_parity(size):
    timm = pytest.importorskip("timm")
    from libreyolo.models.mobilenetv4.nn import MobileNetV4

    tm = timm.create_model(TAGS[size], pretrained=True).eval()
    ours = MobileNetV4(size=size, num_classes=1000)
    result = ours.load_state_dict(tm.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    ours.eval()

    x = torch.randn(2, 3, RES[size], RES[size])
    with torch.no_grad():
        diff = (tm(x) - ours(x)).abs().max().item()
    assert diff == 0.0, f"{size}: max_abs_diff={diff}"
