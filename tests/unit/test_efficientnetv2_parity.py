"""LibreEfficientNetV2 <-> timm exact inference parity (acceptance gate).

Gated as external_data/network because it pulls timm's pretrained weights.
Asserts the native model loads timm's state_dict strictly and produces
bit-identical logits (max_abs_diff == 0) in eval mode, for every shipped size.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.external_data, pytest.mark.network]

TAGS = {
    "b0": "tf_efficientnetv2_b0.in1k",
    "b1": "tf_efficientnetv2_b1.in1k",
    "b2": "tf_efficientnetv2_b2.in1k",
    "b3": "tf_efficientnetv2_b3.in1k",
}
RES = {"b0": 224, "b1": 240, "b2": 260, "b3": 300}


@pytest.mark.parametrize("size", ["b0", "b1", "b2", "b3"])
def test_timm_parity(size):
    timm = pytest.importorskip("timm")
    from libreyolo.models.efficientnetv2.nn import EfficientNetV2

    tm = timm.create_model(TAGS[size], pretrained=True).eval()
    ours = EfficientNetV2(size=size, num_classes=1000)
    result = ours.load_state_dict(tm.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    ours.eval()

    x = torch.randn(2, 3, RES[size], RES[size])
    with torch.no_grad():
        diff = (tm(x) - ours(x)).abs().max().item()
    assert diff == 0.0, f"{size}: max_abs_diff={diff}"
