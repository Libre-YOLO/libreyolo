"""LibreResNet <-> timm exact inference parity (acceptance gate).

Gated as external_data/network because it pulls timm's pretrained weights.
Asserts the native model loads timm's state_dict strictly and produces
bit-identical logits (max_abs_diff == 0) in eval mode, for every shipped size.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.external_data, pytest.mark.network]

TAGS = {
    "18": "resnet18.a1_in1k",
    "34": "resnet34.a1_in1k",
    "50": "resnet50.a1_in1k",
    "101": "resnet101.a1_in1k",
}


@pytest.mark.parametrize("size", ["18", "34", "50", "101"])
def test_timm_parity(size):
    timm = pytest.importorskip("timm")
    from libreyolo.models.resnet.nn import ResNet

    tm = timm.create_model(TAGS[size], pretrained=True).eval()
    ours = ResNet(size=size, num_classes=1000)
    result = ours.load_state_dict(tm.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    ours.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        diff = (tm(x) - ours(x)).abs().max().item()
    assert diff == 0.0, f"{size}: max_abs_diff={diff}"
