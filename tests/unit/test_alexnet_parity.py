"""Exact LibreAlexNet parity with the pinned torchvision implementation.

This acceptance gate downloads ``AlexNet_Weights.IMAGENET1K_V1`` from the
official model host. The native graph must strict-load the complete state dict
and return bit-identical evaluation logits for every shipped size.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.alexnet,
]


def test_torchvision_parity_all_shipped_sizes():
    from torchvision.models import AlexNet_Weights, alexnet

    from libreyolo import LibreAlexNet
    from libreyolo.models.alexnet.nn import AlexNet

    weights = AlexNet_Weights.IMAGENET1K_V1
    upstream = alexnet(weights=weights).eval()
    state_dict = upstream.state_dict()

    for size in sorted(LibreAlexNet.INPUT_SIZES):
        ours = AlexNet(num_classes=1000)
        result = ours.load_state_dict(state_dict, strict=True)
        assert not result.missing_keys
        assert not result.unexpected_keys
        ours.eval()

        generator = torch.Generator().manual_seed(637)
        image = torch.randn(
            1,
            3,
            LibreAlexNet.INPUT_SIZES[size],
            LibreAlexNet.INPUT_SIZES[size],
            generator=generator,
        )
        with torch.inference_mode():
            upstream_logits = upstream(image)
            our_logits = ours(image)

        max_abs_diff = (upstream_logits - our_logits).abs().max().item()
        assert max_abs_diff == 0.0, f"size={size}: max_abs_diff={max_abs_diff}"
