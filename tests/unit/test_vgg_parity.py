"""Exact VGG parity against the official torchvision checkpoints.

The external reference is torchvision at commit
``10f68dbd78b9aa5cab9328f3b2e99cfb0b608122`` (BSD-3-Clause). The test uses
its published ImageNet-1K weights and compares eval-mode logits for every
shipped LibreVGG size.
"""

from __future__ import annotations

import gc

import pytest
import torch

pytestmark = [pytest.mark.external_data, pytest.mark.network, pytest.mark.vgg]


@pytest.mark.parametrize("size", ["16", "19", "16bn", "19bn"])
def test_torchvision_vgg_exact_parity(size):
    from torchvision.models import (
        VGG16_BN_Weights,
        VGG16_Weights,
        VGG19_BN_Weights,
        VGG19_Weights,
        vgg16,
        vgg16_bn,
        vgg19,
        vgg19_bn,
    )

    from libreyolo.models.vgg.nn import VGG

    references = {
        "16": (vgg16, VGG16_Weights.IMAGENET1K_V1),
        "19": (vgg19, VGG19_Weights.IMAGENET1K_V1),
        "16bn": (vgg16_bn, VGG16_BN_Weights.IMAGENET1K_V1),
        "19bn": (vgg19_bn, VGG19_BN_Weights.IMAGENET1K_V1),
    }
    builder, weights = references[size]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reference = builder(weights=weights).eval().to(device)
    native = VGG(size=size, num_classes=1000, init_weights=False).eval().to(device)
    incompatible = native.load_state_dict(reference.state_dict(), strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys

    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn((1, 3, 224, 224), generator=generator, device=device)
    with torch.inference_mode():
        expected = reference(x)
        actual = native(x)
    max_abs_diff = (expected - actual).abs().max().item()
    assert max_abs_diff == 0.0, f"{size}: max_abs_diff={max_abs_diff}"

    del actual, expected, x, native, reference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
