"""Exact native-port parity against torchvision's official FCN models.

This network-gated acceptance test downloads the two official torchvision
v0.26.0 checkpoints. The implementation is BSD-3-Clause; the checkpoint
redistribution basis is documented separately in the FCN provenance notice.
Both primary and auxiliary logits are compared before LibreYOLO postprocessing
or export so no downstream operation can hide numerical drift.
"""

from __future__ import annotations

import os

import pytest
import torch
from torchvision.models.segmentation import (
    FCN_ResNet50_Weights,
    FCN_ResNet101_Weights,
    fcn_resnet50,
    fcn_resnet101,
)

from libreyolo.models.fcn.nn import LibreFCNModel

pytestmark = [
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("LIBREYOLO_FCN_ACCEPTANCE") != "1",
        reason=(
            "set LIBREYOLO_FCN_ACCEPTANCE=1 to download both official "
            "checkpoints and run exact parity"
        ),
    ),
]


CASES = {
    "r50": (fcn_resnet50, FCN_ResNet50_Weights.DEFAULT),
    "r101": (fcn_resnet101, FCN_ResNet101_Weights.DEFAULT),
}


def _max_abs_diff(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]):
    maximum = 0.0
    assert tuple(expected) == tuple(actual) == ("out", "aux")
    for key in expected:
        assert expected[key].shape == actual[key].shape
        maximum = max(
            maximum,
            float((expected[key] - actual[key]).abs().max().item()),
        )
    return maximum


@pytest.mark.parametrize("size", ["r50", "r101"])
def test_torchvision_official_weights_exact_parity(size):
    builder, weights = CASES[size]
    upstream = builder(weights=weights, aux_loss=True).eval()
    ours = LibreFCNModel(size=size, num_classes=21, normalize_input=False).eval()
    load_result = ours.load_state_dict(upstream.state_dict(), strict=True)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upstream.to(device)
    ours.to(device)
    generator = torch.Generator(device=device).manual_seed(637)
    normalized_image = torch.randn(
        1,
        3,
        520,
        520,
        generator=generator,
        device=device,
    )

    with torch.inference_mode():
        expected = upstream(normalized_image)
        actual = ours(normalized_image)

    maximum = _max_abs_diff(expected, actual)
    assert maximum == 0.0, f"{size}: max_abs_diff={maximum}"


@pytest.mark.parametrize("size", ["r50", "r101"])
def test_public_graph_internal_normalization_exact_parity(size):
    builder, weights = CASES[size]
    upstream = builder(weights=weights, aux_loss=True).eval()
    ours = LibreFCNModel(size=size, num_classes=21, normalize_input=True).eval()
    ours.load_state_dict(upstream.state_dict(), strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upstream.to(device)
    ours.to(device)
    generator = torch.Generator(device=device).manual_seed(638)
    image = torch.rand(1, 3, 520, 520, generator=generator, device=device)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    with torch.inference_mode():
        expected = upstream((image - mean) / std)
        actual = ours(image)

    maximum = _max_abs_diff(expected, actual)
    assert maximum == 0.0, f"{size}: normalized max_abs_diff={maximum}"
