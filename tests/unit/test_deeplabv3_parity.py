"""Exact acceptance parity against torchvision's official DeepLabv3 weights."""

from __future__ import annotations

import os

import pytest
import torch
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet101_Weights,
    DeepLabV3_ResNet50_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet101,
    deeplabv3_resnet50,
)

pytestmark = [
    pytest.mark.deeplabv3,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("LIBREYOLO_DEEPLABV3_ACCEPTANCE") != "1",
        reason=(
            "set LIBREYOLO_DEEPLABV3_ACCEPTANCE=1 to download the three "
            "official checkpoints and run exact 520x520 parity"
        ),
    ),
]

CASES = {
    "r50": (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights.DEFAULT),
    "r101": (deeplabv3_resnet101, DeepLabV3_ResNet101_Weights.DEFAULT),
    "mv3": (
        deeplabv3_mobilenet_v3_large,
        DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
    ),
}


@pytest.mark.parametrize("size", tuple(CASES))
def test_torchvision_official_logits_are_bit_exact(size):
    from libreyolo.models.deeplabv3.convert import (
        convert_upstream_deeplabv3_state_dict,
    )
    from libreyolo.models.deeplabv3.nn import LibreDeepLabv3Net

    builder, weights = CASES[size]
    reference = builder(weights=weights).eval()
    runtime_state = convert_upstream_deeplabv3_state_dict(reference.state_dict())
    assert runtime_state is not None
    port = LibreDeepLabv3Net(size=size, num_classes=21).eval()
    port.load_state_dict(runtime_state, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    reference.to(device)
    port.to(device)
    input_tensor = torch.linspace(
        -2.0,
        2.0,
        steps=3 * 520 * 520,
        dtype=torch.float32,
        device=device,
    ).reshape(1, 3, 520, 520)

    with torch.inference_mode():
        expected = reference(input_tensor)["out"]
        actual = port(input_tensor)

    difference = (expected - actual).abs()
    assert float(difference.max().item()) == 0.0
    assert int(torch.count_nonzero(difference).item()) == 0
