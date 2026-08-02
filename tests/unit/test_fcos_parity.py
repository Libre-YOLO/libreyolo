"""Exact raw-head parity for the permissive torchvision FCOS checkpoint.

Reference implementation: pytorch/vision v0.26.0, commit
336d36e8db990a905498c73933e35231876e28bc, BSD-3-Clause.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.image_list import ImageList

from libreyolo.models.fcos.nn import LibreFCOSModel


pytestmark = [pytest.mark.unit, pytest.mark.external_data]

_FILENAME = "fcos_resnet50_fpn_coco-99b0c9b7.pth"


def _checkpoint_path() -> Path:
    configured = os.environ.get("LIBREYOLO_FCOS_CHECKPOINT")
    if configured:
        path = Path(configured)
    else:
        path = Path(torch.hub.get_dir()) / "checkpoints" / _FILENAME
    if not path.is_file():
        pytest.skip(
            "set LIBREYOLO_FCOS_CHECKPOINT to the official torchvision FCOS checkpoint"
        )
    return path


def _preprocessed_input() -> torch.Tensor:
    """Return a deterministic, normalized tensor with dimensions divisible by 32."""
    image = torch.linspace(0.0, 1.0, 3 * 192 * 224).reshape(1, 3, 192, 224)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image - mean) / std


def test_fcos_raw_head_matches_torchvision_exactly() -> None:
    """Bypass both transforms and require bit-exact backbone/head outputs."""
    state_dict = torch.load(_checkpoint_path(), map_location="cpu", weights_only=True)

    reference = fcos_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=91,
    ).eval()
    reference.load_state_dict(state_dict, strict=True)

    port = LibreFCOSModel(num_classes=91).eval()
    port.load_state_dict(state_dict, strict=True)

    image = _preprocessed_input()
    with torch.inference_mode():
        reference_features = list(reference.backbone(image).values())
        reference_output = reference.head(reference_features)
        port_output, _ = port.forward_head(image)

        reference_anchors = reference.anchor_generator(
            ImageList(image, [(image.shape[-2], image.shape[-1])]),
            reference_features,
        )[0]
        full_port_output = port(image)

    for name in ("cls_logits", "bbox_regression", "bbox_ctrness"):
        torch.testing.assert_close(
            port_output[name],
            reference_output[name],
            rtol=0.0,
            atol=0.0,
        )

    torch.testing.assert_close(
        full_port_output["anchors"][0],
        reference_anchors,
        rtol=0.0,
        atol=0.0,
    )
    assert full_port_output["level_sizes"].tolist() == [
        [feature.shape[-2] * feature.shape[-1] for feature in reference_features]
    ]
