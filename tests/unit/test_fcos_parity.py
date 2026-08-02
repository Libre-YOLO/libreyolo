"""Exact raw-head parity for the permissive torchvision FCOS checkpoint.

Reference implementation: pytorch/vision v0.26.0, commit
336d36e8db990a905498c73933e35231876e28bc, BSD-3-Clause.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.image_list import ImageList

from libreyolo.models.fcos.model import LibreFCOS
from libreyolo.models.fcos.nn import LibreFCOSModel
from libreyolo.models.fcos.utils import preprocess_image
from libreyolo.utils.coco import COCO91_TO_COCO80


pytestmark = pytest.mark.external_data

_FILENAME = "fcos_resnet50_fpn_coco-99b0c9b7.pth"
_DOG_IMAGE = Path(__file__).parents[1] / "fixtures" / "dog.jpg"


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


def test_fcos_preprocess_matches_torchvision_exactly() -> None:
    reference = fcos_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=91,
    ).eval()
    pil_image = Image.open(_DOG_IMAGE).convert("RGB")
    image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0

    port_input, _, _, _ = preprocess_image(_DOG_IMAGE)
    with torch.inference_mode():
        reference_images, _ = reference.transform([image])

    assert reference_images.image_sizes == [(800, 1066)]
    torch.testing.assert_close(
        port_input,
        reference_images.tensors,
        rtol=0.0,
        atol=0.0,
    )


def test_fcos_real_image_detections_match_torchvision_exactly() -> None:
    state_dict = torch.load(_checkpoint_path(), map_location="cpu", weights_only=True)
    reference = fcos_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=91,
    ).eval()
    reference.load_state_dict(state_dict, strict=True)

    port = LibreFCOS(size="r50", device="cpu")
    port.model.load_state_dict(state_dict, strict=True)
    pil_image = Image.open(_DOG_IMAGE).convert("RGB")
    image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
    port_input, _, original_size, ratio = preprocess_image(_DOG_IMAGE)

    with torch.inference_mode():
        reference_result = reference([image])[0]
        port_output = port._forward(port_input)
        port_result = port._postprocess(
            port_output,
            conf_thres=0.2,
            iou_thres=0.6,
            original_size=original_size,
            max_det=100,
            ratio=ratio,
            input_size=800,
        )

    mapped_mask = torch.tensor(
        [int(label) in COCO91_TO_COCO80 for label in reference_result["labels"]]
    )
    reference_boxes = reference_result["boxes"][mapped_mask]
    reference_scores = reference_result["scores"][mapped_mask]
    reference_classes = torch.tensor(
        [
            COCO91_TO_COCO80[int(label)]
            for label in reference_result["labels"][mapped_mask]
        ],
        dtype=torch.int64,
    )

    assert port_result["num_detections"] == len(reference_boxes) == 64
    torch.testing.assert_close(
        torch.from_numpy(port_result["boxes"]),
        reference_boxes,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        torch.from_numpy(port_result["scores"]),
        reference_scores,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        torch.from_numpy(port_result["classes"]),
        reference_classes,
        rtol=0.0,
        atol=0.0,
    )
