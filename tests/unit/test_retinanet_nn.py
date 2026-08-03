"""RetinaNet native graph topology and deterministic tensor parity."""

from __future__ import annotations

from functools import partial

import pytest
import torch
from torchvision.models import resnet50
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d

from libreyolo.models.retinanet.nn import LibreRetinaNetModel
from libreyolo.utils.coco import COCO91_TO_COCO80

pytestmark = [pytest.mark.unit, pytest.mark.retinanet]


EXPECTED_PARAMETERS = {"r50": 34_014_999, "r50v2": 38_198_935}


def _build_reference(size: str) -> RetinaNet:
    v2 = size == "r50v2"
    backbone = resnet50(
        weights=None,
        norm_layer=torch.nn.BatchNorm2d if v2 else FrozenBatchNorm2d,
    )
    backbone = _resnet_fpn_extractor(
        backbone,
        trainable_layers=5,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(2048 if v2 else 256, 256),
    )
    if v2:
        head = RetinaNetHead(
            256,
            9,
            91,
            norm_layer=partial(torch.nn.GroupNorm, 32),
        )
        model = RetinaNet(backbone, 91, head=head)
    else:
        model = RetinaNet(backbone, 91)
        overwrite_eps(model, 0.0)
    return model.eval()


@pytest.mark.parametrize("size", ["r50", "r50v2"])
def test_topology_head_and_decoded_output_match_pinned_reference(size):
    torch.manual_seed(0)
    reference = _build_reference(size)
    ours = LibreRetinaNetModel(size, num_classes=91).eval()

    assert (
        sum(parameter.numel() for parameter in ours.parameters())
        == (EXPECTED_PARAMETERS[size])
    )
    assert set(ours.state_dict()) == set(reference.state_dict())
    for key, tensor in reference.state_dict().items():
        assert ours.state_dict()[key].shape == tensor.shape
    ours.load_state_dict(reference.state_dict(), strict=True)

    image = torch.randn(1, 3, 64, 96)
    with torch.inference_mode():
        reference_features = list(reference.backbone(image).values())
        reference_heads = reference.head(reference_features)
        actual_heads, actual_features = ours.forward_head(image)

        reference_anchors = reference.anchor_generator(
            ImageList(image, [image.shape[-2:]]), reference_features
        )[0]
        actual_anchors, _ = ours.anchor_generator(image.shape[-2:], actual_features)
        reference_boxes = reference.box_coder.decode_single(
            reference_heads["bbox_regression"][0], reference_anchors
        ).unsqueeze(0)
        source_indices = torch.tensor(
            [
                source
                for source, _target in sorted(
                    COCO91_TO_COCO80.items(), key=lambda item: item[1]
                )
            ],
            dtype=torch.int64,
        )
        reference_scores = torch.sigmoid(reference_heads["cls_logits"]).index_select(
            2, source_indices
        )
        expected_output = torch.cat((reference_boxes, reference_scores), dim=2)
        actual_output = ours(image)

    for expected, actual in zip(reference_features, actual_features):
        assert torch.equal(expected, actual)
    for key in reference_heads:
        assert torch.equal(reference_heads[key], actual_heads[key])
    assert torch.equal(reference_anchors, actual_anchors)
    assert torch.equal(expected_output, actual_output)
    assert actual_output.shape[-1] == 84


def test_custom_class_head_keeps_all_scores():
    model = LibreRetinaNetModel("r50", num_classes=3).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 774, 7)
