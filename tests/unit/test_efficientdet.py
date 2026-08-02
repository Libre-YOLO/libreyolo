"""EfficientDet registry signatures and fixed-scale contracts."""

from __future__ import annotations

import torch
import pytest

from libreyolo import (
    LibreConvNeXt,
    LibreEfficientDet,
    LibreEfficientNetV2,
    LibreMobileNetV4,
    LibrePICODET,
    LibreRTMDet,
    LibreYOLOX,
)

pytestmark = pytest.mark.unit


def _efficientdet_state(fpn_channels: int = 64, classes: int = 90) -> dict[str, torch.Tensor]:
    return {
        "backbone.conv_stem.weight": torch.empty(32, 3, 3, 3),
        "fpn.cell.0.fnode.0.combine.edge_weights": torch.empty(2),
        "class_net.predict.conv_pw.weight": torch.empty(9 * classes, fpn_channels, 1, 1),
        "box_net.predict.conv_pw.weight": torch.empty(36, fpn_channels, 1, 1),
    }


def _neighbor_states() -> list[tuple[type, dict[str, torch.Tensor]]]:
    return [
        (
            LibreEfficientNetV2,
            {
                "conv_stem.weight": torch.empty(32, 3, 3, 3),
                "conv_head.weight": torch.empty(1280, 320, 1, 1),
                "classifier.weight": torch.empty(1000, 1280),
                "blocks.1.0.se.conv_reduce.weight": torch.empty(8, 32, 1, 1),
            },
        ),
        (
            LibreMobileNetV4,
            {
                "conv_stem.weight": torch.empty(32, 3, 3, 3),
                "conv_head.weight": torch.empty(1280, 960, 1, 1),
                "classifier.weight": torch.empty(1000, 1280),
                "blocks.0.0.conv.weight": torch.empty(32, 32, 3, 3),
                "blocks.1.0.pw_exp.conv.weight": torch.empty(64, 32, 1, 1),
            },
        ),
        (
            LibreConvNeXt,
            {
                "stem.0.weight": torch.empty(96, 3, 4, 4),
                "head.fc.weight": torch.empty(1000, 768),
                "stages.0.blocks.0.gamma": torch.empty(96),
                "stages.2.blocks.8.gamma": torch.empty(384),
            },
        ),
        (
            LibrePICODET,
            {
                "backbone.blocks.0.conv_pw_2.conv.weight": torch.empty(48, 24, 1, 1),
                "neck.trans.0.conv.weight": torch.empty(96, 128, 1, 1),
                "head.gfl_cls.0.weight": torch.empty(112, 96, 1, 1),
            },
        ),
        (LibreRTMDet, {"head.rtm_cls.0.weight": torch.empty(80, 96, 1, 1)}),
        (LibreYOLOX, {"head.stems.0.conv.weight": torch.empty(128, 256, 1, 1)}),
    ]


def test_efficientdet_recognizes_only_its_full_signature():
    state = _efficientdet_state()
    assert LibreEfficientDet.can_load(state)
    assert LibreEfficientDet.detect_size(state) == "d0"
    assert LibreEfficientDet.detect_nb_classes(state) == 80

    for missing in state:
        incomplete = dict(state)
        incomplete.pop(missing)
        assert not LibreEfficientDet.can_load(incomplete)


def test_efficientdet_and_neighbor_discriminators_reject_each_other():
    efficientdet = _efficientdet_state()
    for neighbor, state in _neighbor_states():
        assert neighbor.can_load(state), f"invalid {neighbor.__name__} test fixture"
        assert not LibreEfficientDet.can_load(state)
        assert not neighbor.can_load(efficientdet)


def test_efficientdet_size_and_filename_contracts():
    expected = {"d0": 64, "d1": 88, "d2": 112, "d3": 160, "d4": 224}
    for size, channels in expected.items():
        state = _efficientdet_state(fpn_channels=channels)
        assert LibreEfficientDet.detect_size(state) == size
        assert LibreEfficientDet.detect_size_from_filename(f"LibreEfficientDet{size}.pt") == size
        assert LibreEfficientDet.detect_size_from_filename(f"tf_efficientdet_{size}_40.pth") == size

    assert LibreEfficientDet.INPUT_SIZES == {
        "d0": 512,
        "d1": 640,
        "d2": 768,
        "d3": 896,
        "d4": 1024,
    }
    assert LibreEfficientDet.detect_size(_efficientdet_state(fpn_channels=72)) is None
    assert LibreEfficientDet.detect_size_from_filename("LibreEfficientNetV2b0-cls.pt") is None


def test_efficientdet_rejects_wrong_head_shapes():
    bad_classes = _efficientdet_state()
    bad_classes["class_net.predict.conv_pw.weight"] = torch.empty(811, 64, 1, 1)
    assert not LibreEfficientDet.can_load(bad_classes)

    bad_boxes = _efficientdet_state()
    bad_boxes["box_net.predict.conv_pw.weight"] = torch.empty(45, 64, 1, 1)
    assert not LibreEfficientDet.can_load(bad_boxes)
