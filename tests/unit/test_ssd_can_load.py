"""Strict SSD checkpoint recognition and sibling rejection."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def _ssd_state(num_classes: int = 91) -> dict[str, torch.Tensor]:
    return {
        "backbone.extra.4.2.weight": torch.zeros(256, 128, 3, 3),
        "head.classification_head.module_list.0.weight": torch.zeros(
            4 * num_classes, 512, 3, 3
        ),
        "head.regression_head.module_list.5.weight": torch.zeros(16, 256, 3, 3),
    }


def test_ssd_recognizes_its_structural_signature():
    from libreyolo import LibreSSD

    state = _ssd_state()
    assert LibreSSD.can_load(state) is True
    assert LibreSSD.detect_size(state) == "300"
    assert LibreSSD.detect_nb_classes(state) == 80


@pytest.mark.parametrize(
    "state",
    [
        {"head.stems.0.conv.weight": torch.zeros(1)},
        {"head.cls_convs.0.0.conv.weight": torch.zeros(1)},
        {"head.cls_conv_dw.0.weight": torch.zeros(1)},
        {"head.heads.0.weight": torch.zeros(1)},
    ],
)
def test_ssd_rejects_other_nms_detector_layouts(state):
    from libreyolo import LibreSSD

    assert LibreSSD.can_load(state) is False
