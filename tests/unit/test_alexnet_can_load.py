"""LibreAlexNet factory recognition and filename contract."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.alexnet]


def _alexnet_signature(nc: int = 1000) -> dict[str, torch.Tensor]:
    return {
        "features.0.weight": torch.empty((64, 3, 11, 11), device="meta"),
        "classifier.1.weight": torch.empty((4096, 256 * 6 * 6), device="meta"),
        "classifier.4.weight": torch.empty((4096, 4096), device="meta"),
        "classifier.6.weight": torch.empty((nc, 4096), device="meta"),
    }


def test_registered_with_classification_contract():
    from libreyolo import LibreAlexNet
    from libreyolo.models.base import BaseModel

    assert LibreAlexNet in BaseModel._registry
    model = LibreAlexNet(size="b", device="cpu")
    assert model.family == "alexnet"
    assert model.task == "classify"
    assert model.input_size == 224
    assert model.crop_pct == 0.875
    assert model.interpolation == "bilinear"


def test_filename_requires_classification_suffix():
    from libreyolo import LibreAlexNet

    canonical = "LibreAlexNetb-cls.pt"
    assert LibreAlexNet.detect_size_from_filename(canonical) == "b"
    assert LibreAlexNet.detect_task_from_filename(canonical) == "classify"
    assert LibreAlexNet.detect_size_from_filename("LibreAlexNetb.pt") is None


def test_detects_only_the_shipped_architecture_signature():
    from libreyolo import LibreAlexNet

    state_dict = _alexnet_signature(nc=17)
    assert LibreAlexNet.can_load(state_dict) is True
    assert LibreAlexNet.detect_size(state_dict) == "b"
    assert LibreAlexNet.detect_nb_classes(state_dict) == 17

    wrong_stem = dict(state_dict)
    wrong_stem["features.0.weight"] = torch.empty((64, 3, 3, 3), device="meta")
    assert LibreAlexNet.can_load(wrong_stem) is False

    wrong_hidden = dict(state_dict)
    wrong_hidden["classifier.1.weight"] = torch.empty(
        (2048, 256 * 6 * 6), device="meta"
    )
    assert LibreAlexNet.can_load(wrong_hidden) is False


def test_training_is_explicitly_out_of_scope():
    from libreyolo import LibreAlexNet

    with pytest.raises(NotImplementedError, match="inference-only museum"):
        LibreAlexNet(size="b", device="cpu").train(data="unused")
