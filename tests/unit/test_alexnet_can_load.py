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


def test_classifier_family_discriminators_are_bidirectional():
    """AlexNet and the four native classifier siblings reject one another."""
    from libreyolo import (
        LibreAlexNet,
        LibreConvNeXt,
        LibreEfficientNetV2,
        LibreMobileNetV4,
        LibreResNet,
    )

    def meta(*shape):
        return torch.empty(shape, device="meta")

    sibling_signatures = [
        (
            LibreResNet,
            {
                "conv1.weight": meta(64, 3, 7, 7),
                "fc.weight": meta(1000, 512),
                **{
                    f"layer{stage}.{block}.conv1.weight": meta(1)
                    for stage in range(1, 5)
                    for block in range(2)
                },
            },
        ),
        (
            LibreConvNeXt,
            {
                "stem.0.weight": meta(96, 3, 4, 4),
                "head.fc.weight": meta(1000, 768),
                **{f"stages.2.blocks.{block}.gamma": meta(384) for block in range(9)},
            },
        ),
        (
            LibreEfficientNetV2,
            {
                "conv_stem.weight": meta(32, 3, 3, 3),
                "conv_head.weight": meta(1280, 192, 1, 1),
                "classifier.weight": meta(1000, 1280),
                "blocks.1.0.se.conv_reduce.weight": meta(8, 32, 1, 1),
            },
        ),
        (
            LibreMobileNetV4,
            {
                "conv_stem.weight": meta(32, 3, 3, 3),
                "conv_head.weight": meta(1280, 960, 1, 1),
                "classifier.weight": meta(1000, 1280),
                "blocks.0.0.conv.weight": meta(32, 32, 3, 3),
                "blocks.1.0.pw_exp.conv.weight": meta(64, 32, 1, 1),
            },
        ),
    ]

    alexnet = _alexnet_signature()
    for sibling, signature in sibling_signatures:
        assert sibling.can_load(signature) is True
        assert LibreAlexNet.can_load(signature) is False
        assert sibling.can_load(alexnet) is False


def test_training_is_explicitly_out_of_scope():
    from libreyolo import LibreAlexNet

    with pytest.raises(NotImplementedError, match="inference-only museum"):
        LibreAlexNet(size="b", device="cpu").train(data="unused")
