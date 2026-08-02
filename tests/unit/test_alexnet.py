"""LibreAlexNet architecture smoke tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.unit, pytest.mark.alexnet]


def test_architecture_signature_and_parameter_count():
    from libreyolo import LibreAlexNet
    from libreyolo.models.alexnet.nn import AlexNet

    model = AlexNet(num_classes=1000)
    state_dict = model.state_dict()

    assert LibreAlexNet.can_load(state_dict) is True
    assert LibreAlexNet.detect_size(state_dict) == "b"
    assert LibreAlexNet.detect_nb_classes(state_dict) == 1000
    assert sum(parameter.numel() for parameter in model.parameters()) == 61_100_840

    stem = model.features[0]
    assert isinstance(stem, nn.Conv2d)
    assert stem.out_channels == 64
    assert stem.kernel_size == (11, 11)
    assert stem.groups == 1
    assert not any(
        isinstance(module, nn.LocalResponseNorm) for module in model.modules()
    )


def test_forward_shape():
    from libreyolo.models.alexnet.nn import AlexNet

    model = AlexNet(num_classes=17).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 224, 224))
    assert logits.shape == (1, 17)
