"""LibreResNet unit tests: registry, discriminators (incl. backbone rejection), forward, postprocess."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

from libreyolo.models.resnet.model import LibreResNet  # noqa: E402
from libreyolo.models.resnet.nn import ResNet  # noqa: E402
from libreyolo.postprocess.resnet import postprocess  # noqa: E402


def test_registered_and_classify_task():
    from libreyolo.models.base import BaseModel

    assert any(c.__name__ == "LibreResNet" for c in BaseModel._registry)
    m = LibreResNet(size="50", device="cpu")
    assert m.family == "resnet"
    assert m.task == "classify"
    assert m.input_size == 224
    assert m.crop_pct == 0.95


def test_filename_detection():
    for s in ["18", "34", "50", "101"]:
        assert LibreResNet.detect_size_from_filename(f"LibreResNet{s}-cls.pt") == s
    assert LibreResNet.detect_task_from_filename("LibreResNet50-cls.pt") == "classify"


@pytest.mark.parametrize("size", ["18", "34", "50", "101"])
def test_detect_size_and_nc(size):
    sd = ResNet(size=size, num_classes=1000).state_dict()
    assert LibreResNet.can_load(sd) is True
    assert LibreResNet.detect_size(sd) == size
    assert LibreResNet.detect_nb_classes(sd) == 1000


def test_rejects_resnet_backbones_and_siblings():
    """ResNet is also a *backbone* inside l2cs/rt-detr; can_load must match only
    the standalone classifier and reject those embeds + the other classifiers."""
    from libreyolo import LibreL2CS, LibreYOLOX
    from libreyolo.models.convnext.nn import ConvNeXt
    from libreyolo.models.efficientnetv2.nn import EfficientNetV2
    from libreyolo.models.mobilenetv4.nn import MobileNetV4

    resnet_sd = ResNet(size="50", num_classes=1000).state_dict()
    l2cs_sd = LibreL2CS(model_path=None, size="r50", device="cpu").model.state_dict()
    yolox_sd = LibreYOLOX(size="s", device="cpu").model.state_dict()

    assert LibreResNet.can_load(resnet_sd) is True
    # l2cs is a ResNet but its head is fc_yaw_gaze/fc_pitch_gaze (no plain `fc`):
    assert LibreResNet.can_load(l2cs_sd) is False
    assert LibreResNet.can_load(yolox_sd) is False
    assert LibreResNet.can_load(MobileNetV4(size="s").state_dict()) is False
    assert LibreResNet.can_load(ConvNeXt(size="t").state_dict()) is False
    assert LibreResNet.can_load(EfficientNetV2(size="b0").state_dict()) is False
    # reverse: l2cs must not claim a standalone resnet classifier
    assert LibreL2CS.can_load(resnet_sd) is False


@pytest.mark.parametrize("size,nc", [("18", 1000), ("50", 10)])
def test_forward_shape(size, nc):
    net = ResNet(size=size, num_classes=nc).eval()
    with torch.no_grad():
        out = net(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, nc)


def test_reset_classifier():
    net = ResNet(size="50", num_classes=1000)
    net.reset_classifier(7)
    assert net.fc.out_features == 7
    net.eval()
    with torch.no_grad():
        assert net(torch.zeros(1, 3, 224, 224)).shape == (1, 7)


def test_postprocess_probs():
    logits = torch.randn(1, 1000)
    out = postprocess(logits)
    assert set(out) == {"probs"}
    probs = out["probs"]
    assert probs.shape == (1000,)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert int(probs.argmax()) == int(logits.argmax())
