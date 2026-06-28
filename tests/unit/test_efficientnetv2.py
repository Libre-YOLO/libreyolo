"""LibreEfficientNetV2 unit tests: registry, discriminators, forward, postprocess."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

from libreyolo.models.efficientnetv2.model import LibreEfficientNetV2  # noqa: E402
from libreyolo.models.efficientnetv2.nn import EfficientNetV2  # noqa: E402
from libreyolo.postprocess.efficientnetv2 import postprocess  # noqa: E402

# (size, eval resolution, num_features signature used by detect_size)
SIZES = [("b0", 224, 1280), ("b1", 240, 1280), ("b2", 260, 1408), ("b3", 300, 1536)]


def test_registered_and_classify_task():
    from libreyolo.models.base import BaseModel

    assert any(c.__name__ == "LibreEfficientNetV2" for c in BaseModel._registry)
    m = LibreEfficientNetV2(size="b0", device="cpu")
    assert m.family == "efficientnetv2"
    assert m.task == "classify"
    assert m.input_size == 224
    assert m.crop_pct == 0.875


def test_filename_detection():
    for size in ("b0", "b1", "b2", "b3"):
        fn = f"LibreEfficientNetV2{size}-cls.pt"
        assert LibreEfficientNetV2.detect_size_from_filename(fn) == size
        assert LibreEfficientNetV2.detect_task_from_filename(fn) == "classify"


@pytest.mark.parametrize("size,res,num_features", SIZES)
def test_detect_size_and_nc(size, res, num_features):
    sd = EfficientNetV2(size=size, num_classes=1000).state_dict()
    assert LibreEfficientNetV2.can_load(sd) is True
    assert LibreEfficientNetV2.detect_size(sd) == size
    assert LibreEfficientNetV2.detect_nb_classes(sd) == 1000
    assert int(sd["conv_head.weight"].shape[0]) == num_features


def test_sibling_rejection_bidirectional():
    """EfficientNetV2 must not steal MobileNetV4 or detector checkpoints, and vice versa."""
    from libreyolo import LibreYOLOX
    from libreyolo.models.mobilenetv4.model import LibreMobileNetV4
    from libreyolo.models.mobilenetv4.nn import MobileNetV4

    effv2_sd = EfficientNetV2(size="b0", num_classes=1000).state_dict()
    mnv4_sd = MobileNetV4(size="s", num_classes=1000).state_dict()
    yolox_sd = LibreYOLOX(size="s", device="cpu").model.state_dict()

    assert LibreEfficientNetV2.can_load(effv2_sd) is True
    assert LibreEfficientNetV2.can_load(mnv4_sd) is False
    assert LibreEfficientNetV2.can_load(yolox_sd) is False
    assert LibreMobileNetV4.can_load(effv2_sd) is False
    assert LibreYOLOX.can_load(effv2_sd) is False


@pytest.mark.parametrize("size,res,num_features", SIZES)
def test_forward_shape(size, res, num_features):
    nc = 1000 if size == "b0" else 17
    net = EfficientNetV2(size=size, num_classes=nc).eval()
    with torch.no_grad():
        out = net(torch.zeros(2, 3, res, res))
    assert out.shape == (2, nc)


def test_reset_classifier():
    net = EfficientNetV2(size="b0", num_classes=1000)
    net.reset_classifier(7)
    assert net.classifier.out_features == 7
    net.eval()  # eval -> BatchNorm uses running stats (batch-1 head is fine)
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
