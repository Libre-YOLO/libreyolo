"""LibreMobileNetV4 unit tests: registry, discriminators, forward, postprocess."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

from libreyolo.models.mobilenetv4.model import LibreMobileNetV4  # noqa: E402
from libreyolo.models.mobilenetv4.nn import MobileNetV4  # noqa: E402
from libreyolo.postprocess.mobilenetv4 import postprocess  # noqa: E402


def test_registered_and_classify_task():
    from libreyolo.models.base import BaseModel

    assert any(c.__name__ == "LibreMobileNetV4" for c in BaseModel._registry)
    m = LibreMobileNetV4(size="s", device="cpu")
    assert m.family == "mobilenetv4"
    assert m.task == "classify"
    assert m.input_size == 224
    assert m.crop_pct == 0.875


def test_filename_detection():
    assert LibreMobileNetV4.detect_size_from_filename("LibreMobileNetV4s-cls.pt") == "s"
    assert LibreMobileNetV4.detect_size_from_filename("LibreMobileNetV4m-cls.pt") == "m"
    assert LibreMobileNetV4.detect_size_from_filename("LibreMobileNetV4l-cls.pt") == "l"
    assert LibreMobileNetV4.detect_task_from_filename("LibreMobileNetV4s-cls.pt") == "classify"


@pytest.mark.parametrize("size,stem", [("s", 32), ("m", 32), ("l", 24)])
def test_detect_size_and_nc(size, stem):
    sd = MobileNetV4(size=size, num_classes=1000).state_dict()
    assert LibreMobileNetV4.can_load(sd) is True
    assert LibreMobileNetV4.detect_size(sd) == size
    assert LibreMobileNetV4.detect_nb_classes(sd) == 1000
    assert int(sd["conv_stem.weight"].shape[0]) == stem


def test_sibling_rejection_bidirectional():
    """MobileNetV4 must not steal detector checkpoints, and vice versa."""
    from libreyolo import LibreYOLOX

    mnv4_sd = MobileNetV4(size="s", num_classes=1000).state_dict()
    yolox_sd = LibreYOLOX(size="s", device="cpu").model.state_dict()

    assert LibreMobileNetV4.can_load(mnv4_sd) is True
    assert LibreMobileNetV4.can_load(yolox_sd) is False
    assert LibreYOLOX.can_load(mnv4_sd) is False


@pytest.mark.parametrize("size,res,nc", [("s", 224, 1000), ("m", 224, 37), ("l", 256, 5)])
def test_forward_shape(size, res, nc):
    net = MobileNetV4(size=size, num_classes=nc).eval()
    with torch.no_grad():
        out = net(torch.zeros(2, 3, res, res))
    assert out.shape == (2, nc)


def test_reset_classifier():
    net = MobileNetV4(size="s", num_classes=1000)
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
