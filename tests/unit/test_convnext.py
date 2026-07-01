"""LibreConvNeXt unit tests: registry, discriminators, forward, postprocess."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

from libreyolo.models.convnext.model import LibreConvNeXt  # noqa: E402
from libreyolo.models.convnext.nn import ConvNeXt  # noqa: E402
from libreyolo.postprocess.convnext import postprocess  # noqa: E402


def test_registered_and_classify_task():
    from libreyolo.models.base import BaseModel

    assert any(c.__name__ == "LibreConvNeXt" for c in BaseModel._registry)
    m = LibreConvNeXt(size="t", device="cpu")
    assert m.family == "convnext"
    assert m.task == "classify"
    assert m.input_size == 224
    assert m.crop_pct == 0.875


def test_filename_detection():
    assert LibreConvNeXt.detect_size_from_filename("LibreConvNeXtt-cls.pt") == "t"
    assert LibreConvNeXt.detect_size_from_filename("LibreConvNeXts-cls.pt") == "s"
    assert LibreConvNeXt.detect_size_from_filename("LibreConvNeXtb-cls.pt") == "b"
    assert LibreConvNeXt.detect_task_from_filename("LibreConvNeXtt-cls.pt") == "classify"


@pytest.mark.parametrize("size,dim0", [("t", 96), ("s", 96), ("b", 128)])
def test_detect_size_and_nc(size, dim0):
    sd = ConvNeXt(size=size, num_classes=1000).state_dict()
    assert LibreConvNeXt.can_load(sd) is True
    assert LibreConvNeXt.detect_size(sd) == size
    assert LibreConvNeXt.detect_nb_classes(sd) == 1000
    assert int(sd["stem.0.weight"].shape[0]) == dim0


def test_sibling_rejection_bidirectional():
    """ConvNeXt must not steal MobileNetV4/detector checkpoints, and vice versa."""
    from libreyolo import LibreYOLOX
    from libreyolo.models.mobilenetv4.nn import MobileNetV4

    cnx_sd = ConvNeXt(size="t", num_classes=1000).state_dict()
    mnv4_sd = MobileNetV4(size="s", num_classes=1000).state_dict()
    yolox_sd = LibreYOLOX(size="s", device="cpu").model.state_dict()

    from libreyolo.models.mobilenetv4.model import LibreMobileNetV4

    assert LibreConvNeXt.can_load(cnx_sd) is True
    assert LibreConvNeXt.can_load(mnv4_sd) is False
    assert LibreConvNeXt.can_load(yolox_sd) is False
    assert LibreMobileNetV4.can_load(cnx_sd) is False
    assert LibreYOLOX.can_load(cnx_sd) is False


@pytest.mark.parametrize("size,res,nc", [("t", 224, 1000), ("s", 224, 37), ("b", 224, 5)])
def test_forward_shape(size, res, nc):
    net = ConvNeXt(size=size, num_classes=nc).eval()
    with torch.no_grad():
        out = net(torch.zeros(2, 3, res, res))
    assert out.shape == (2, nc)


def test_reset_classifier():
    net = ConvNeXt(size="t", num_classes=1000)
    net.reset_classifier(7)
    assert net.head.fc.out_features == 7
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
