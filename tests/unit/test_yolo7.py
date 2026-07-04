"""Unit tests for LibreYOLO7 (YOLOv7, native port of MIT MultimediaTechLab/YOLO).

PR-gate safe: no real weights, no network. The upstream v7.pt strict-loads into
this port (verified 564/564 keys, 0 missing/unexpected) — that structural parity
plus these smokes are the offline floor; formal max_abs_diff parity vs the MMT
package is a gated one-off.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo import LibreYOLO, LibreYOLO7
from libreyolo.models.base import BaseModel
from libreyolo.models.yolo7.blocks import IDetection, RepConv
from libreyolo.models.yolo7.net import YOLOv7Model
from libreyolo.postprocess.yolo7 import decode_v7_head, postprocess

pytestmark = pytest.mark.unit


def test_model_builds_and_forward_shapes():
    model = YOLOv7Model(num_classes=80).eval()
    assert len(model.state_dict()) == 564  # matches upstream v7.pt tensor count
    with torch.no_grad():
        outs = model(torch.zeros(1, 3, 640, 640))
    assert [tuple(o.shape) for o in outs] == [
        (1, 255, 80, 80), (1, 255, 40, 40), (1, 255, 20, 20)
    ]
    assert model.strides == [8, 16, 32]
    assert len(model.anchors) == 3


def test_repconv_two_branch_sum():
    rep = RepConv(8, 8).eval()
    x = torch.randn(1, 8, 4, 4)
    with torch.no_grad():
        out = rep(x)
        expected = rep.act(rep.conv1(x) + rep.conv2(x))
    assert torch.allclose(out, expected)


def test_idetection_implicit_output_channels():
    head = IDetection(64, num_classes=80, anchor_num=3).eval()
    with torch.no_grad():
        out = head(torch.randn(1, 64, 8, 8))
    assert out.shape == (1, 3 * (5 + 80), 8, 8)


def test_decode_v7_golden():
    # all-zero logits -> sigmoid 0.5; xy=(0.5*2-0.5)*stride, wh=(1)^2*anchor
    anchors = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]
    out = torch.zeros(1, 3 * (5 + 1), 1, 1)
    boxes, scores = decode_v7_head(out, anchors, stride=32, num_classes=1)
    assert torch.allclose(boxes[0], torch.tensor([16.0, 16.0, 10.0, 20.0]))
    assert scores[0].item() == pytest.approx(0.25)  # obj 0.5 * cls 0.5


def test_postprocess_recovers_box():
    anchors = [[20.0, 40.0, 20.0, 40.0, 20.0, 40.0]]  # 3 anchors, one head
    out = torch.zeros(1, 3 * (5 + 1), 1, 1)
    out[0, 4, 0, 0] = 10.0  # anchor-0 objectness high
    out[0, 5, 0, 0] = 10.0  # anchor-0 class high
    res = postprocess([out], anchors, [64], num_classes=1, conf_thres=0.5,
                      iou_thres=0.5, input_size=64, original_size=(64, 64))
    assert res["num_detections"] >= 1
    assert res["scores"][0] > 0.9


def test_can_load_unique_and_rejects_siblings():
    v7_sd = YOLOv7Model(num_classes=80).state_dict()
    assert LibreYOLO7.can_load(v7_sd)
    assert LibreYOLO7.detect_size(v7_sd) == "b"
    assert LibreYOLO7.detect_nb_classes(v7_sd) == 80

    # exactly one registered family claims the v7 checkpoint
    matches = [c for c in BaseModel._registry if c.can_load(v7_sd)]
    assert matches == [LibreYOLO7]

    # v7 rejects a YOLOX checkpoint
    from libreyolo.models.yolox.nn import LibreYOLOXModel
    assert not LibreYOLO7.can_load(LibreYOLOXModel(config="s", nb_classes=80).state_dict())


def test_detect_nb_classes_custom():
    sd = YOLOv7Model(num_classes=7).state_dict()
    assert LibreYOLO7.detect_nb_classes(sd) == 7


def test_factory_load_and_predict(tmp_path):
    import numpy as np
    from PIL import Image
    from libreyolo.utils.serialization import build_class_names, wrap_libreyolo_checkpoint

    model = YOLOv7Model(num_classes=80).eval()
    ckpt = wrap_libreyolo_checkpoint(
        model.state_dict(), model_family="yolo7", size="b", task="detect",
        nc=80, names=build_class_names(80), imgsz=640,
    )
    path = tmp_path / "LibreYOLO7b.pt"
    torch.save(ckpt, path)

    m = LibreYOLO(str(path))
    assert type(m).__name__ == "LibreYOLO7"
    assert m.size == "b" and m.nb_classes == 80 and m.task == "detect"

    img = Image.fromarray((np.random.rand(360, 640, 3) * 255).astype(np.uint8))
    res = m.predict(img, conf=0.001)
    r = res[0] if isinstance(res, list) else res
    assert hasattr(r, "boxes")


def test_train_raises():
    m = LibreYOLO7(size="b")
    with pytest.raises(NotImplementedError, match="inference-only"):
        m.train(data="coco.yaml")
