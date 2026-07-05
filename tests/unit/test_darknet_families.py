"""Unit tests for the Darknet-lineage families (YOLOv2 / v3 / v4).

All PR-gate safe: no network, no real weights, no GPU. The engine is exercised
with synthetic ``.weights`` blobs (built by saving a random-init net) so the
full cfg-parse -> build -> read -> load -> decode pipeline is covered offline.
Numerical parity vs a Darknet oracle and COCO mAP are separate external gates.
"""

from __future__ import annotations

import math

import pytest
import torch

from libreyolo import LibreYOLO, LibreYOLO2, LibreYOLO3, LibreYOLO4
from libreyolo.models.base import BaseModel
from libreyolo.models.darknet import (
    build_darknet,
    load_bundled_cfg,
    load_darknet_weights,
    save_darknet_weights,
)
from libreyolo.models.darknet.blocks import (
    DARKNET_BN_EPS,
    DarknetBatchNorm2d,
    Mish,
    Reorg,
)
from libreyolo.models.darknet.net import DetectionSpec
from libreyolo.postprocess.darknet_yolo import decode_head, postprocess

pytestmark = pytest.mark.unit


# Known architecture facts (conv counts are byte-exact vs the real .weights).
CFG_FACTS = {
    "yolov3": dict(convs=75, heads=3, kind="yolo", strides={8, 16, 32}, nch=255),
    "yolov3-tiny": dict(convs=13, heads=2, kind="yolo", strides={16, 32}, nch=255),
    "yolov3-spp": dict(convs=76, heads=3, kind="yolo", strides={8, 16, 32}, nch=255),
    "yolov4": dict(convs=110, heads=3, kind="yolo", strides={8, 16, 32}, nch=255),
    "yolov4-tiny": dict(convs=21, heads=2, kind="yolo", strides={16, 32}, nch=255),
    "yolov2": dict(convs=23, heads=1, kind="region", strides={32}, nch=425),
    "yolov2-tiny": dict(convs=9, heads=1, kind="region", strides={32}, nch=425),
}
TINY_CFGS = ["yolov3-tiny", "yolov4-tiny", "yolov2-tiny"]
ALL_CFGS = list(CFG_FACTS)


# ---------------------------------------------------------------------------
# cfg parser
# ---------------------------------------------------------------------------
def test_cfg_parser_reads_bundled_yolov3():
    net_options, layers = load_bundled_cfg("yolov3")
    assert int(net_options["channels"]) == 3
    types = [l.type for l in layers]
    assert types.count("convolutional") == 75
    assert types.count("yolo") == 3
    assert "shortcut" in types and "route" in types and "upsample" in types


def test_cfg_parser_missing_cfg_lists_available():
    from libreyolo.models.darknet.cfg import load_bundled_cfg as _load

    with pytest.raises(FileNotFoundError) as e:
        _load("does-not-exist")
    assert "yolov3.cfg" in str(e.value)


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------
def test_darknet_batchnorm_uses_std_plus_eps_not_var_plus_eps():
    bn = DarknetBatchNorm2d(2).eval()
    with torch.no_grad():
        bn.weight.copy_(torch.tensor([2.0, 1.0]))
        bn.bias.copy_(torch.tensor([0.5, -1.0]))
        bn.running_mean.copy_(torch.tensor([1.0, 0.0]))
        bn.running_var.copy_(torch.tensor([4.0, 9.0]))
    x = torch.tensor([3.0, 6.0]).view(1, 2, 1, 1)
    out = bn(x).view(-1)
    # exact darknet formula: (x-mean)/(sqrt(var)+eps)*w + b
    exp0 = (3.0 - 1.0) / (math.sqrt(4.0) + DARKNET_BN_EPS) * 2.0 + 0.5
    exp1 = (6.0 - 0.0) / (math.sqrt(9.0) + DARKNET_BN_EPS) * 1.0 - 1.0
    assert out[0].item() == pytest.approx(exp0, abs=1e-5)
    assert out[1].item() == pytest.approx(exp1, abs=1e-5)

    # must differ from torch's (x-mean)/sqrt(var+eps): the whole point of the
    # custom module (small but compounding difference).
    torch_bn = torch.nn.BatchNorm2d(2, eps=DARKNET_BN_EPS).eval()
    with torch.no_grad():
        torch_bn.weight.copy_(bn.weight)
        torch_bn.bias.copy_(bn.bias)
        torch_bn.running_mean.copy_(bn.running_mean)
        torch_bn.running_var.copy_(bn.running_var)
    assert not torch.allclose(out, torch_bn(x).view(-1), rtol=0.0, atol=1e-7)


def test_mish_matches_formula():
    x = torch.linspace(-3, 3, 7)
    out = Mish()(x)
    exp = x * torch.tanh(torch.nn.functional.softplus(x))
    assert torch.allclose(out, exp, atol=1e-6)


def test_reorg_shape_and_channel_expansion():
    x = torch.randn(1, 4, 4, 4)
    out = Reorg(stride=2)(x)
    assert out.shape == (1, 16, 2, 2)


def test_reorg_channel_ordering_golden():
    # Pins the exact darknet2pytorch (marvis) channel interleaving used to load
    # real YOLOv2 passthrough weights — a regression guard on the ordering, not
    # just the shape. (Functional correctness vs Darknet is confirmed by real
    # LibreYOLO2b detections; OpenCV's Reorg layer is not a valid oracle here.)
    x = torch.arange(2 * 4 * 4, dtype=torch.float32).view(1, 2, 4, 4)
    out = Reorg(stride=2)(x)
    assert out.shape == (1, 8, 2, 2)
    expected = [
        0, 2, 8, 10, 16, 18, 24, 26, 1, 3, 9, 11, 17, 19, 25, 27,
        4, 6, 12, 14, 20, 22, 28, 30, 5, 7, 13, 15, 21, 23, 29, 31,
    ]
    assert out.flatten().tolist() == [float(v) for v in expected]


# ---------------------------------------------------------------------------
# net build + weights reader/writer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", ALL_CFGS)
def test_net_builds_with_expected_structure(cfg):
    facts = CFG_FACTS[cfg]
    net = build_darknet(cfg, num_classes=80).eval()
    n_conv = sum(1 for m in net.layers if type(m).__name__ == "DarknetConv")
    specs = net.detection_specs()
    assert n_conv == facts["convs"]
    assert len(specs) == facts["heads"]
    assert {s.stride for s in specs} == facts["strides"]
    assert all(s.kind == facts["kind"] for s in specs)


@pytest.mark.parametrize("cfg", TINY_CFGS)
def test_forward_output_shapes(cfg):
    net = build_darknet(cfg, num_classes=80).eval()
    size = net.input_width
    with torch.no_grad():
        outs = net(torch.randn(1, 3, size, size))
    specs = net.detection_specs()
    assert len(outs) == len(specs)
    for out, spec in zip(outs, specs):
        a = len(spec.anchors)
        assert out.shape[1] == a * (5 + spec.num_classes)
        assert out.shape[2] == size // spec.stride


@pytest.mark.parametrize("cfg", TINY_CFGS)
def test_weights_roundtrip_exact(cfg):
    net = build_darknet(cfg, num_classes=80).eval()
    blob = save_darknet_weights(net)
    net2 = build_darknet(cfg, num_classes=80).eval()
    load_darknet_weights(net2, blob)
    sd1, sd2 = net.state_dict(), net2.state_dict()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), k


def test_weights_byte_exact_size_matches_real_yolov3():
    # The real yolov3.weights is 248,007,048 bytes. A synthetic blob built from
    # our architecture must match — proof the param layout matches Darknet.
    net = build_darknet("yolov3", num_classes=80).eval()
    assert len(save_darknet_weights(net)) == 248007048


def test_truncated_weights_raises():
    net = build_darknet("yolov3-tiny", num_classes=80).eval()
    blob = save_darknet_weights(net)
    with pytest.raises(ValueError, match="truncated|not fully consumed"):
        load_darknet_weights(net, blob[: len(blob) // 2])


def test_extra_weights_raises():
    net = build_darknet("yolov3-tiny", num_classes=80).eval()
    blob = save_darknet_weights(net)
    with pytest.raises(ValueError, match="not fully consumed"):
        load_darknet_weights(net, blob + b"\x00\x00\x00\x00" * 100)


# ---------------------------------------------------------------------------
# decode goldens (hand-computed)
# ---------------------------------------------------------------------------
def test_decode_yolo_center_golden():
    spec = DetectionSpec("yolo", 0, [(10.0, 20.0)], 1, stride=32)
    boxes, scores = decode_head(torch.zeros(1, 6, 1, 1), spec, input_size=32)
    assert torch.allclose(boxes[0], torch.tensor([16.0, 16.0, 10.0, 20.0]))
    assert scores[0].item() == pytest.approx(0.25)  # sigmoid(0)obj * sigmoid(0)cls


def test_decode_yolov4_scale_xy_center_unchanged():
    spec = DetectionSpec("yolo", 0, [(10.0, 20.0)], 1, stride=32, scale_x_y=2.0)
    boxes, _ = decode_head(torch.zeros(1, 6, 1, 1), spec, 32)
    assert torch.allclose(boxes[0], torch.tensor([16.0, 16.0, 10.0, 20.0]))


def test_decode_region_uses_gridcell_anchors_and_softmax():
    spec = DetectionSpec("region", 0, [(0.5, 1.0)], 2, stride=32, softmax=True)
    boxes, scores = decode_head(torch.zeros(1, 7, 1, 1), spec, 32)
    # grid-cell anchor 0.5 -> pixel width 0.5*32 = 16
    assert torch.allclose(boxes[0], torch.tensor([16.0, 16.0, 16.0, 32.0]))
    # softmax over two equal logits -> 0.5 each, times obj 0.5
    assert torch.allclose(scores[0], torch.tensor([0.25, 0.25]))


def test_decode_grid_offsets():
    # a single anchor on a 2x2 grid, all logits 0 -> centers at cell centers*stride
    spec = DetectionSpec("yolo", 0, [(4.0, 4.0)], 1, stride=10)
    boxes, _ = decode_head(torch.zeros(1, 6, 2, 2), spec, 20)
    centers = {(round(b[0].item()), round(b[1].item())) for b in boxes}
    assert centers == {(5, 5), (15, 5), (5, 15), (15, 15)}


def test_postprocess_recovers_single_box():
    # one head, 1 anchor, 1 class, 1x1 grid, strong detection at center
    spec = DetectionSpec("yolo", 0, [(20.0, 40.0)], 1, stride=64)
    out = torch.zeros(1, 6, 1, 1)
    out[0, 4, 0, 0] = 10.0  # objectness logit high
    out[0, 5, 0, 0] = 10.0  # class logit high
    res = postprocess([out], [spec], conf_thres=0.5, iou_thres=0.5,
                      input_size=64, original_size=(64, 64), ratio=1.0)
    assert res["num_detections"] == 1
    x1, y1, x2, y2 = res["boxes"][0]
    assert (x1 + x2) / 2 == pytest.approx(32.0, abs=1.0)
    assert res["scores"][0] > 0.9


def test_postprocess_length_mismatch_raises():
    spec = DetectionSpec("yolo", 0, [(1.0, 1.0)], 1, stride=32)
    with pytest.raises(ValueError, match="one spec per head"):
        postprocess([torch.zeros(1, 6, 1, 1)], [spec, spec], input_size=32)


# ---------------------------------------------------------------------------
# family registry discrimination
# ---------------------------------------------------------------------------
DISCRIMINATION = [
    ("yolov3", "yolo3", LibreYOLO3, "b"),
    ("yolov3-tiny", "yolo3", LibreYOLO3, "t"),
    ("yolov3-spp", "yolo3", LibreYOLO3, "spp"),
    ("yolov4", "yolo4", LibreYOLO4, "b"),
    ("yolov4-tiny", "yolo4", LibreYOLO4, "t"),
    ("yolov2", "yolo2", LibreYOLO2, "b"),
    ("yolov2-tiny", "yolo2", LibreYOLO2, "t"),
]


def _prefixed_state_dict(cfg, family, nc=80):
    net = build_darknet(cfg, num_classes=nc).eval()
    return {f"{family}.{k}": v for k, v in net.state_dict().items()}


@pytest.mark.parametrize("cfg,family,cls,size", DISCRIMINATION)
def test_can_load_is_exclusive(cfg, family, cls, size):
    sd = _prefixed_state_dict(cfg, family)
    matches = [c for c in BaseModel._registry if c.can_load(sd)]
    assert matches == [cls]
    assert cls.detect_size(sd) == size
    assert cls.detect_nb_classes(sd) == 80


def test_detect_nb_classes_custom_count():
    sd = _prefixed_state_dict("yolov3-tiny", "yolo3", nc=7)
    assert LibreYOLO3.detect_nb_classes(sd) == 7


def test_darknet_families_reject_foreign_checkpoint():
    from libreyolo.models.yolox.nn import LibreYOLOXModel

    yolox_sd = LibreYOLOXModel(config="s", nb_classes=80).state_dict()
    for cls in (LibreYOLO2, LibreYOLO3, LibreYOLO4):
        assert not cls.can_load(yolox_sd)


# ---------------------------------------------------------------------------
# end-to-end: convert -> factory load -> predict (synthetic weights)
# ---------------------------------------------------------------------------
def _make_checkpoint(tmp_path, cfg, family, size):
    from libreyolo.utils.serialization import (
        build_class_names,
        wrap_libreyolo_checkpoint,
    )

    net = build_darknet(cfg, num_classes=80).eval()
    blob = save_darknet_weights(net)
    net2 = build_darknet(cfg, num_classes=80).eval()
    load_darknet_weights(net2, blob)
    sd = {f"{family}.{k}": v for k, v in net2.state_dict().items()}
    ckpt = wrap_libreyolo_checkpoint(
        sd, model_family=family, size=size, task="detect",
        nc=80, names=build_class_names(80), imgsz=net.input_width,
    )
    path = tmp_path / f"Libre{family.upper()}{size}.pt"
    torch.save(ckpt, path)
    return str(path)


def test_checkpoint_metadata_is_valid(tmp_path):
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    path = _make_checkpoint(tmp_path, "yolov3-tiny", "yolo3", "t")
    loaded = torch.load(path, weights_only=False)
    assert validate_checkpoint_metadata(loaded, strict=True) == []


@pytest.mark.parametrize("cfg,family,cls,size", [
    ("yolov3-tiny", "yolo3", LibreYOLO3, "t"),
    ("yolov4-tiny", "yolo4", LibreYOLO4, "t"),
    ("yolov2-tiny", "yolo2", LibreYOLO2, "t"),
])
def test_factory_load_and_predict(tmp_path, cfg, family, cls, size):
    import numpy as np
    from PIL import Image

    path = _make_checkpoint(tmp_path, cfg, family, size)
    model = LibreYOLO(path)  # no size/nc hints -> must auto-detect
    assert type(model).__name__ == cls.__name__
    assert model.size == size and model.nb_classes == 80 and model.task == "detect"

    img = Image.fromarray((np.random.rand(320, 480, 3) * 255).astype(np.uint8))
    results = model.predict(img, conf=0.001)
    r = results[0] if isinstance(results, list) else results
    assert hasattr(r, "boxes")


def test_train_raises_not_implemented():
    model = LibreYOLO3(size="t")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="coco.yaml")
