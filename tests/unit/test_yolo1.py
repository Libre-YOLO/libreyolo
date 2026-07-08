"""Unit tests for the LibreYOLO1 (YOLOv1) museum family.

All PR-gate safe: no network, no real weights, no GPU. The cfg-driven engine is
exercised with synthetic ``.weights`` blobs (built by saving a random-init net)
so the full parse -> build -> read -> load -> decode -> predict pipeline is
covered offline. The ``[connected]`` (FC) and ``[local]`` (locally-connected)
heads are the YOLOv1-specific hazards, so they get dedicated numeric checks.

Numerical faithfulness against the real pretrained weights (the classic
dog/bicycle/car detection) is the separately-gated ``test_yolo1b_golden_*``
below; it needs the converted checkpoint and stays skipped in CI.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from libreyolo import LibreYOLO, LibreYOLO1, LibreYOLO2, LibreYOLO3, LibreYOLO4
from libreyolo.models.base import BaseModel
from libreyolo.models.darknet import (
    build_darknet,
    load_bundled_cfg,
    load_darknet_weights,
    save_darknet_weights,
)
from libreyolo.models.darknet.blocks import DarknetConnected, DarknetLocal
from libreyolo.models.darknet.net import DetectionSpec
from libreyolo.models.darknet.preprocess import preprocess_numpy_stretch
from libreyolo.models.yolo1.model import VOC_NAMES
from libreyolo.postprocess.darknet_yolo import decode_detection, postprocess

pytestmark = pytest.mark.unit


# Byte-exact vs the real pjreddie .weights files (16-byte v0.1 header + floats).
_FULL_BYTES = 777018888       # yolov1.weights (Nov 2016 release)
_TINY_BYTES = 108399816       # tiny-yolo.weights


# ---------------------------------------------------------------------------
# cfg parser
# ---------------------------------------------------------------------------
def test_cfg_parser_reads_bundled_yolov1_full():
    net_options, layers = load_bundled_cfg("yolov1")
    assert int(net_options["width"]) == 448 and int(net_options["height"]) == 448
    types = [layer.type for layer in layers]
    assert types.count("convolutional") == 24
    assert types.count("local") == 1
    assert types.count("connected") == 1
    assert types.count("dropout") == 1
    assert types.count("detection") == 1
    det = layers[-1]
    assert det.type == "detection"
    assert det.get_int("side") == 7
    assert det.get_int("num") == 2
    assert det.get_int("classes") == 20
    assert det.get_int("sqrt") == 1
    # The head width must equal S*S*(C + B*(coords+1)) = 49*30 = 1470.
    connected = [layer for layer in layers if layer.type == "connected"][0]
    assert connected.get_int("output") == 1470


def test_cfg_parser_reads_bundled_yolov1_tiny():
    _, layers = load_bundled_cfg("yolov1-tiny")
    types = [layer.type for layer in layers]
    assert types.count("convolutional") == 8
    assert types.count("local") == 0  # tiny has no locally-connected layer
    assert types.count("connected") == 1
    det = layers[-1]
    assert det.get_int("side") == 7 and det.get_int("num") == 2


# ---------------------------------------------------------------------------
# byte-exact weight layout (the reader must consume the file to the last byte)
# ---------------------------------------------------------------------------
def test_full_weights_byte_count_matches_real_file():
    # Building the full net once (194M params). Assert the serialized size equals
    # the real yolov1.weights byte-for-byte via the state-dict numel sum instead
    # of materializing a 777MB blob. The sum is exact because every state-dict
    # entry is a float32 the reader/writer serializes: DarknetBatchNorm2d is a
    # custom module with no ``num_batches_tracked`` (guarded below so a future
    # swap to stock BatchNorm2d cannot silently skew the count).
    net = build_darknet("yolov1", num_classes=20).eval()
    state = net.state_dict()
    assert all(v.dtype == torch.float32 for v in state.values())
    assert not any("num_batches_tracked" in k for k in state)
    floats = sum(v.numel() for v in state.values())
    assert 16 + 4 * floats == _FULL_BYTES


def test_tiny_weights_byte_count_and_roundtrip():
    net = build_darknet("yolov1-tiny", num_classes=20).eval()
    blob = save_darknet_weights(net, major=0, minor=1)  # v1 header (uint32 seen)
    assert len(blob) == _TINY_BYTES
    net2 = build_darknet("yolov1-tiny", num_classes=20).eval()
    load_darknet_weights(net2, blob)
    sd1, sd2 = net.state_dict(), net2.state_dict()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), k


def test_truncated_weights_raises():
    net = build_darknet("yolov1-tiny", num_classes=20).eval()
    blob = save_darknet_weights(net, major=0, minor=1)
    with pytest.raises(ValueError, match="truncated|not fully consumed"):
        load_darknet_weights(net, blob[: len(blob) // 2])


def test_connected_head_roundtrips_without_transpose():
    # v0.1 files store the FC matrix as [out, in] (no transpose); a save->load
    # cycle must reproduce the connected weight exactly.
    net = build_darknet("yolov1-tiny", num_classes=20).eval()
    blob = save_darknet_weights(net, major=0, minor=1)
    net2 = build_darknet("yolov1-tiny", num_classes=20).eval()
    load_darknet_weights(net2, blob)
    connected = [m for m in net.layers if isinstance(m, DarknetConnected)]
    connected2 = [m for m in net2.layers if isinstance(m, DarknetConnected)]
    assert len(connected) == 1
    assert torch.equal(connected[0].linear.weight, connected2[0].linear.weight)
    assert torch.equal(connected[0].linear.bias, connected2[0].linear.bias)


# ---------------------------------------------------------------------------
# [connected] and [local] blocks (the YOLOv1 port hazards)
# ---------------------------------------------------------------------------
def test_connected_flattens_channel_major_then_linear():
    torch.manual_seed(0)
    conn = DarknetConnected(in_features=2 * 3 * 4, out_features=5, activation="linear")
    x = torch.randn(1, 2, 3, 4)
    out = conn(x)
    ref = conn.linear(torch.flatten(x, 1))
    assert out.shape == (1, 5)
    assert torch.allclose(out, ref, atol=1e-6)


def test_local_layer_matches_naive_per_location_gemm():
    torch.manual_seed(0)
    c_in, n, k, oh, ow = 3, 4, 3, 5, 5
    local = DarknetLocal(c_in, n, k, stride=1, pad=k // 2, activation="linear",
                         out_h=oh, out_w=ow)
    with torch.no_grad():
        local.weight.copy_(torch.randn_like(local.weight))
        local.bias.copy_(torch.randn_like(local.bias))
    x = torch.randn(1, c_in, oh, ow)  # size3/stride1/pad1 -> same spatial size
    out = local(x)  # (1, n, oh, ow)

    cols = F.unfold(x, kernel_size=k, padding=k // 2, stride=1)  # (1, c*k*k, L)
    locations = oh * ow
    ref = torch.zeros(1, n, locations)
    for loc in range(locations):
        ref[0, :, loc] = local.weight[loc] @ cols[0, :, loc] + local.bias[:, loc]
    ref = ref.view(1, n, oh, ow)
    assert torch.allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# decode goldens (hand-computed synthetic detection tensors)
# ---------------------------------------------------------------------------
def _det_spec(side, num, nc, sqrt=True, softmax=False):
    return DetectionSpec(
        kind="detection", layer_index=0, anchors=[], num_classes=nc,
        stride=0, softmax=softmax, side=side, boxes_per_cell=num,
        coords=4, sqrt=sqrt,
    )


def test_decode_detection_single_cell_golden():
    # S=1, B=1, C=1 -> layout [class(1)][conf(1)][box(4)] length 6.
    spec = _det_spec(side=1, num=1, nc=1, sqrt=True)
    preds = torch.tensor([[0.8, 0.5, 0.5, 0.5, 0.4, 0.6]])  # cls, conf, x, y, w, h
    boxes, scores = decode_detection(preds, spec, input_size=100)
    # cx=(0.5+0)/1*100=50, cy=50, bw=0.4^2*100=16, bh=0.6^2*100=36
    assert torch.allclose(boxes[0], torch.tensor([50.0, 50.0, 16.0, 36.0]), atol=1e-4)
    # score = conf * class = 0.5 * 0.8 = 0.4
    assert scores[0, 0].item() == pytest.approx(0.4, abs=1e-5)


def test_decode_detection_no_sqrt_uses_raw_wh():
    spec = _det_spec(side=1, num=1, nc=1, sqrt=False)
    preds = torch.tensor([[1.0, 1.0, 0.5, 0.5, 0.4, 0.6]])
    boxes, _ = decode_detection(preds, spec, input_size=100)
    # raw w/h (no square): bw=0.4*100=40, bh=0.6*100=60
    assert torch.allclose(boxes[0], torch.tensor([50.0, 50.0, 40.0, 60.0]), atol=1e-4)


def test_decode_detection_grid_cell_offsets():
    # S=2, B=1, C=1. raw x=y=0.5 for all boxes -> centers at (0.5+col)/2*input.
    spec = _det_spec(side=2, num=1, nc=1, sqrt=False)
    cls = [0.0, 0.0, 0.0, 0.0]
    conf = [1.0, 1.0, 1.0, 1.0]
    box = []
    for _ in range(4):
        box += [0.5, 0.5, 0.0, 0.0]
    preds = torch.tensor([cls + conf + box])
    boxes, _ = decode_detection(preds, spec, input_size=200)
    centers = {(round(b[0].item()), round(b[1].item())) for b in boxes}
    # cell (row,col): cx=(0.5+col)*100, cy=(0.5+row)*100
    assert centers == {(50, 50), (150, 50), (50, 150), (150, 150)}


def test_decode_detection_softmax_over_classes():
    spec = _det_spec(side=1, num=1, nc=2, sqrt=False, softmax=True)
    # two equal class logits -> softmax 0.5 each; conf 1.0
    preds = torch.tensor([[1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0]])
    _, scores = decode_detection(preds, spec, input_size=100)
    assert torch.allclose(scores[0], torch.tensor([0.5, 0.5]), atol=1e-5)


def test_postprocess_recovers_single_detection():
    spec = _det_spec(side=1, num=1, nc=1, sqrt=False)
    preds = torch.tensor([[0.9, 0.9, 0.5, 0.5, 0.5, 0.5]])  # centered, half-size
    res = postprocess([preds], [spec], conf_thres=0.5, iou_thres=0.5,
                      input_size=100, original_size=(100, 100), ratio=1.0)
    assert res["num_detections"] == 1
    x1, y1, x2, y2 = res["boxes"][0]
    assert (x1 + x2) / 2 == pytest.approx(50.0, abs=1.0)


# ---------------------------------------------------------------------------
# family registry discrimination
# ---------------------------------------------------------------------------
def _prefixed_state_dict(cfg, family, nc):
    net = build_darknet(cfg, num_classes=nc).eval()
    return {f"{family}.{k}": v for k, v in net.state_dict().items()}


@pytest.mark.parametrize("cfg,size", [("yolov1", "b"), ("yolov1-tiny", "t")])
def test_yolo1_can_load_is_exclusive(cfg, size):
    sd = _prefixed_state_dict(cfg, "yolo1", nc=20)
    matches = [c for c in BaseModel._registry if c.can_load(sd)]
    assert matches == [LibreYOLO1]
    assert LibreYOLO1.detect_size(sd) == size
    assert LibreYOLO1.detect_nb_classes(sd) == 20


def test_yolo1_rejects_sibling_darknet_checkpoints():
    for cfg, family, cls in [
        ("yolov2-tiny", "yolo2", LibreYOLO2),
        ("yolov3-tiny", "yolo3", LibreYOLO3),
        ("yolov4-tiny", "yolo4", LibreYOLO4),
    ]:
        sd = _prefixed_state_dict(cfg, family, nc=80)
        assert not LibreYOLO1.can_load(sd)
    # and the siblings must reject a yolo1 checkpoint
    yolo1_sd = _prefixed_state_dict("yolov1-tiny", "yolo1", nc=20)
    for cls in (LibreYOLO2, LibreYOLO3, LibreYOLO4):
        assert not cls.can_load(yolo1_sd)


def test_yolo1_rejects_foreign_checkpoint():
    from libreyolo.models.yolox.nn import LibreYOLOXModel

    yolox_sd = LibreYOLOXModel(config="s", nb_classes=80).state_dict()
    assert not LibreYOLO1.can_load(yolox_sd)


def test_yolo1_default_is_voc_20_classes():
    model = LibreYOLO1(size="t", device="cpu")
    assert model.nb_classes == 20
    assert model.input_size == 448
    assert model.task == "detect"
    assert tuple(model.names[i] for i in range(20)) == VOC_NAMES


# ---------------------------------------------------------------------------
# stretch preprocessing (YOLOv1 needs square stretch, not letterbox)
# ---------------------------------------------------------------------------
def test_stretch_preprocess_fills_square_canvas():
    import numpy as np

    img = (np.random.rand(300, 500, 3) * 255).astype(np.uint8)  # non-square
    chw, ratio = preprocess_numpy_stretch(img, input_size=448)
    assert chw.shape == (3, 448, 448)  # whole image stretched, no padding band
    assert ratio == 1.0
    assert 0.0 <= float(chw.min()) and float(chw.max()) <= 1.0


# ---------------------------------------------------------------------------
# end-to-end: convert -> factory load -> predict (synthetic weights)
# ---------------------------------------------------------------------------
def _make_checkpoint(tmp_path, cfg, size):
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    net = build_darknet(cfg, num_classes=20).eval()
    blob = save_darknet_weights(net, major=0, minor=1)
    net2 = build_darknet(cfg, num_classes=20).eval()
    load_darknet_weights(net2, blob)
    sd = {f"yolo1.{k}": v for k, v in net2.state_dict().items()}
    names = {i: n for i, n in enumerate(VOC_NAMES)}
    ckpt = wrap_libreyolo_checkpoint(
        sd, model_family="yolo1", size=size, task="detect",
        nc=20, names=names, imgsz=448,
    )
    path = tmp_path / f"LibreYOLO1{size}.pt"
    torch.save(ckpt, path)
    return str(path)


def test_checkpoint_metadata_is_valid(tmp_path):
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    path = _make_checkpoint(tmp_path, "yolov1-tiny", "t")
    loaded = torch.load(path, weights_only=False)
    assert validate_checkpoint_metadata(loaded, strict=True) == []


def test_factory_load_and_predict_is_deterministic(tmp_path):
    import numpy as np
    from PIL import Image

    path = _make_checkpoint(tmp_path, "yolov1-tiny", "t")
    model = LibreYOLO(path)  # no size/nc hints -> must auto-detect
    assert type(model).__name__ == "LibreYOLO1"
    assert model.size == "t" and model.nb_classes == 20 and model.task == "detect"

    img = Image.fromarray((np.random.rand(360, 640, 3) * 255).astype(np.uint8))
    r1 = model.predict(img, conf=0.001)
    r2 = model.predict(img, conf=0.001)
    b1 = (r1[0] if isinstance(r1, list) else r1).boxes
    b2 = (r2[0] if isinstance(r2, list) else r2).boxes
    assert torch.equal(b1.xyxy, b2.xyxy)  # stable across runs (golden determinism)
    assert torch.equal(b1.conf, b2.conf)


def test_train_raises_not_implemented():
    model = LibreYOLO1(size="t", device="cpu")
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="voc.yaml")


# ---------------------------------------------------------------------------
# ONNX export (fixed 448; the FC head forbids dynamic shapes)
# ---------------------------------------------------------------------------
def test_export_wrapper_output_shape():
    from libreyolo.models.darknet.export import DarknetExportWrapper

    model = LibreYOLO1(size="t", device="cpu")
    model.model.eval()
    with torch.no_grad():
        out = DarknetExportWrapper(model.model)(torch.zeros(1, 3, 448, 448))
    # (B, 4+nc, N): N = S*S*B = 7*7*2 = 98; scores are raw conf*class (YOLOv1
    # does not sigmoid its head), so they are intentionally NOT bounded to [0, 1].
    assert out.shape == (1, 4 + 20, 98)


def test_onnx_graph_matches_native_wrapper(tmp_path):
    import numpy as np

    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from libreyolo.models.darknet.export import DarknetExportWrapper

    model = LibreYOLO1(size="t", device="cpu")
    model.model.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, 448, 448)
    with torch.no_grad():
        native = DarknetExportWrapper(model.model)(x).numpy()

    onnx_path = model.export(format="onnx", imgsz=448)
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        onnx_out = sess.run(None, {iname: x.numpy()})[0]
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

    assert onnx_out.shape == native.shape
    assert np.abs(native - onnx_out).max() < 1e-3


# ---------------------------------------------------------------------------
# gated real-weights faithfulness (the classic dog/bicycle/car golden)
# ---------------------------------------------------------------------------
_DOG = os.path.join(os.path.dirname(__file__), "..", "fixtures", "dog.jpg")


@pytest.mark.skipif(
    not os.environ.get("LIBREYOLO1B_CKPT"),
    reason="Set LIBREYOLO1B_CKPT to the converted LibreYOLO1b.pt to run the golden.",
)
def test_yolo1b_golden_dog_bicycle_car():
    """Real weights must reproduce YOLOv1's canonical dog.jpg detections."""
    model = LibreYOLO(os.environ["LIBREYOLO1B_CKPT"])
    assert type(model).__name__ == "LibreYOLO1" and model.size == "b"
    res = model.predict(_DOG, conf=0.2)
    result = res[0] if isinstance(res, list) else res
    labels = {model.names[int(c)] for c in result.boxes.cls}
    assert {"dog", "bicycle", "car"} <= labels
