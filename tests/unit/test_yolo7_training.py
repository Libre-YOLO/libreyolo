"""YOLOv7 training smoke + overfit tests.

Covers the SimOTA training path added on top of the (previously inference-only)
v7 port: the loss dict, gradient flow, train/inference decode parity, the fp32
AMP-safety path, the trainer's label-format conversion, the overfit smoke (the
model can learn), and that the loss never pollutes the checkpoint.

Loss-only tests feed synthetic head maps directly to :class:`YOLOv7Loss`
(building the full 37M-param net just to produce tensors of a known shape is
CI waste); only the tests that genuinely exercise the network build it.
"""

import pytest
import torch

from libreyolo.models.yolo7.net import YOLOv7Model
from libreyolo.models.yolo7.loss import YOLOv7Loss
from libreyolo.models.yolo7.trainer import YOLOv7Trainer
from libreyolo.postprocess.yolo7 import decode_v7_head

pytestmark = pytest.mark.unit

# v7.yaml anchors/strides (pixel units), used for loss-only tests without a net.
_ANCHORS = [[12, 16, 19, 36, 40, 28],
            [36, 75, 76, 55, 72, 146],
            [142, 110, 192, 243, 459, 401]]
_STRIDES = [8, 16, 32]


def _fresh(nc=3):
    torch.manual_seed(0)
    m = YOLOv7Model(num_classes=nc)
    m.initialize_biases(0.01)
    return m


def _synthetic_maps(nc=3, imgsz=320, batch=2, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    return [
        torch.randn(batch, 3 * (5 + nc), imgsz // s, imgsz // s,
                    dtype=dtype, requires_grad=True)
        for s in _STRIDES
    ]


def _targets(nc=3, device="cpu"):
    # cxcywh pixels, well inside a 320px image.
    t = torch.zeros(2, 50, 5, device=device)
    t[0, 0] = torch.tensor([0.0, 200.0, 180.0, 120.0, 100.0], device=device)
    t[0, 1] = torch.tensor([float(nc - 1), 100.0, 120.0, 80.0, 90.0], device=device)
    t[1, 0] = torch.tensor([1.0, 220.0, 160.0, 120.0, 140.0], device=device)
    return t


def test_train_forward_returns_loss_dict_and_backprops():
    m = _fresh().train()
    imgs = torch.rand(2, 3, 320, 320)
    out = m(imgs, _targets())

    for k in ("total_loss", "iou_loss", "obj_loss", "cls_loss"):
        assert k in out, sorted(out)
        assert torch.isfinite(out[k]).all(), (k, out[k])
    assert out["total_loss"].item() > 0
    assert out["num_fg"] > 0  # SimOTA assigned at least one positive (raw count)

    out["total_loss"].backward()
    nonzero = sum(
        1 for p in m.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    assert nonzero > 0, "no parameter received a gradient"


def test_inference_path_unchanged_when_no_targets():
    m = _fresh().eval()
    imgs = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        heads = m(imgs)
    assert isinstance(heads, list) and len(heads) == 3
    assert [h.shape[-1] for h in heads] == [40, 20, 10]


def test_train_and_inference_decode_agree():
    """The loss's internal box decode must match the inference decoder, else
    training optimises boxes the deployed model never produces."""
    heads = [m.detach()[:1] for m in _synthetic_maps(nc=3)]
    crit = YOLOv7Loss(3, _ANCHORS, _STRIDES)
    outputs, *_ = crit._decode(heads)  # [1, N, 4+1+nc], boxes cxcywh px
    train_boxes = outputs[0, :, :4]

    infer_boxes = []
    for h, flat, st in zip(heads, _ANCHORS, _STRIDES):
        pairs = list(zip(flat[0::2], flat[1::2]))
        b, _ = decode_v7_head(h, pairs, st, 3)
        infer_boxes.append(b)
    infer_boxes = torch.cat(infer_boxes, 0)

    assert train_boxes.shape == infer_boxes.shape
    assert torch.allclose(train_boxes, infer_boxes, atol=1e-4), \
        (train_boxes[:3], infer_boxes[:3])


def test_loss_is_fp32_safe_under_half_inputs():
    """Regression: v7's pixel-scale anchors overflow fp16 box areas (max 65504),
    which made large-box IoUs NaN under AMP. The loss must upcast internally."""
    maps = _synthetic_maps(nc=3, dtype=torch.float16, batch=1)
    t = torch.zeros(1, 10, 5)
    t[0, 0] = torch.tensor([0.0, 160.0, 160.0, 300.0, 280.0])  # large box
    crit = YOLOv7Loss(3, _ANCHORS, _STRIDES)
    out = crit(maps, t)
    for k in ("total_loss", "iou_loss", "obj_loss", "cls_loss"):
        assert torch.isfinite(out[k]).all(), (k, out[k])
    out["total_loss"].backward()
    assert all(torch.isfinite(mp.grad).all() for mp in maps), "NaN grads under fp16"


def test_trainer_converts_normalized_xyxy_labels():
    """YOLO9TrainTransform emits [cls, x1,y1,x2,y2] normalized; on_forward must
    hand the loss [cls, cx,cy,w,h] in pixels (padded rows stay zero)."""
    tr = YOLOv7Trainer.__new__(YOLOv7Trainer)
    seen = {}

    class _Probe:
        def __call__(self, imgs, targets):
            seen["targets"] = targets
            return {"total_loss": torch.tensor(0.0)}

    tr.model = _Probe()
    imgs = torch.zeros(2, 3, 320, 320)
    tgt = torch.zeros(2, 4, 5)
    tgt[0, 0] = torch.tensor([2.0, 0.25, 0.25, 0.75, 0.75])
    tr.on_forward(imgs, tgt)
    converted = seen["targets"]
    assert torch.allclose(converted[0, 0],
                          torch.tensor([2.0, 160.0, 160.0, 160.0, 160.0]))
    assert (converted[0, 1:] == 0).all() and (converted[1] == 0).all()


def test_loss_never_enters_checkpoint():
    m = _fresh().train()
    # trigger lazy criterion creation
    _ = m(torch.rand(1, 3, 320, 320), _targets())
    keys = list(m.state_dict())
    assert not any(
        ("criterion" in k or "iou_loss" in k or "bce" in k) for k in keys
    ), [k for k in keys if "criterion" in k or "bce" in k]


def test_initialize_biases_is_idempotent():
    """Set-semantics (not +=): repeated train() on the same object must not
    compound the focal prior into the head biases."""
    m = _fresh(nc=2)
    head = m.layers[m._head_index].heads[0]
    once = head.head_conv.bias.detach().clone()
    m.initialize_biases(0.01)
    assert torch.equal(once, head.head_conv.bias.detach())


def test_assignment_survives_degenerate_offgrid_box():
    """A GT whose centre lands off the feature grid must not crash SimOTA."""
    maps = [mp.detach() for mp in _synthetic_maps(nc=3, batch=1)]
    t = torch.zeros(1, 10, 5)
    t[0, 0] = torch.tensor([0.0, 5000.0, 5000.0, 20.0, 20.0])  # far off-grid
    crit = YOLOv7Loss(3, _ANCHORS, _STRIDES)
    out = crit(maps, t)
    assert torch.isfinite(out["total_loss"]).all()
    assert out["total_loss"].item() > 0  # objectness BCE over all-negatives
    assert out["num_fg"] == 0.0  # raw match count: nothing assigned, no crash


def test_overfit_smoke_loss_decreases():
    """Learning-signal smoke: optimising fixed input/targets drives the loss
    down within a few steps. Pinned to CPU so CI and dev machines run the same
    numerics; the full convergence proof lives at the e2e tier."""
    torch.manual_seed(0)
    m = YOLOv7Model(num_classes=2)
    m.initialize_biases(0.01)
    m.train()
    img = torch.rand(1, 3, 128, 128)
    t = torch.zeros(1, 10, 5)
    t[0, 0] = torch.tensor([0.0, 64.0, 64.0, 40.0, 30.0])
    t[0, 1] = torch.tensor([1.0, 96.0, 48.0, 24.0, 24.0])

    opt = torch.optim.SGD(m.parameters(), lr=0.02, momentum=0.9)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        out = m(img, t)
        out["total_loss"].backward()
        opt.step()
        losses.append(out["total_loss"].item())

    # Momentum can bounce single steps; the min of the tail must clearly beat
    # the start.
    assert min(losses[-5:]) < 0.85 * losses[0], (losses[0], losses[-5:])
