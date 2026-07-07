"""YOLOv7 training smoke + overfit tests.

Covers the SimOTA training path added on top of the (previously inference-only)
v7 port: the loss dict, gradient flow, train/inference decode parity, the
overfit gate (the model can learn planted boxes), and that the loss never
pollutes the checkpoint.
"""

import pytest
import torch

from libreyolo.models.yolo7.net import YOLOv7Model
from libreyolo.models.yolo7.loss import YOLOv7Loss
from libreyolo.postprocess.yolo7 import decode_v7_head

pytestmark = pytest.mark.unit


def _fresh(nc=3):
    torch.manual_seed(0)
    m = YOLOv7Model(num_classes=nc)
    m.initialize_biases(0.01)
    return m


def _targets(nc=3, device="cpu"):
    # cxcywh pixels, well inside a 320px image.
    t = torch.zeros(2, 50, 5, device=device)
    t[0, 0] = torch.tensor([0.0, 200.0, 180.0, 120.0, 100.0], device=device)
    t[0, 1] = torch.tensor([float(nc - 1), 100.0, 120.0, 80.0, 90.0], device=device)
    t[1, 0] = torch.tensor([1.0, 220.0, 160.0, 120.0, 140.0], device=device)
    return t


def test_assignment_survives_degenerate_offgrid_box():
    """A GT whose centre lands off the feature grid must not crash SimOTA."""
    m = _fresh(nc=3).train()
    imgs = torch.rand(1, 3, 320, 320)
    t = torch.zeros(1, 10, 5)
    t[0, 0] = torch.tensor([0.0, 5000.0, 5000.0, 20.0, 20.0])  # far off-grid
    out = m(imgs, t)
    # The guard returns an empty assignment instead of crashing SimOTA's topk;
    # objectness BCE over the all-negative anchors keeps the loss finite and > 0.
    assert torch.isfinite(out["total_loss"]).all()
    assert out["total_loss"].item() > 0


def test_train_forward_returns_loss_dict_and_backprops():
    m = _fresh().train()
    imgs = torch.rand(2, 3, 320, 320)
    out = m(imgs, _targets())

    for k in ("total_loss", "iou_loss", "obj_loss", "cls_loss"):
        assert k in out, sorted(out)
        assert torch.isfinite(out[k]).all(), (k, out[k])
    assert out["total_loss"].item() > 0
    assert out["num_fg"] > 0  # SimOTA assigned at least one positive

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
    m = _fresh(nc=3).eval()
    imgs = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        heads = m(imgs)

    crit = YOLOv7Loss(3, m.anchors, m.strides)
    outputs, *_ = crit._decode(heads)  # [1, N, 4+1+nc], boxes cxcywh px
    train_boxes = outputs[0, :, :4]

    infer_boxes = []
    for h, flat, st in zip(heads, m.anchors, m.strides):
        pairs = list(zip(flat[0::2], flat[1::2]))
        b, _ = decode_v7_head(h, pairs, st, 3)
        infer_boxes.append(b)
    infer_boxes = torch.cat(infer_boxes, 0)

    assert train_boxes.shape == infer_boxes.shape
    assert torch.allclose(train_boxes, infer_boxes, atol=1e-4), \
        (train_boxes[:3], infer_boxes[:3])


def test_loss_never_enters_checkpoint():
    m = _fresh().train()
    # trigger lazy criterion creation
    _ = m(torch.rand(1, 3, 320, 320), _targets())
    keys = list(m.state_dict())
    assert not any(
        ("criterion" in k or "iou_loss" in k or "bce" in k) for k in keys
    ), [k for k in keys if "criterion" in k or "bce" in k]


def test_overfit_gate_loss_decreases():
    """The model must be able to learn: optimising fixed input/targets drives
    the loss down substantially within a handful of steps."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _fresh(nc=2).to(dev).train()
    img = torch.rand(1, 3, 160, 160, device=dev)
    t = torch.zeros(1, 10, 5, device=dev)
    t[0, 0] = torch.tensor([0.0, 80.0, 80.0, 50.0, 40.0], device=dev)
    t[0, 1] = torch.tensor([1.0, 120.0, 60.0, 30.0, 30.0], device=dev)

    opt = torch.optim.SGD(m.parameters(), lr=0.02, momentum=0.9)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        out = m(img, t)
        out["total_loss"].backward()
        opt.step()
        losses.append(out["total_loss"].item())

    # Robust across seeds/hardware: the min of the last few steps (momentum can
    # bounce the very last one) is well below the start.
    assert min(losses[-5:]) < 0.6 * losses[0], (losses[0], losses[-5:])
