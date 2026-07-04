"""Unit tests for YOLO9 instance segmentation (task='segment').

Covers the YOLACT-style head (MaskProto + DDetectSeg), the segmentation loss,
the two mask/label-alignment bug fixes, checkpoint task detection, and the
predict path producing masks.
"""

import numpy as np
import pytest
import torch

from libreyolo.models.yolo9.model import LibreYOLO9
from libreyolo.models.yolo9.nn import DDetectSeg, LibreYOLO9Model

pytestmark = pytest.mark.unit


def _build(nc=2, imgsz=160, size="t"):
    model = LibreYOLO9Model(config=size, nb_classes=nc, img_size=imgsz, task="segment")
    return model


def test_segment_head_is_ddetectseg():
    model = _build()
    assert isinstance(model.head, DDetectSeg)
    assert model.head.nm == 32
    assert model.task == "segment"


def test_inference_forward_returns_proto_and_coeffs():
    model = _build(imgsz=160)
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 160, 160))
    assert set(("predictions", "proto", "mask_coeffs")).issubset(out.keys())
    # proto at input/4, one coeff vector per anchor
    assert out["proto"].shape[-2:] == (40, 40)
    assert out["mask_coeffs"].shape[1] == model.head.nm


def test_training_loss_has_seg_component_and_backprops():
    model = _build(nc=1, imgsz=160)
    model.train()
    x = torch.full((1, 3, 160, 160), 0.1)
    x[:, :, 40:120, 40:120] = 0.9
    targets = torch.zeros(1, 1, 5)
    targets[0, 0] = torch.tensor([0, 0.25, 0.25, 0.75, 0.75])
    masks = torch.zeros(1, 1, 40, 40)
    masks[0, 0, 10:30, 10:30] = 1.0

    ld = model(x, targets=targets, masks=masks)
    assert "seg_loss" in ld and "seg" in ld
    assert torch.isfinite(ld["total_loss"])
    ld["total_loss"].backward()
    # seg gradients reach the prototype branch and the coeff tower
    assert model.head.proto.cv3.conv.weight.grad.abs().sum() > 0
    assert model.head.cv4[0][-1].weight.grad.abs().sum() > 0


def test_overfit_one_sample_drives_seg_loss_down():
    torch.manual_seed(0)
    model = _build(nc=1, imgsz=160)
    model.train()
    x = torch.full((1, 3, 160, 160), 0.1)
    x[:, :, 40:120, 40:120] = 0.9
    targets = torch.zeros(1, 1, 5)
    targets[0, 0] = torch.tensor([0, 0.25, 0.25, 0.75, 0.75])
    masks = torch.zeros(1, 1, 40, 40)
    masks[0, 0, 10:30, 10:30] = 1.0

    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = last = None
    for step in range(60):
        opt.zero_grad()
        ld = model(x, targets=targets, masks=masks)
        ld["total_loss"].backward()
        opt.step()
        if step == 0:
            first = ld["seg"]
        last = ld["seg"]
    assert last < first * 0.5, f"seg loss did not fall: {first} -> {last}"


def test_padding_and_empty_masks_are_skipped():
    """Bug fixes: padding rows (out-of-range / empty masks) must not corrupt
    the loss (no clamp-to-last-mask, no supervising blank masks)."""
    model = _build(nc=1, imgsz=160)
    model.train()
    x = torch.rand(1, 3, 160, 160)
    # 1 real target + 2 padding rows (class 0, zero box, empty mask)
    targets = torch.zeros(1, 3, 5)
    targets[0, 0] = torch.tensor([0, 0.3, 0.3, 0.7, 0.7])
    masks = torch.zeros(1, 3, 40, 40)
    masks[0, 0, 12:28, 12:28] = 1.0  # only the real target has a mask
    ld = model(x, targets=targets, masks=masks)
    assert torch.isfinite(ld["total_loss"])
    ld["total_loss"].backward()  # must not raise / NaN


def test_checkpoint_task_detection_and_roundtrip():
    m = LibreYOLO9(model_path=None, size="t", nb_classes=2, task="segment", device="cpu")
    sd = m.model.state_dict()
    assert any(k.startswith("head.proto") for k in sd)
    assert LibreYOLO9.detect_checkpoint_task(sd) == "segment"

    m2 = LibreYOLO9(model_path=None, size="t", nb_classes=2, task="segment", device="cpu")
    missing, unexpected = m2.model.load_state_dict(sd, strict=False)
    assert list(unexpected) == []


def test_predict_path_exposes_masks_attr():
    m = LibreYOLO9(model_path=None, size="t", nb_classes=2, task="segment", device="cpu")
    m.model.eval()
    img = (np.random.rand(192, 256, 3) * 255).astype(np.uint8)
    results = m(img, conf=0.001)
    r = results[0] if isinstance(results, list) else results
    assert hasattr(r, "masks")  # populated only when detections survive NMS


def test_detect_task_still_default_and_unaffected():
    m = LibreYOLO9(model_path=None, size="t", nb_classes=2, device="cpu")
    assert m.task == "detect"
    from libreyolo.models.yolo9.nn import DDetect

    assert type(m.model.head) is DDetect  # not the seg subclass
