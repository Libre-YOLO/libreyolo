"""Unit tests for foundation-model (DINOv2) feature distillation into YOLO9.

These cover the feat_mse loss, the single-scale foundation path through the
Distiller (via ``teacher_feature_fn``), and the YOLO9 backbone distill config.
The real DINOv2 teacher needs a network download, so it is not exercised here;
these tests use a stub teacher feature function.
"""
import pytest
import torch
import torch.nn as nn

from libreyolo.distillation import Distiller, FeatureMSELoss, is_foundation_teacher
from libreyolo.distillation.teachers import DINOV2_ALIASES

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Teacher identification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ident,expected",
    [
        ("dinov2", True),
        ("dinov2_vitb14", True),
        ("facebook/dinov2-base", True),
        ("facebook/dinov2-large", True),
        ("yolov9c.pt", False),
        ("runs/train/exp/best.pt", False),
        ("", False),
        (None, False),
    ],
)
def test_is_foundation_teacher(ident, expected):
    assert is_foundation_teacher(ident) is expected


def test_aliases_all_point_to_facebook_dinov2():
    for hub_id in DINOV2_ALIASES.values():
        assert hub_id.startswith("facebook/dinov2")


# --------------------------------------------------------------------------- #
# FeatureMSELoss
# --------------------------------------------------------------------------- #
def test_feature_mse_channel_and_spatial_mismatch():
    """Align (96->768) + bilinear resize (16x16 teacher -> 4x4 student)."""
    loss_fn = FeatureMSELoss(student_channels=96, teacher_channels=768)
    assert loss_fn.align is not None
    s = torch.randn(2, 96, 4, 4, requires_grad=True)
    t = torch.randn(2, 768, 16, 16)
    out = loss_fn(s, t)
    assert out.ndim == 0 and torch.isfinite(out) and out.item() > 0
    out.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_feature_mse_no_align_when_channels_match():
    loss_fn = FeatureMSELoss(student_channels=256, teacher_channels=256)
    assert loss_fn.align is None
    s = torch.randn(1, 256, 8, 8)
    t = torch.randn(1, 256, 8, 8)
    assert torch.isfinite(loss_fn(s, t))


def test_feature_mse_normalize_option():
    loss_fn = FeatureMSELoss(64, 64, normalize=True)
    s = torch.randn(1, 64, 5, 5)
    t = torch.randn(1, 64, 5, 5)
    # normalized MSE is bounded (features on the unit sphere)
    assert 0.0 <= loss_fn(s, t).item() <= 4.0


# --------------------------------------------------------------------------- #
# Distiller foundation path (stub teacher, tiny student)
# --------------------------------------------------------------------------- #
class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.elan3 = nn.Conv2d(3, 96, 7, stride=16, padding=3)  # /16 stride

    def forward(self, x):
        return self.elan3(x)


class _TinyStudent(nn.Module):
    """Minimal stand-in with a ``backbone.elan3`` tap point."""

    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()

    def forward(self, x):
        return self.backbone(x)


def _make_foundation_distiller(hidden=768):
    student = _TinyStudent()
    teacher_cfg = {"tap_points": [], "channels": [hidden], "strides": [14]}
    student_cfg = {"tap_points": ["backbone.elan3"], "channels": [96], "strides": [16]}

    def stub_extract(images):
        n, _, h, w = images.shape
        return [torch.randn(n, hidden, h // 14, w // 14, device=images.device)]

    distiller = Distiller(
        teacher_model=nn.Identity(),
        student_model=student,
        teacher_config=teacher_cfg,
        student_config=student_cfg,
        loss_type="feat_mse",
        loss_weight=1.0,
        teacher_feature_fn=stub_extract,
    )
    return student, distiller


def test_foundation_distiller_builds_single_scale_feat_mse():
    _, distiller = _make_foundation_distiller()
    assert distiller.num_scales == 1
    assert distiller.t_hooks is None  # no teacher hooks in foundation mode
    assert isinstance(distiller.loss_modules[0], FeatureMSELoss)
    assert distiller.loss_modules[0].align.in_channels == 96
    assert distiller.loss_modules[0].align.out_channels == 768


def test_foundation_distiller_loss_flows_and_trains_adapter():
    student, distiller = _make_foundation_distiller()
    imgs = torch.rand(2, 3, 224, 224)
    distiller.teacher_forward(imgs)
    student(imgs)
    loss = distiller.compute_loss()
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    grad = distiller.loss_modules[0].align.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_foundation_distiller_step_clears_teacher_feats():
    student, distiller = _make_foundation_distiller()
    imgs = torch.rand(1, 3, 112, 112)
    distiller.teacher_forward(imgs)
    assert distiller._teacher_feats is not None
    student(imgs)
    distiller.compute_loss()
    distiller.step()
    assert distiller._teacher_feats is None


def test_compute_loss_without_teacher_forward_raises():
    student, distiller = _make_foundation_distiller()
    student(torch.rand(1, 3, 112, 112))
    with pytest.raises(RuntimeError, match="Teacher features not extracted"):
        distiller.compute_loss()


# --------------------------------------------------------------------------- #
# YOLO9 backbone distill config
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size,ch", [("t", 96), ("s", 192)])
def test_yolo9_backbone_distill_config(size, ch):
    from libreyolo.models.yolo9.model import LibreYOLO9

    model = LibreYOLO9(model_path=None, size=size)
    cfg = model.get_backbone_distill_config()
    assert cfg["tap_points"] == ["backbone.elan3"]
    assert cfg["strides"] == [16]
    assert cfg["channels"] == [ch]
