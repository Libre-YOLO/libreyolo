"""Unit tests for RF-DETR GroupPose keypoint support.

These tests target the GroupPose keypoint architecture ported from RF-DETR
v1.8.0 (the clean-room ``keypoint_head`` MLP / ``_decode_keypoints`` /
``reinitialize_keypoint_head(int)`` path was removed). The model is built via
``libreyolo.models.rfdetr.nn._make_args`` + ``lwdetr.build_model`` directly so
we can exercise the GroupPose modules with random weights on CPU (no downloads).

A tiny size config keeps construction + forward fast while still building every
GroupPose module:
  * patch_size=12 / num_windows=2 / resolution=48 (divisible by 24) keeps the
    windowed DINOv2 backbone valid.
  * num_queries=13 == group_detr (the ``_make_args`` default). This is REQUIRED:
    the GroupPose encoder keypoint branch in ``transformer.py`` chunks the
    keypoint memory by ``len(enc_out_keypoint_embed)`` (== group_detr == 13)
    regardless of eval mode, and ``torch.chunk`` only yields 13 pieces when the
    query dimension is a multiple of 13. Smaller/odd query counts (e.g. 10, 14)
    crash with ``IndexError: tuple index out of range``. See the module-level
    note in the test report.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.rfdetr.lwdetr import build_model
from libreyolo.models.rfdetr.nn import RFDETRSizeConfig, _make_args
from libreyolo.models.rfdetr.tensors import NestedTensor

pytestmark = [pytest.mark.unit, pytest.mark.rfdetr]

# group_detr default from ``_make_args``; num_queries must be a multiple of it so
# ``torch.chunk(num_queries, group_detr)`` yields exactly group_detr pieces in the
# GroupPose encoder keypoint branch.
_GROUP_DETR = 13
_RES = 48  # divisible by patch_size(12) * num_windows(2) = 24
_PE = 4    # _RES // patch_size


def _small_pose_config() -> RFDETRSizeConfig:
    """Minimal-but-valid GroupPose-capable size config (CPU fast)."""
    return RFDETRSizeConfig(
        patch_size=12,
        num_windows=2,
        dec_layers=1,
        num_queries=_GROUP_DETR,
        num_select=_GROUP_DETR,
        projector_scale=("P4",),
        out_feature_indexes=(3, 6, 9, 12),
        resolution=_RES,
        positional_encoding_size=_PE,
    )


def _build_pose_model(num_keypoints_per_class=(0, 17), nb_classes=2):
    """Build a GroupPose keypoint LWDETR with random weights on CPU."""
    cfg = _small_pose_config()
    args = _make_args(
        cfg,
        pose=True,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=list(num_keypoints_per_class),
        dual_projector=True,
        dual_projector_kp_only=True,
        nb_classes=nb_classes,
        device="cpu",
        segmentation=False,
    )
    model = build_model(args)
    model.eval()
    return model


def _build_detect_model(nb_classes=3):
    """Build a plain detection LWDETR (no pose / keypoint flags)."""
    cfg = _small_pose_config()
    args = _make_args(
        cfg,
        pose=False,
        nb_classes=nb_classes,
        device="cpu",
        segmentation=False,
    )
    model = build_model(args)
    model.eval()
    return model


def _random_nested_tensor(batch=1):
    images = torch.rand(batch, 3, _RES, _RES)
    mask = torch.zeros(batch, _RES, _RES, dtype=torch.bool)
    return NestedTensor(images, mask)


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------
def test_grouppose_model_builds_with_expected_head_shapes():
    model = _build_pose_model(num_keypoints_per_class=(0, 17), nb_classes=2)

    # build_model passes num_classes = args.num_classes + 1 = (2 - 1) + 1 = 2.
    assert model.class_embed.out_features == 2
    assert tuple(model._kp_active_mask.shape) == (2, 17)
    # Compact GroupPose keypoint head: MLP(256, 256, 8, 3) -> last layer emits 8.
    assert model.keypoint_embed.layers[-1].out_features == 8
    assert model.use_grouppose_keypoints is True
    assert model.get_num_keypoints_per_class() == [0, 17]


# ---------------------------------------------------------------------------
# 2. Forward shapes
# ---------------------------------------------------------------------------
def test_grouppose_forward_emits_logits_boxes_keypoints():
    model = _build_pose_model(num_keypoints_per_class=(0, 17), nb_classes=2)
    samples = _random_nested_tensor(batch=1)

    with torch.no_grad():
        out = model(samples)

    q = _GROUP_DETR
    assert tuple(out["pred_logits"].shape) == (1, q, 2)
    assert tuple(out["pred_boxes"].shape) == (1, q, 4)
    # (B, Q, num_classes * max_K, KEYPOINT_PRED_DIM) = (1, 13, 2 * 17, 8) = (1, 13, 34, 8)
    assert tuple(out["pred_keypoints"].shape) == (1, q, 34, 8)


# ---------------------------------------------------------------------------
# 3. Keypoint decode sanity + no vestigial projector
# ---------------------------------------------------------------------------
def test_grouppose_keypoint_xy_finite_and_no_vestigial_keypoint_proj():
    model = _build_pose_model(num_keypoints_per_class=(0, 17), nb_classes=2)
    samples = _random_nested_tensor(batch=1)

    with torch.no_grad():
        out = model(samples)

    kp_xy = out["pred_keypoints"][..., :2]
    assert torch.isfinite(kp_xy).all()
    # xy are normalized box-relative coordinates; with zero-initialized keypoint
    # head deltas they sit near the box centers, comfortably within a plausible
    # normalized band.
    assert kp_xy.min() >= -1.0
    assert kp_xy.max() <= 2.0

    # The removed clean-room head exposed ``keypoint_head.keypoint_proj``; the
    # GroupPose architecture must not carry that vestigial module.
    module_names = {name for name, _ in model.named_modules()}
    assert not any("keypoint_proj" in name for name in module_names)
    assert not hasattr(model, "keypoint_head")


# ---------------------------------------------------------------------------
# 4. Schema reinit + round-trip
# ---------------------------------------------------------------------------
def test_grouppose_reinitialize_keypoint_head_updates_schema():
    model = _build_pose_model(num_keypoints_per_class=(0, 17), nb_classes=2)

    model.reinitialize_keypoint_head([0, 15])

    assert tuple(model._kp_active_mask.shape) == (2, 15)
    assert model.get_num_keypoints_per_class() == [0, 15]

    # keypoint_pos_embed rows track total active keypoints (sum == 15).
    decoder = model.transformer.decoder
    keypoint_pos_embed = getattr(decoder, "keypoint_pos_embed", None)
    assert keypoint_pos_embed is not None
    assert keypoint_pos_embed.shape[0] == 15

    # Schema round-trips through the checkpoint helpers.
    state_dict = model.state_dict()
    assert model.get_num_keypoints_per_class_from_checkpoint(state_dict) == [0, 15]


# ---------------------------------------------------------------------------
# 5. Checkpoint round-trip (strict)
# ---------------------------------------------------------------------------
def test_grouppose_state_dict_strict_roundtrip():
    source = _build_pose_model(num_keypoints_per_class=(0, 17), nb_classes=2)
    fresh = _build_pose_model(num_keypoints_per_class=(0, 17), nb_classes=2)

    result = fresh.load_state_dict(source.state_dict(), strict=True)

    assert list(result.missing_keys) == []
    assert list(result.unexpected_keys) == []


# ---------------------------------------------------------------------------
# 6. Postprocess decode
# ---------------------------------------------------------------------------
def test_grouppose_postprocess_emits_keypoints_with_normalized_visibility():
    from libreyolo.postprocess.rfdetr import postprocess

    num_queries, num_classes = 5, 2
    logits = torch.full((1, num_queries, num_classes), -5.0)
    logits[0, 0, 1] = 5.0  # query 0 -> person (class 1), high score
    logits[0, 1, 1] = 4.0  # query 1 -> person, slightly lower

    boxes = torch.rand(1, num_queries, 4) * 0.2 + 0.4  # cxcywh in [0, 1]

    # padded slots = num_kp_classes * max_K = 2 * 17 = 34, KEYPOINT_PRED_DIM = 8.
    keypoints = torch.zeros(1, num_queries, 34, 8)
    keypoints[..., 0] = 0.5  # x normalized
    keypoints[..., 1] = 0.5  # y normalized
    keypoints[..., 2] = 3.0  # findable logit -> sigmoid ~0.95

    result = postprocess(
        {"pred_logits": logits, "pred_boxes": boxes, "pred_keypoints": keypoints},
        torch.tensor([[100.0, 200.0]]),  # (height, width)
        num_select=4,
        num_keypoints_per_class=[0, 17],
        trace_alpha=0.0,
    )[0]

    keypoints_out = result["keypoints"]
    assert tuple(keypoints_out.shape) == (4, 17, 3)
    vis = keypoints_out[..., 2]
    assert float(vis.min()) >= 0.0
    assert float(vis.max()) <= 1.0
    # A person detection lands its keypoints at (x*width, y*height) = (100, 50).
    person_xy = keypoints_out[0, 0, :2]
    assert person_xy.tolist() == pytest.approx([100.0, 50.0])


# ---------------------------------------------------------------------------
# 7. Detection path unaffected
# ---------------------------------------------------------------------------
def test_detection_model_has_no_keypoint_modules():
    model = _build_detect_model(nb_classes=3)

    assert model.use_grouppose_keypoints is False
    assert getattr(model, "keypoint_embed", None) is None
    # Detection registers an empty (0, 0) active mask, never a populated one.
    assert model._kp_active_mask.numel() == 0
    module_names = {name for name, _ in model.named_modules()}
    assert not any("keypoint" in name for name in module_names)
