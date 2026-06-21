"""Unit tests for RF-DETR pose keypoint support.

These tests cover both the GroupPose keypoint architecture ported from RF-DETR
v1.8.0 and the classic ``keypoint_head`` path used by public n/s/m/l pose
checkpoints. The model is built via
``libreyolo.models.rfdetr.nn._make_args`` + ``lwdetr.build_model`` directly so
we can exercise the keypoint modules with random weights on CPU (no downloads).

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


def _build_pose_model(num_keypoints_per_class=(0, 17), nb_classes=2) -> torch.nn.Module:
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


def _build_classic_pose_model(num_keypoints=17, nb_classes=1) -> torch.nn.Module:
    """Build a classic keypoint_head LWDETR with random weights on CPU."""
    cfg = _small_pose_config()
    args = _make_args(
        cfg,
        pose=True,
        use_grouppose_keypoints=False,
        num_keypoints=num_keypoints,
        nb_classes=nb_classes,
        device="cpu",
        segmentation=False,
    )
    model = build_model(args)
    model.eval()
    return model


def _build_detect_model(nb_classes=3) -> torch.nn.Module:
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


def test_rfdetr_pose_size_table_includes_public_checkpoints():
    from libreyolo.models.rfdetr.model import LibreRFDETR
    from libreyolo.models.rfdetr.nn import RFDETR_POSE_CONFIGS

    assert LibreRFDETR.POSE_INPUT_SIZES == {
        "n": 512,
        "s": 768,
        "m": 576,
        "l": 704,
        "x": 576,
    }
    assert set(RFDETR_POSE_CONFIGS) == {"n", "s", "m", "l", "x"}
    assert RFDETR_POSE_CONFIGS["x"].grouppose_head is True
    assert all(
        not RFDETR_POSE_CONFIGS[size].grouppose_head for size in ("n", "s", "m", "l")
    )


@pytest.mark.parametrize("size", ["n", "s", "m", "l", "x"])
def test_rfdetr_pose_download_url_for_public_sizes(size):
    from libreyolo.models.rfdetr.model import LibreRFDETR

    name = f"LibreRFDETR{size}-pose.pt"

    assert LibreRFDETR.get_download_url(name) == (
        f"https://huggingface.co/LibreYOLO/LibreRFDETR{size}-pose/"
        f"resolve/main/{name}"
    )


def _random_nested_tensor(batch: int = 1) -> NestedTensor:
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

    # GroupPose must not carry a classic keypoint_head module.
    module_names = {name for name, _ in model.named_modules()}
    assert not any("keypoint_proj" in name for name in module_names)
    assert getattr(model, "keypoint_head", None) is None


def test_classic_pose_model_forward_emits_decoded_keypoints():
    model = _build_classic_pose_model(num_keypoints=5, nb_classes=1)
    samples = _random_nested_tensor(batch=1)

    with torch.no_grad():
        out = model(samples)

    q = _GROUP_DETR
    assert tuple(out["pred_logits"].shape) == (1, q, 1)
    assert tuple(out["pred_boxes"].shape) == (1, q, 4)
    assert tuple(out["pred_keypoints"].shape) == (1, q, 5, 3)
    xy = out["pred_keypoints"][..., :2]
    assert torch.isfinite(xy).all()
    assert float(xy.min()) >= -1.0
    assert float(xy.max()) <= 2.0


def test_classic_pose_export_preserves_keypoint_batch_axis():
    from libreyolo.models.rfdetr.nn import RFDETRExportWrapper

    model = _build_classic_pose_model(num_keypoints=5, nb_classes=1)
    wrapper = RFDETRExportWrapper(model).eval()
    x = torch.rand(1, 3, _RES, _RES)

    with torch.no_grad():
        boxes, logits, keypoints = wrapper(x)

    q = _GROUP_DETR
    assert tuple(boxes.shape) == (1, q, 4)
    assert tuple(logits.shape) == (1, q, 1)
    assert tuple(keypoints.shape) == (1, q, 5, 3)


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
    logits[0, 0, 1] = 5.0  # query 0 -> person (internal class 1), high score
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
    labels_out = result["labels"]
    vis = keypoints_out[..., 2]
    assert float(vis.min()) >= 0.0
    assert float(vis.max()) <= 1.0

    # Class-index remap: the keypoint-bearing internal class 1 ("person") is
    # emitted as the LibreYOLO contiguous pose label 0, and detections whose
    # predicted class is NOT keypoint-bearing (internal class 0) are dropped.
    assert labels_out.numel() >= 2  # both explicit person queries survive
    assert torch.all(labels_out == 0)
    # The surviving detections (person) and their keypoints stay in lockstep.
    assert keypoints_out.shape[0] == labels_out.numel()
    assert tuple(keypoints_out.shape[1:]) == (17, 3)
    assert result["boxes"].shape[0] == labels_out.numel()
    assert result["scores"].shape[0] == labels_out.numel()
    # The top person detection lands its keypoints at (x*width, y*height) = (100, 50).
    person_xy = keypoints_out[0, 0, :2]
    assert person_xy.tolist() == pytest.approx([100.0, 50.0])


def test_classic_pose_postprocess_keeps_keypoints_without_grouppose_schema():
    from libreyolo.postprocess.rfdetr import postprocess

    logits = torch.tensor([[[5.0], [4.0]]])
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.1, 0.1]]])
    keypoints = torch.zeros(1, 2, 2, 3)
    keypoints[0, 0, :, 0] = torch.tensor([0.25, 0.75])
    keypoints[0, 0, :, 1] = torch.tensor([0.5, 0.25])
    keypoints[0, 0, :, 2] = torch.tensor([2.0, -2.0])

    result = postprocess(
        {"pred_logits": logits, "pred_boxes": boxes, "pred_keypoints": keypoints},
        torch.tensor([[100.0, 200.0]]),
        num_select=1,
        num_keypoints_per_class=None,
    )[0]

    assert result["labels"].tolist() == [0]
    assert tuple(result["keypoints"].shape) == (1, 2, 3)
    assert result["keypoints"][0, :, 0].tolist() == pytest.approx([50.0, 150.0])
    assert result["keypoints"][0, :, 1].tolist() == pytest.approx([50.0, 25.0])
    assert result["keypoints"][0, :, 2].tolist() == pytest.approx(
        [0.880797, 0.119203],
        rel=1e-5,
    )


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


def test_detection_checkpoint_without_kp_active_mask_loads(monkeypatch):
    import libreyolo.models.rfdetr.model as rfdetr_model
    from libreyolo.models.rfdetr.model import LibreRFDETR

    cfg = _small_pose_config()
    monkeypatch.setitem(rfdetr_model.RFDETR_CONFIGS, "tiny", cfg)
    monkeypatch.setitem(LibreRFDETR.INPUT_SIZES, "tiny", cfg.resolution)
    monkeypatch.setitem(LibreRFDETR.TASK_INPUT_SIZES["detect"], "tiny", cfg.resolution)

    source = _build_detect_model(nb_classes=3)
    state = dict(source.state_dict())
    assert "_kp_active_mask" in state
    state.pop("_kp_active_mask")

    model = LibreRFDETR(
        model_path={"model": state, "task": "detect", "size": "tiny", "nc": 3},
        size="tiny",
        task="detect",
        nb_classes=3,
        device="cpu",
    )

    assert model.task == "detect"
    assert model.model.model._kp_active_mask.numel() == 0


# ---------------------------------------------------------------------------
# 8. Class-index convention: postprocess remaps to contiguous 0
# ---------------------------------------------------------------------------
def test_grouppose_postprocess_remaps_keypoint_class_to_contiguous_zero():
    """The keypoint-bearing internal class (1 for ``[0, 17]``) is emitted as 0.

    The GroupPose schema places "person" at internal index 1, so the model emits
    detection label 1. LibreYOLO's person-only pose convention is contiguous
    index 0, so the postprocess must remap 1 -> 0 and drop non-keypoint-bearing
    detections (internal class 0), without disturbing keypoint xy/conf.
    """
    from libreyolo.postprocess.rfdetr import postprocess

    num_queries, num_classes = 4, 2
    # Make every selected detection a confident person (internal class 1) so we
    # can assert the full set survives and is relabeled.
    logits = torch.full((1, num_queries, num_classes), -10.0)
    logits[0, :, 1] = 6.0  # all queries -> person (internal class 1)

    boxes = torch.rand(1, num_queries, 4) * 0.2 + 0.4  # cxcywh in [0, 1]
    keypoints = torch.zeros(1, num_queries, 34, 8)
    keypoints[..., 0] = 0.25  # x normalized
    keypoints[..., 1] = 0.75  # y normalized
    keypoints[..., 2] = 2.0   # findable logit

    result = postprocess(
        {"pred_logits": logits, "pred_boxes": boxes, "pred_keypoints": keypoints},
        torch.tensor([[80.0, 120.0]]),  # (height, width)
        num_select=4,
        num_keypoints_per_class=[0, 17],
        trace_alpha=0.0,
    )[0]

    labels = result["labels"]
    # Every kept detection is the keypoint-bearing class, remapped to contiguous 0.
    assert labels.numel() == num_queries
    assert torch.all(labels == 0)
    # The internal class 1 never leaks through.
    assert int(labels.max()) == 0
    # Keypoint xy still scale to (x*width, y*height) = (30, 60); the remap only
    # touches the integer label, not the decoded keypoints.
    assert result["keypoints"][0, 0, :2].tolist() == pytest.approx([30.0, 60.0])


# ---------------------------------------------------------------------------
# 9. Class-index convention: person-only target trains a nonzero keypoint loss
# ---------------------------------------------------------------------------
def test_grouppose_person_only_target_trains_nonzero_keypoint_loss():
    """A LibreYOLO person target (label 0) must produce a finite, positive loss.

    Without the schema remap the contiguous label 0 selects the empty slot
    (``num_keypoints_per_class[0] == 0``) and every keypoint loss term collapses
    to zero, so the keypoint head never trains. The criterion-boundary remap
    lifts label 0 -> internal class 1 so the loss is finite and strictly > 0.
    """
    from libreyolo.models.rfdetr.keypoints import map_labels_to_keypoint_schema
    from libreyolo.models.rfdetr.loss import SetCriterion
    from libreyolo.models.rfdetr.matcher import HungarianMatcher

    torch.manual_seed(0)
    schema = [0, 17]
    k_total = len(schema) * max(schema)  # 2 * 17 = 34

    # The remap helper itself lifts the contiguous person label to schema class 1.
    assert map_labels_to_keypoint_schema(torch.tensor([0]), schema).tolist() == [1]

    matcher = HungarianMatcher(
        num_keypoints_per_class=schema,
        keypoint_l1_loss_coef=1.0,
        keypoint_findable_loss_coef=1.0,
        keypoint_visible_loss_coef=1.0,
        keypoint_nll_loss_coef=1.0,
    )
    criterion = SetCriterion(
        num_classes=2,
        matcher=matcher,
        weight_dict={},
        focal_alpha=0.25,
        losses=["keypoints"],
        group_detr=1,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=schema,
    )

    num_queries = 5
    outputs = {"pred_keypoints": torch.randn(1, num_queries, k_total, 8)}
    target_keypoints = torch.rand(1, max(schema), 3)
    target_keypoints[..., 2] = 2.0  # all keypoints fully visible
    targets = [
        {
            "labels": torch.tensor([0]),  # LibreYOLO contiguous "person"
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),  # cxcywh
            "keypoints": target_keypoints,
        }
    ]
    indices = [(torch.tensor([0]), torch.tensor([0]))]  # query 0 <-> target 0

    losses = criterion.loss_keypoints(outputs, targets, indices, num_boxes=1.0)

    for value in losses.values():
        assert torch.isfinite(value)
    # The location/findable/visible terms are strictly positive once the person
    # target reaches the populated schema slot.
    assert float(losses["loss_keypoints_l1"]) > 0.0
    assert float(losses["loss_keypoints_findable"]) > 0.0
    assert float(losses["loss_keypoints_visible"]) > 0.0
    total = sum(losses.values())
    assert torch.isfinite(total)
    assert float(total) > 0.0
