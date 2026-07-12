"""Train-path correctness tests for the RF-DETR GroupPose keypoint head.

These cover the *training* path fixes (classification-head label mapping +
detection-head preservation) layered on top of the GroupPose keypoint criterion
ported from RF-DETR v1.8.0. Inference parity is covered separately by
``tests/e2e/test_rfdetr_keypoint_parity.py``; here we exercise the synthetic
matcher -> loss -> backward step end to end.

The GroupPose pose head uses a 2-logit detection head where "person" is the
*internal* class 1 (schema ``num_keypoints_per_class=[0, 17]``). LibreYOLO's
person-only convention labels people as contiguous class 0, so the criterion /
matcher must lift label 0 -> internal class 1 for BOTH the keypoint terms and the
classification term (otherwise the classification loss supervises the empty
zero-keypoint column while the per-keypoint class-logit boost lands on column 1).

A tiny size config (mirrored from ``test_rfdetr_pose.py``) keeps the real
GroupPose modules buildable on CPU with no downloads. ``num_queries`` must be a
multiple of ``group_detr`` (== 13) so the encoder keypoint branch's
``torch.chunk`` yields exactly ``group_detr`` pieces.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.rfdetr.keypoints import (
    compute_keypoint_matching_cost,
    compute_l1_keypoint_loss,
    map_labels_to_keypoint_schema,
)
from libreyolo.models.rfdetr.loss import SetCriterion, _classic_keypoint_losses
from libreyolo.models.rfdetr.lwdetr import build_criterion_and_postprocessors, build_model
from libreyolo.models.rfdetr.matcher import (
    HungarianMatcher,
    _classic_keypoint_matching_cost,
)
from libreyolo.models.rfdetr.nn import RFDETRSizeConfig, _make_args
from libreyolo.models.rfdetr.tensors import NestedTensor

pytestmark = [pytest.mark.unit, pytest.mark.rfdetr]

_GROUP_DETR = 13
_RES = 48
_PE = 4
_SCHEMA = [0, 17]
_K_TOTAL = len(_SCHEMA) * max(_SCHEMA)  # 2 * 17 = 34


def _small_pose_config() -> RFDETRSizeConfig:
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


def _build_pose_args():
    return _make_args(
        _small_pose_config(),
        pose=True,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=list(_SCHEMA),
        dual_projector=True,
        dual_projector_kp_only=True,
        nb_classes=2,
        device="cpu",
        segmentation=False,
    )


def _person_targets():
    """One person-only instance (LibreYOLO contiguous label 0) per image."""
    target_keypoints = torch.rand(1, max(_SCHEMA), 3)
    target_keypoints[..., 2] = 2.0  # all keypoints fully visible
    return [
        {
            "labels": torch.tensor([0]),  # LibreYOLO contiguous "person"
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),  # cxcywh
            "keypoints": target_keypoints,
        }
    ]


# ---------------------------------------------------------------------------
# 1. Full synthetic train step: matcher -> loss -> backward on a real model
# ---------------------------------------------------------------------------
def test_grouppose_train_step_losses_finite_positive_and_grads_finite():
    torch.manual_seed(0)
    args = _build_pose_args()
    model = build_model(args)
    model.train()
    criterion, _ = build_criterion_and_postprocessors(args)
    criterion.train()

    # The criterion is built from args: 2-logit head preserved, keypoint losses on.
    assert criterion.use_grouppose_keypoints is True
    assert criterion.num_keypoints_per_class == _SCHEMA
    assert criterion.num_classes == 2  # args.num_classes(1) + 1
    assert model.class_embed.out_features == 2

    samples = NestedTensor(
        torch.rand(1, 3, _RES, _RES), torch.zeros(1, _RES, _RES, dtype=torch.bool)
    )
    outputs = model(samples)
    # In train mode the GroupPose decoder replicates queries by group_detr; only
    # the class column count (2) and keypoint slot count (2 * 17) are fixed.
    assert outputs["pred_logits"].shape[0] == 1 and outputs["pred_logits"].shape[-1] == 2
    assert tuple(outputs["pred_keypoints"].shape[-2:]) == (_K_TOTAL, 8)

    targets = _person_targets()
    losses = criterion(outputs, targets)

    # Classification loss AND the three keypoint terms must be finite + positive.
    for key in (
        "loss_ce",
        "loss_keypoints_l1",
        "loss_keypoints_findable",
        "loss_keypoints_visible",
    ):
        value = losses[key]
        assert torch.isfinite(value), f"{key} not finite: {value}"
        assert float(value) > 0.0, f"{key} not strictly positive: {value}"

    weight_dict = criterion.weight_dict
    total = sum(losses[k] * weight_dict.get(k, 1.0) for k in losses if "error" not in k)
    assert torch.isfinite(total)
    assert float(total) > 0.0

    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    for grad in grads:
        assert torch.isfinite(grad).all(), "non-finite gradient detected"


# ---------------------------------------------------------------------------
# 2. The matched person target supervises internal classification column 1
# ---------------------------------------------------------------------------
def test_grouppose_classification_supervises_internal_column_one():
    """A person (contiguous label 0) must supervise internal logit column 1.

    Column 1 is the keypoint-bearing class where the per-keypoint class-logit
    boost is added; column 0 is the empty zero-keypoint slot. The remap must move
    supervision onto column 1, exercised via the focal-loss path's gradient
    (positive class column receives a negative-pushing-up gradient, i.e. nonzero).
    """
    # Boundary helper: contiguous person 0 -> internal schema class 1.
    assert map_labels_to_keypoint_schema(torch.tensor([0]), _SCHEMA).tolist() == [1]

    torch.manual_seed(1)
    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        num_keypoints_per_class=_SCHEMA,
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
        losses=["labels"],
        group_detr=1,
        ia_bce_loss=False,  # plain focal path so the positive column is unambiguous
        use_grouppose_keypoints=True,
        num_keypoints_per_class=_SCHEMA,
    )

    num_queries = 13
    # Query 0's box matches the target (cxcywh [0.5,0.5,0.4,0.4]) so it wins the
    # match; the other queries sit far away. Logits stay neutral (0.0) so the
    # focal-loss gradient on the matched query cleanly separates the supervised
    # positive column (pushed up -> negative grad) from background columns
    # (pushed down -> positive grad).
    pred_logits = torch.zeros(1, num_queries, 2, requires_grad=True)
    pred_boxes = torch.full((1, num_queries, 4), 0.1)
    pred_boxes[0, 0] = torch.tensor([0.5, 0.5, 0.4, 0.4])
    pred_boxes = pred_boxes.detach().requires_grad_(True)

    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
    targets = _person_targets()

    losses = criterion(outputs, targets)
    loss_ce = losses["loss_ce"]
    assert torch.isfinite(loss_ce)
    assert float(loss_ce) > 0.0

    loss_ce.backward()
    grad = pred_logits.grad[0, 0]  # gradient at the matched query (query 0)
    # The positive (person) column is supervised toward 1 -> its logit gradient is
    # negative (push logit up). The empty zero-keypoint background column is
    # supervised toward 0 -> positive gradient (push down). Column 1 must be the
    # positive one; if the remap were missing, column 0 would be positive instead.
    assert grad[1] < 0.0, f"internal column 1 not supervised as positive: grad={grad.tolist()}"
    assert grad[0] > 0.0, f"internal column 0 not supervised as background: grad={grad.tolist()}"


# ---------------------------------------------------------------------------
# 3. Detection-head width preserved (2 logits) when the criterion is built
# ---------------------------------------------------------------------------
def test_grouppose_criterion_keeps_two_logit_head():
    args = _build_pose_args()
    criterion, _ = build_criterion_and_postprocessors(args)
    # SetCriterion num_classes == args.num_classes + 1 == (2 - 1) + 1 == 2, i.e.
    # the full 2-logit GroupPose head width, never collapsed to the person-only 1.
    assert criterion.num_classes == 2
    assert "loss_keypoints_l1" in criterion.weight_dict
    assert "keypoints" in criterion.losses


def test_grouppose_nonfinite_invisible_targets_keep_losses_and_costs_finite():
    pred = torch.zeros(1, _K_TOTAL, 8, requires_grad=True)
    target = torch.zeros(1, max(_SCHEMA), 3)
    target[..., :2] = 0.5
    target[..., 2] = 2.0
    target[0, 0] = torch.tensor([float("nan"), float("nan"), 0.0])
    target[0, 1] = torch.tensor([float("nan"), float("nan"), float("nan")])
    target_classes = torch.tensor([1])
    target_areas = torch.tensor([0.25])

    losses = compute_l1_keypoint_loss(
        pred, target, target_classes, target_areas, _SCHEMA
    )
    assert all(torch.isfinite(loss).all() for loss in losses)
    sum(loss.sum() for loss in losses).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

    costs = compute_keypoint_matching_cost(
        pred.detach().unsqueeze(0),
        target,
        target_classes,
        target_areas,
        _SCHEMA,
    )
    assert all(torch.isfinite(cost).all() for cost in costs)


def test_classic_pose_nonfinite_invisible_targets_keep_loss_and_grad_finite():
    pred = torch.zeros(1, 3, 3, requires_grad=True)
    target = torch.tensor(
        [[[float("nan"), float("nan"), 0.0], [0.5, 0.5, 2.0], [0.0, 0.0, float("nan")]]]
    )

    losses = _classic_keypoint_losses(pred, target)
    assert all(torch.isfinite(loss).all() for loss in losses)
    sum(loss.sum() for loss in losses).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

    costs = _classic_keypoint_matching_cost(pred.detach().unsqueeze(0), target)
    assert all(torch.isfinite(cost).all() for cost in costs)


def test_pose_validation_evaluates_and_restores_criterion_mode():
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    class DummyModel(torch.nn.Module):
        def forward(self, imgs, targets=None):
            return {"loss": imgs.sum() * 0.0}

    class RecordingCriterion(torch.nn.Module):
        weight_dict = {"loss": 1.0}

        def __init__(self):
            super().__init__()
            self.seen_modes = []

        def forward(self, outputs, targets):
            self.seen_modes.append(self.training)
            return {"loss": outputs["loss"]}

    trainer = object.__new__(RFDETRTrainer)
    trainer.wrapper_model = type("Wrapper", (), {"task": "pose"})()
    trainer.model = DummyModel().train()
    trainer.criterion = RecordingCriterion().train()
    trainer.ema_model = None
    trainer.val_loader = [(torch.zeros(1, 3, 8, 8), torch.zeros(1, 1, 1))]
    trainer.device = torch.device("cpu")
    trainer.is_distributed = False
    trainer._targets_to_rfdetr_list = lambda targets, height, width: []
    trainer._run_pose_metric_validation = (
        lambda model, epoch, save_plots=None: None
    )

    trainer._run_validation(0)

    assert trainer.criterion.seen_modes == [False]
    assert trainer.model.training is True
    assert trainer.criterion.training is True


def test_pose_validation_restores_modes_after_metric_failure():
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    trainer = object.__new__(RFDETRTrainer)
    trainer.wrapper_model = type("Wrapper", (), {"task": "pose"})()
    trainer.model = torch.nn.Linear(1, 1).eval()
    trainer.criterion = torch.nn.Linear(1, 1).train()
    trainer.criterion.weight_dict = {}
    trainer.ema_model = None
    trainer.val_loader = []
    trainer.device = torch.device("cpu")
    trainer.is_distributed = False

    def fail_validation(model, epoch, save_plots=None):
        raise RuntimeError("metric failure")

    trainer._run_pose_metric_validation = fail_validation

    with pytest.raises(RuntimeError, match="metric failure"):
        trainer._run_validation(0)

    assert trainer.model.training is False
    assert trainer.criterion.training is True
