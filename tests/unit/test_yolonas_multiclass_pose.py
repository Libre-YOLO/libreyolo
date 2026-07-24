"""Multi-class (shared-skeleton) keypoint support for YOLO-NAS pose (issue #484).

The user's meter-reading case needs one keypoint per object while classifying the
object into several types. YOLO-NAS pose gains ``num_classes`` on the class/box
branch; the single shared keypoint skeleton is unchanged. Single-class pose stays
byte-identical. These tests cover the nc > 1 additions:

- head width and the train/eval output shapes
- ``replace_num_classes`` / ``replace_num_keypoints`` rebuild round-trips
- the loss + assigner routing score mass to the correct class column
- shape-based class-count detection (multi-class and legacy single-class)
- ``postprocess_pose`` (argmax + per-class NMS; single-class parity)
- the ONNX backend parser
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from libreyolo.backends.base import BaseBackend
from libreyolo.models.yolonas.loss import YoloNASPoseLoss
from libreyolo.models.yolonas.model import LibreYOLONAS
from libreyolo.models.yolonas.nn import LibreYOLONASPoseModel
from libreyolo.postprocess.yolonas import postprocess_pose

pytestmark = [pytest.mark.unit, pytest.mark.yolonas]


def _decoded(out):
    """Head forward returns (decoded, raw) unless tracing; unwrap the decoded 4-tuple."""
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], tuple):
        return out[0]
    return out


# ---------------------------------------------------------------------------
# Head width + rebuild round-trips
# ---------------------------------------------------------------------------


class TestMultiClassPoseHead:
    def test_single_class_default_unchanged(self):
        m = LibreYOLONASPoseModel(config="n", num_keypoints=17)
        assert m.num_classes == 1
        # fused cls head: num_classes (1) + K visibility channels
        assert m.heads.head1.cls_pred.out_channels == 1 + 17

    def test_multiclass_head_widths(self):
        m = LibreYOLONASPoseModel(config="n", num_keypoints=1, num_classes=3)
        assert m.heads.head1.cls_pred.out_channels == 3 + 1
        assert m.heads.head2.cls_pred.out_channels == 3 + 1
        assert m.heads.head3.cls_pred.out_channels == 3 + 1

    def test_eval_output_shapes(self):
        m = LibreYOLONASPoseModel(config="n", num_keypoints=1, num_classes=3).eval()
        with torch.no_grad():
            bboxes, scores, pose_xy, pose_scores = _decoded(m(torch.rand(2, 3, 320, 320)))
        assert bboxes.shape[0] == 2 and bboxes.shape[-1] == 4
        assert scores.shape[-1] == 3
        assert pose_xy.shape[-2:] == (1, 2)  # [B, A, K=1, 2]
        assert pose_scores.shape[-1] == 1

    def test_train_raw_shapes(self):
        m = LibreYOLONASPoseModel(config="n", num_keypoints=1, num_classes=3).train()
        _, raw = m(torch.rand(2, 3, 320, 320))
        cls_logits, _reg, pose_coords, pose_logits = raw[0], raw[1], raw[2], raw[3]
        assert cls_logits.shape[-1] == 3
        assert pose_coords.shape[-2:] == (1, 2)
        assert pose_logits.shape[-1] == 1

    def test_replace_num_classes_preserves_visibility(self):
        m = LibreYOLONASPoseModel(config="n", num_keypoints=2, num_classes=1)
        vis_before = m.heads.head1.cls_pred.weight[1:].clone()  # K=2 visibility rows
        m.replace_num_classes(5)
        assert m.num_classes == 5
        assert m.heads.head1.cls_pred.out_channels == 5 + 2
        vis_after = m.heads.head1.cls_pred.weight[5:].clone()
        assert torch.allclose(vis_before, vis_after)

    def test_keypoints_then_classes_rebuild(self):
        # COCO-style start (nc=1, K=17), fine-tune to a 1-keypoint, 4-class dataset.
        m = LibreYOLONASPoseModel(config="n", num_keypoints=17, num_classes=1)
        m.replace_num_keypoints(1)
        vis_before = m.heads.head1.cls_pred.weight[1:].clone()  # freshly sized vis
        m.replace_num_classes(4)
        assert m.heads.head1.cls_pred.out_channels == 4 + 1
        vis_after = m.heads.head1.cls_pred.weight[4:].clone()
        assert torch.allclose(vis_before, vis_after)
        m.eval()
        with torch.no_grad():
            _, scores, pose_xy, _ = _decoded(m(torch.rand(1, 3, 320, 320)))
        assert scores.shape[-1] == 4
        assert pose_xy.shape[-2:] == (1, 2)


# ---------------------------------------------------------------------------
# Loss + assigner class routing
# ---------------------------------------------------------------------------


class TestMultiClassPoseLoss:
    def test_loss_finite_and_assigner_routes_classes(self):
        torch.manual_seed(0)
        nc, K = 3, 1
        model = LibreYOLONASPoseModel(config="n", num_keypoints=K, num_classes=nc).train()
        loss_fn = YoloNASPoseLoss(oks_sigmas=[0.1] * K, num_classes=nc)

        targets = torch.zeros(2, 2, 5 + 3 * K)
        targets[0, 0] = torch.tensor([2, 160, 160, 80, 80, 160, 160, 2])  # class 2
        targets[1, 0] = torch.tensor([0, 100, 100, 60, 60, 100, 100, 2])  # class 0
        targets[1, 1] = torch.tensor([1, 220, 220, 60, 60, 220, 220, 2])  # class 1

        out = model(torch.rand(2, 3, 320, 320))
        loss, logs = loss_fn(out, targets)
        assert torch.isfinite(loss)
        loss.backward()  # gradients flow

        # The assigner one-hot must land in the gt's class column.
        tgt = loss_fn._unpack_padded_targets(targets)
        assert tgt["gt_class"][0, 0, 0].item() == 2
        _, preds = out
        (pred_scores, pred_distri, pred_pose_coords, _pl, _a, anchor_points,
         _nal, stride_tensor) = preds
        pred_bboxes, _ = loss_fn._bbox_decode(anchor_points / stride_tensor, pred_distri)
        ar = loss_fn.assigner(
            pred_scores=pred_scores.detach().sigmoid(),
            pred_bboxes=pred_bboxes.detach() * stride_tensor,
            pred_pose_coords=pred_pose_coords.detach(),
            anchor_points=anchor_points,
            gt_labels=tgt["gt_class"], gt_bboxes=tgt["gt_bbox"],
            gt_poses=tgt["gt_poses"], gt_crowd=tgt["gt_crowd"],
            pad_gt_mask=tgt["pad_gt_mask"], bg_index=nc,
        )
        sc = ar.assigned_scores  # [B, A, nc]
        pos0 = sc[0].sum(dim=-1) > 0
        assert sc[0][pos0].argmax(dim=-1).unique().tolist() == [2]
        pos1 = sc[1].sum(dim=-1) > 0
        assert sorted(sc[1][pos1].argmax(dim=-1).unique().tolist()) == [0, 1]

    def test_single_class_default(self):
        loss_fn = YoloNASPoseLoss(oks_sigmas=[0.1])
        assert loss_fn.num_classes == 1


# ---------------------------------------------------------------------------
# Shape-based class-count detection (backward compatible)
# ---------------------------------------------------------------------------


class TestShapeBasedClassDetection:
    @staticmethod
    def _state_dict(nc, K):
        return {
            "heads.head1.cls_pred.weight": torch.zeros(nc + K, 8, 1, 1),
            "heads.head1.pose_pred.weight": torch.zeros(2 * K, 8, 1, 1),
        }

    def test_legacy_single_class_checkpoint(self):
        sd = self._state_dict(nc=1, K=17)  # cls_pred out = 18
        assert LibreYOLONAS.is_pose_state_dict(sd)
        assert LibreYOLONAS.detect_num_keypoints(sd) == 17
        assert LibreYOLONAS.detect_nb_classes(sd) == 1

    def test_multiclass_checkpoint(self):
        sd = self._state_dict(nc=17, K=1)  # cls_pred out = 18, pose_pred out = 2
        assert LibreYOLONAS.detect_num_keypoints(sd) == 1
        assert LibreYOLONAS.detect_nb_classes(sd) == 17


# ---------------------------------------------------------------------------
# postprocess_pose
# ---------------------------------------------------------------------------


def _pose_output(scores):
    """Build a head-style output dict with two well-separated boxes."""
    a = scores.shape[0]
    boxes = torch.zeros(a, 4)
    boxes[0] = torch.tensor([10.0, 10.0, 40.0, 40.0])
    if a > 1:
        boxes[1] = torch.tensor([200.0, 200.0, 240.0, 240.0])
    pose_xy = torch.zeros(a, 1, 2)
    pose_conf = torch.ones(a, 1)
    return {"boxes": boxes, "scores": scores, "keypoints_xy": pose_xy, "keypoints_conf": pose_conf}


class TestPostprocessPose:
    def test_single_class_classes_all_zero(self):
        scores = torch.tensor([[0.9], [0.8]])  # [A, 1]
        res = postprocess_pose(_pose_output(scores), conf_thres=0.5)
        assert res["num_detections"] == 2
        assert res["classes"].tolist() == [0, 0]

    def test_multiclass_argmax_classes(self):
        scores = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.1, 0.1]])  # [A, 3]
        res = postprocess_pose(_pose_output(scores), conf_thres=0.5)
        assert res["num_detections"] == 2
        # box 0 -> class 1 (0.9), box 1 -> class 0 (0.8); ordered by score
        assert sorted(res["classes"].tolist()) == [0, 1]

    def test_per_class_nms_keeps_overlapping_different_classes(self):
        # Two identical boxes, different top class -> both survive under batched_nms.
        out = {
            "boxes": torch.tensor([[10.0, 10, 40, 40], [10.0, 10, 40, 40]]),
            "scores": torch.tensor([[0.1, 0.9], [0.8, 0.1]]),
            "keypoints_xy": torch.zeros(2, 1, 2),
            "keypoints_conf": torch.ones(2, 1),
        }
        res = postprocess_pose(out, conf_thres=0.5, iou_thres=0.5)
        assert res["num_detections"] == 2
        assert sorted(res["classes"].tolist()) == [0, 1]

    def test_per_class_nms_suppresses_same_class(self):
        out = {
            "boxes": torch.tensor([[10.0, 10, 40, 40], [11.0, 11, 41, 41]]),
            "scores": torch.tensor([[0.1, 0.9], [0.1, 0.8]]),  # both top class 1
            "keypoints_xy": torch.zeros(2, 1, 2),
            "keypoints_conf": torch.ones(2, 1),
        }
        res = postprocess_pose(out, conf_thres=0.5, iou_thres=0.5)
        assert res["num_detections"] == 1
        assert res["classes"].tolist() == [1]


# ---------------------------------------------------------------------------
# ONNX backend parser
# ---------------------------------------------------------------------------


class TestBackendParser:
    def test_parse_multiclass_scores(self):
        a = 3
        boxes = np.zeros((1, a, 4), np.float32)
        boxes[0, 0] = [10, 10, 40, 40]
        boxes[0, 1] = [200, 200, 240, 240]
        boxes[0, 2] = [50, 50, 60, 60]
        scores = np.zeros((1, a, 3), np.float32)
        scores[0, 0] = [0.1, 0.9, 0.2]   # class 1
        scores[0, 1] = [0.8, 0.1, 0.1]   # class 0
        scores[0, 2] = [0.0, 0.0, 0.0]   # below conf
        kpxy = np.zeros((1, a, 1, 2), np.float32)
        kpconf = np.ones((1, a, 1), np.float32)

        dummy = types.SimpleNamespace()
        boxes_o, scores_o, class_ids, _m, _extra, keypoints = (
            BaseBackend._parse_yolonas_pose(
                dummy, [boxes, scores, kpxy, kpconf],
                effective_imgsz=320, orig_w=320, orig_h=320, conf=0.5, ratio=1.0,
            )
        )
        assert set(class_ids.tolist()) == {0, 1}
        assert keypoints.shape[-2:] == (1, 3)

    def test_parse_single_class_scores(self):
        a = 2
        boxes = np.zeros((1, a, 4), np.float32)
        boxes[0, 0] = [10, 10, 40, 40]
        boxes[0, 1] = [200, 200, 240, 240]
        scores = np.array([[[0.9], [0.8]]], np.float32)  # [1, A, 1]
        kpxy = np.zeros((1, a, 1, 2), np.float32)
        kpconf = np.ones((1, a, 1), np.float32)
        dummy = types.SimpleNamespace()
        _b, scores_o, class_ids, _m, _e, _k = BaseBackend._parse_yolonas_pose(
            dummy, [boxes, scores, kpxy, kpconf],
            effective_imgsz=320, orig_w=320, orig_h=320, conf=0.5, ratio=1.0,
        )
        assert class_ids.tolist() == [0, 0]


# ---------------------------------------------------------------------------
# Class-id safety (Greptile P1 on PR #530 + label range validation)
# ---------------------------------------------------------------------------


class TestClassIdSafety:
    def test_single_class_nonzero_label_ids_train_class_zero(self):
        # Historical contract: single-class pose ignores the label class column.
        # A raw id of 1 (e.g. COCO person=1) must NOT reach the assigner, whose
        # background index is also 1 — that would silently drop the GT.
        loss_fn = YoloNASPoseLoss(oks_sigmas=[0.1], num_classes=1)
        targets = torch.zeros(1, 2, 5 + 3 * 1)
        targets[0, 0] = torch.tensor([1, 160, 160, 80, 80, 160, 160, 2])
        targets[0, 1] = torch.tensor([1, 60, 60, 40, 40, 60, 60, 2])
        tgt = loss_fn._unpack_padded_targets(targets)
        assert tgt["gt_class"].unique().tolist() == [0]

    def test_multiclass_gt_class_passes_through(self):
        loss_fn = YoloNASPoseLoss(oks_sigmas=[0.1], num_classes=3)
        targets = torch.zeros(1, 2, 5 + 3 * 1)
        targets[0, 0] = torch.tensor([2, 160, 160, 80, 80, 160, 160, 2])
        targets[0, 1] = torch.tensor([0, 60, 60, 40, 40, 60, 60, 2])
        tgt = loss_fn._unpack_padded_targets(targets)
        assert tgt["gt_class"][0, :, 0].tolist() == [2, 0]

    def test_parser_rejects_out_of_range_class(self):
        from libreyolo.data.pose_dataset import parse_yolo_pose_label_line

        line = "2 0.5 0.5 0.2 0.2 0.5 0.5 2".split()
        # id == nc and negatives are rejected when num_classes is given
        with pytest.raises(ValueError, match="out of range"):
            parse_yolo_pose_label_line(line, 1, 3, num_classes=2)
        with pytest.raises(ValueError, match="out of range"):
            parse_yolo_pose_label_line(
                "-1 0.5 0.5 0.2 0.2 0.5 0.5 2".split(), 1, 3, num_classes=2
            )
        # in-range id passes; without num_classes stays lenient
        cls_id, _, _ = parse_yolo_pose_label_line(line, 1, 3, num_classes=3)
        assert cls_id == 2
        cls_id, _, _ = parse_yolo_pose_label_line(line, 1, 3)
        assert cls_id == 2

    def test_dataset_skips_out_of_range_class_lines(self, tmp_path):
        from libreyolo.data.pose_dataset import YOLOPoseDataset

        img = tmp_path / "im.jpg"
        img.write_bytes(b"")  # labels load at init; the image is never opened
        lbl = tmp_path / "im.txt"
        lbl.write_text(
            "0 0.5 0.5 0.2 0.2 0.5 0.5 2\n"
            "5 0.3 0.3 0.1 0.1 0.3 0.3 2\n"  # out of range for nc=2
            "1 0.7 0.7 0.1 0.1 0.7 0.7 2\n"
        )
        ds = YOLOPoseDataset(
            [img], num_keypoints=1, label_files=[lbl], num_classes=2
        )
        assert ds.labels[0][1].tolist() == [0.0, 1.0]
        # without num_classes the historical leniency is preserved
        ds = YOLOPoseDataset([img], num_keypoints=1, label_files=[lbl])
        assert ds.labels[0][1].tolist() == [0.0, 5.0, 1.0]
