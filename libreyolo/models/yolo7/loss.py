"""YOLOv7 training loss — SimOTA dynamic assignment over the anchor-based head.

Why this exists and where it comes from
---------------------------------------
The MIT source of our v7 architecture (MultimediaTechLab/YOLO) ships **no** v7
training loss: its ``YOLOLoss``/``DualLoss`` are anchor-free (``Vec2Box`` + DFL),
built for YOLOv9's dual head only. v7 there is inference-only (``Anc2Box``
decode). So there is nothing to port from that repo for training.

Rather than adapt the GPL-3.0 ``WongKinYiu/yolov7`` / AGPL YOLOv5 anchor loss,
this module is adapted from LibreYOLO's own in-repo **YOLOX SimOTA**
(``libreyolo/models/yolox/``, Apache-2.0, Megvii YOLOX lineage). SimOTA is the
same dynamic-assignment family as v7's OTA, and it is head-agnostic: it operates
on decoded boxes plus objectness/class logits and per-anchor grid shifts. We
drive it with v7's ``Anc2Box``-decoded anchor predictions (3 anchors/cell,
``xy=(2σ-0.5+grid)·stride``, ``wh=(2σ)²·anchor``). No GPL YOLOv5/YOLOv7 code is
read or used.

``bboxes_iou`` and ``_IoULoss`` are copied verbatim from the in-repo Apache-2.0
YOLOX modules; the assignment/geometry/matching logic mirrors YOLOX's
``get_losses``/``get_assignments`` with v7's decode substituted for the
anchor-free one.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# IoU utilities — verbatim from libreyolo/models/yolox (Apache-2.0, Megvii).
# ---------------------------------------------------------------------------
def bboxes_iou(bboxes_a: torch.Tensor, bboxes_b: torch.Tensor,
               xyxy: bool = True) -> torch.Tensor:
    """Pairwise IoU between two box sets. cxcywh when ``xyxy=False``."""
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).to(device=tl.device, dtype=tl.dtype).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, None] + area_b - area_i)


class _IoULoss(nn.Module):
    """IoU regression loss on cxcywh boxes (Apache-2.0 YOLOX ``IoULoss``)."""

    def __init__(self, reduction: str = "none", loss_type: str = "iou"):
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape[0] == target.shape[0]
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2),
                       (target[:, :2] - target[:, 2:] / 2))
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2),
                       (target[:, :2] + target[:, 2:] / 2))
        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)
        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2),
                             (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2),
                             (target[:, :2] + target[:, 2:] / 2))
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        else:
            raise ValueError(self.loss_type)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class YOLOv7Loss:
    """SimOTA loss for the v7 anchor head.

    Plain class (not an ``nn.Module``) on purpose: it holds only stateless loss
    modules and constant anchor tensors, so it is never registered on the model
    and never enters the checkpoint (v7.pt has no loss keys → strict load stays
    564/564).

    Args:
        num_classes: number of classes.
        anchors: per-head flat anchor list ``[w0,h0,w1,h1,w2,h2]`` (pixel units),
            exactly as stored on ``YOLOv7Model.anchors`` / ``v7.yaml``.
        strides: per-head strides, e.g. ``[8, 16, 32]``.
    """

    def __init__(self, num_classes: int,
                 anchors: Sequence[Sequence[float]],
                 strides: Sequence[int]):
        self.num_classes = int(num_classes)
        self.anchors = [
            torch.tensor(list(zip(a[0::2], a[1::2])), dtype=torch.float32)
            for a in anchors
        ]  # list of [A, 2] (w, h) pixel sizes
        self.strides = [int(s) for s in strides]
        self.n_anchors = self.anchors[0].shape[0]
        self.iou_loss = _IoULoss(reduction="none")
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(self, raw_outputs: List[torch.Tensor],
                 labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.get_losses(raw_outputs, labels)

    # -- decode -------------------------------------------------------------
    def _decode(self, raw_outputs: List[torch.Tensor]):
        """Decode raw head maps to flat predictions + grid metadata.

        Returns ``(outputs, x_shifts, y_shifts, expanded_strides)`` where
        ``outputs`` is ``[B, N, 4+1+nc]`` (cxcywh **pixels** + obj/cls **logits**)
        and the shift/stride lists are per-head ``[1, A·H·W]`` — the SimOTA
        center-prior operates on the grid-cell centres shared by the A anchors.
        """
        nc = self.num_classes
        A = self.n_anchors
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
        for raw, anchors, stride in zip(raw_outputs, self.anchors, self.strides):
            B, _, H, W = raw.shape
            device, dtype = raw.device, raw.dtype
            anchors = anchors.to(device=device, dtype=dtype)  # [A, 2]

            # [B, A, 5+nc, H, W] -> [B, A, H, W, 5+nc]
            r = raw.view(B, A, 5 + nc, H, W).permute(0, 1, 3, 4, 2)
            sig = r[..., :4].sigmoid()

            yv, xv = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing="ij",
            )  # each [H, W]; yv = row index, xv = col index
            col = xv.view(1, 1, H, W)
            row = yv.view(1, 1, H, W)
            aw = anchors[:, 0].view(1, A, 1, 1)
            ah = anchors[:, 1].view(1, A, 1, 1)

            cx = (sig[..., 0] * 2 - 0.5 + col) * stride
            cy = (sig[..., 1] * 2 - 0.5 + row) * stride
            bw = (sig[..., 2] * 2) ** 2 * aw
            bh = (sig[..., 3] * 2) ** 2 * ah
            box = torch.stack([cx, cy, bw, bh], dim=-1)          # [B,A,H,W,4]
            out = torch.cat([box, r[..., 4:]], dim=-1)           # + obj/cls logits
            outputs.append(out.reshape(B, A * H * W, 5 + nc))

            # Grid shifts tiled anchor-major to match the (A,H,W) flatten order.
            gx = xv.reshape(-1).repeat(A).view(1, -1)
            gy = yv.reshape(-1).repeat(A).view(1, -1)
            x_shifts.append(gx)
            y_shifts.append(gy)
            expanded_strides.append(
                torch.full((1, A * H * W), stride, device=device, dtype=dtype)
            )
        return torch.cat(outputs, dim=1), x_shifts, y_shifts, expanded_strides

    # -- losses -------------------------------------------------------------
    def get_losses(self, raw_outputs: List[torch.Tensor],
                   labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs, x_shifts, y_shifts, expanded_strides = self._decode(raw_outputs)
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]

        # labels: [B, max_gt, 5] = (cls, cx, cy, w, h) in pixels; zero-rows pad.
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)

        cls_targets, reg_targets, obj_targets, fg_masks = [], [], [], []
        num_fg = 0.0
        num_gts = 0.0
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(
                    batch_idx, num_gt, gt_bboxes, gt_classes, bbox_preds[batch_idx],
                    expanded_strides, x_shifts, y_shifts, cls_preds, obj_preds,
                )
                num_fg += num_fg_img
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes[matched_gt_inds]

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(outputs.dtype))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        loss_iou = self.iou_loss(
            bbox_preds.view(-1, 4)[fg_masks], reg_targets
        ).sum() / num_fg
        loss_obj = self.bce(obj_preds.view(-1, 1), obj_targets).sum() / num_fg
        loss_cls = self.bce(
            cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
        ).sum() / num_fg

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls
        return {
            "total_loss": loss,
            "iou_loss": reg_weight * loss_iou,
            "obj_loss": loss_obj,
            "cls_loss": loss_cls,
            "num_fg": num_fg / max(num_gts, 1),
        }

    # -- SimOTA (adapted from Apache-2.0 YOLOX) -----------------------------
    @torch.no_grad()
    def get_assignments(self, batch_idx, num_gt, gt_bboxes, gt_classes,
                        bboxes_preds_per_image, expanded_strides,
                        x_shifts, y_shifts, cls_preds, obj_preds):
        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes, expanded_strides, x_shifts, y_shifts
        )
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        # No anchor centre falls near any GT (e.g. a degenerate/off-grid box):
        # nothing to match, so return an empty assignment instead of letting
        # SimOTA's topk crash. fg_mask stays all-False for this image.
        if num_in_boxes_anchor == 0:
            return (
                gt_classes.new_zeros(0),
                fg_mask,
                gt_classes.new_zeros(0, dtype=torch.float),
                gt_classes.new_zeros(0, dtype=torch.long),
                0,
            )

        pair_wise_ious = bboxes_iou(gt_bboxes, bboxes_preds_per_image, False)
        gt_cls_per_image = F.one_hot(
            gt_classes.to(torch.int64), self.num_classes
        ).float()
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # BCE on probabilities is unsafe under AMP autocast, so force fp32 for
        # this block (matches YOLOX). device_type keeps it CPU/CUDA-agnostic.
        with torch.autocast(device_type=cls_preds_.device.type, enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none",
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        return (gt_matched_classes, fg_mask, pred_ious_this_matching,
                matched_gt_inds, num_fg)

    def get_geometry_constraint(self, gt_bboxes, expanded_strides,
                                x_shifts, y_shifts):
        """Center-prior: keep anchors whose cell centre is near a GT centre."""
        expanded_strides_per_image = expanded_strides[0]
        x_centers = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_l = gt_bboxes[:, 0:1] - center_dist
        gt_r = gt_bboxes[:, 0:1] + center_dist
        gt_t = gt_bboxes[:, 1:2] - center_dist
        gt_b = gt_bboxes[:, 1:2] + center_dist

        c_l = x_centers - gt_l
        c_r = gt_r - x_centers
        c_t = y_centers - gt_t
        c_b = gt_b - y_centers
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]
        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1
        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
