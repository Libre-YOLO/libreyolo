"""YOLOv7 training loss — SimOTA dynamic assignment over the anchor-based head.

Why this exists and where it comes from
---------------------------------------
The MIT source of our v7 architecture (MultimediaTechLab/YOLO) ships **no** v7
training loss: its ``YOLOLoss``/``DualLoss`` are anchor-free (``Vec2Box`` + DFL),
built for YOLOv9's dual head only. v7 there is inference-only (``Anc2Box``
decode). So there is nothing to port from that repo for training.

Rather than adapt the GPL-3.0 ``WongKinYiu/yolov7`` / AGPL YOLOv5 anchor loss,
this module reuses LibreYOLO's own in-repo **YOLOX SimOTA**
(``libreyolo/models/yolox/``, Apache-2.0, Megvii YOLOX lineage). SimOTA is the
same dynamic-assignment family as v7's OTA, and it is head-agnostic: it operates
on decoded boxes plus objectness/class logits and per-anchor grid shifts. We
drive it with v7's ``Anc2Box``-decoded anchor predictions (3 anchors/cell,
``xy=(2σ-0.5+grid)·stride``, ``wh=(2σ)²·anchor``). No GPL YOLOv5/YOLOv7 code is
read or used.

``bboxes_iou`` and ``IoULoss`` are imported from the in-repo Apache-2.0 YOLOX
modules (single shared implementation, no copies); the assignment logic
(``get_losses``/``get_assignments``/``get_geometry_constraint``/
``simota_matching`` including the CUDA-OOM CPU retry) mirrors YOLOX's with
v7's decode substituted for the anchor-free one.

AMP note: the head maps are upcast to fp32 in ``_decode`` before any box math.
v7's pixel-scale anchors (up to 459x401 -> ~184k px^2 at init) overflow fp16's
65504 max inside ``torch.prod`` box areas, which yields NaN IoUs/gradients under
autocast; the loss therefore always runs in fp32 regardless of AMP.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...training.distributed import all_reduce_avg_scalar
from ..yolox.loss import IoULoss
from ..yolox.nn import bboxes_iou


class YOLOv7Loss:
    """SimOTA loss for the v7 anchor head.

    Plain class (not an ``nn.Module``) on purpose: it holds only stateless loss
    modules and constant anchor/grid tensors, so it is never registered on the
    model and never enters the checkpoint (v7.pt has no loss keys → strict load
    stays 564/564).

    Args:
        num_classes: number of classes.
        anchors: per-head flat anchor list ``[w0,h0,w1,h1,w2,h2]`` (pixel units),
            exactly as stored on ``YOLOv7Model.anchors`` / ``v7.yaml``.
        strides: per-head strides, e.g. ``[8, 16, 32]``.
    """

    def __init__(self, num_classes: int,
                 anchors: Sequence[Sequence[float]],
                 strides: Sequence[int],
                 distributed_normalize: bool = True):
        self.num_classes = int(num_classes)
        self.distributed_normalize = distributed_normalize
        self.anchors = [
            torch.tensor(list(zip(a[0::2], a[1::2])), dtype=torch.float32)
            for a in anchors
        ]  # list of [A, 2] (w, h) pixel sizes
        self.strides = [int(s) for s in strides]
        self.n_anchors = self.anchors[0].shape[0]
        self.iou_loss = IoULoss(reduction="none")
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # Grid/shift/stride tensors depend only on (head, H, W, device); cache
        # them across steps like YOLOX's self.grids instead of rebuilding
        # meshgrids and re-copying anchors to the device every iteration.
        self._grid_cache: Dict[Tuple, Tuple[torch.Tensor, ...]] = {}

    def __call__(self, raw_outputs: List[torch.Tensor],
                 labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.get_losses(raw_outputs, labels)

    # -- decode -------------------------------------------------------------
    def _grids_for(self, head_idx: int, H: int, W: int, device: torch.device):
        """Cached per-(head, H, W, device) grid/anchor/shift constants (fp32)."""
        key = (head_idx, H, W, str(device))
        hit = self._grid_cache.get(key)
        if hit is not None:
            return hit
        A = self.n_anchors
        anchors = self.anchors[head_idx].to(device=device)  # [A, 2] fp32
        yv, xv = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )  # each [H, W]; yv = row index, xv = col index
        col = xv.view(1, 1, H, W)
        row = yv.view(1, 1, H, W)
        aw = anchors[:, 0].view(1, A, 1, 1)
        ah = anchors[:, 1].view(1, A, 1, 1)
        # Grid shifts tiled anchor-major to match the (A, H, W) flatten order.
        gx = xv.reshape(-1).repeat(A).view(1, -1)
        gy = yv.reshape(-1).repeat(A).view(1, -1)
        es = torch.full((1, A * H * W), float(self.strides[head_idx]),
                        device=device, dtype=torch.float32)
        hit = (col, row, aw, ah, gx, gy, es)
        self._grid_cache[key] = hit
        return hit

    def _decode(self, raw_outputs: List[torch.Tensor]):
        """Decode raw head maps to flat predictions + grid metadata.

        Returns ``(outputs, x_shifts, y_shifts, expanded_strides)`` where
        ``outputs`` is ``[B, N, 4+1+nc]`` (cxcywh **pixels** + obj/cls **logits**)
        and the shift/stride lists are per-head ``[1, A·H·W]`` — the SimOTA
        center-prior operates on the grid-cell centres shared by the A anchors.

        Always computes in fp32 (see module docstring: fp16 box areas overflow).
        """
        nc = self.num_classes
        A = self.n_anchors
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
        for head_idx, (raw, stride) in enumerate(zip(raw_outputs, self.strides)):
            B, _, H, W = raw.shape
            raw = raw.float()  # fp32 loss path even under AMP
            col, row, aw, ah, gx, gy, es = self._grids_for(head_idx, H, W, raw.device)

            # [B, A, 5+nc, H, W] -> [B, A, H, W, 5+nc]
            r = raw.view(B, A, 5 + nc, H, W).permute(0, 1, 3, 4, 2)
            sig = r[..., :4].sigmoid()

            cx = (sig[..., 0] * 2 - 0.5 + col) * stride
            cy = (sig[..., 1] * 2 - 0.5 + row) * stride
            bw = (sig[..., 2] * 2) ** 2 * aw
            bh = (sig[..., 3] * 2) ** 2 * ah
            box = torch.stack([cx, cy, bw, bh], dim=-1)          # [B,A,H,W,4]
            out = torch.cat([box, r[..., 4:]], dim=-1)           # + obj/cls logits
            outputs.append(out.reshape(B, A * H * W, 5 + nc))

            x_shifts.append(gx)
            y_shifts.append(gy)
            expanded_strides.append(es)
        return torch.cat(outputs, dim=1), x_shifts, y_shifts, expanded_strides

    # -- losses -------------------------------------------------------------
    def get_losses(self, raw_outputs: List[torch.Tensor],
                   labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs, x_shifts, y_shifts, expanded_strides = self._decode(raw_outputs)
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]

        labels = labels.float()
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
                # CUDA-OOM retry ported from YOLOX get_losses: the [num_gt, N]
                # cost/BCE matrices can spike on crowded images (v7 has 3x
                # YOLOX's anchors), so fall back to a CPU assignment pass
                # instead of killing the run.
                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx, num_gt, gt_bboxes, gt_classes,
                        bbox_preds[batch_idx], expanded_strides, x_shifts,
                        y_shifts, cls_preds, obj_preds,
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" not in str(e):
                        raise
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx, num_gt, gt_bboxes, gt_classes,
                        bbox_preds[batch_idx], expanded_strides, x_shifts,
                        y_shifts, cls_preds, obj_preds, "cpu",
                    )
                    torch.cuda.empty_cache()
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

        # Raw match count BEFORE the >=1 normalization clamp, so the reported
        # metric can actually read 0 when SimOTA matched nothing.
        raw_num_fg = num_fg
        # Global (DDP-reduced) positive count: dividing by the global count
        # keeps DDP's gradient averaging equivalent to single-GPU training on
        # the same global batch (issue #484). Identical to the previous
        # ``max(num_fg, 1)`` outside DDP. Rank-0-only validation loss selects
        # the local path because it cannot enter a collective while the other
        # ranks wait at the validation barrier.
        if self.distributed_normalize:
            num_fg = all_reduce_avg_scalar(num_fg, device=bbox_preds.device)
        else:
            num_fg = float(max(num_fg, 1.0))
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
            "num_fg": raw_num_fg / max(num_gts, 1),
        }

    # -- SimOTA (adapted from Apache-2.0 YOLOX) -----------------------------
    @torch.no_grad()
    def get_assignments(self, batch_idx, num_gt, gt_bboxes, gt_classes,
                        bboxes_preds_per_image, expanded_strides,
                        x_shifts, y_shifts, cls_preds, obj_preds, mode="gpu"):
        if mode == "cpu":
            gt_bboxes = gt_bboxes.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

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
            gt_matched_classes = gt_classes.new_zeros(0)
            pred_ious_this_matching = gt_classes.new_zeros(0, dtype=torch.float)
            matched_gt_inds = gt_classes.new_zeros(0, dtype=torch.long)
            if mode == "cpu":
                gt_matched_classes = gt_matched_classes.cuda()
                fg_mask = fg_mask.cuda()
                pred_ious_this_matching = pred_ious_this_matching.cuda()
                matched_gt_inds = matched_gt_inds.cuda()
            return (gt_matched_classes, fg_mask, pred_ious_this_matching,
                    matched_gt_inds, 0)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes, bboxes_preds_per_image, False)
        gt_cls_per_image = F.one_hot(
            gt_classes.to(torch.int64), self.num_classes
        ).float()
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # BCE on probabilities is unsafe under an active autocast region, so
        # force it off exactly as the YOLOX source does. Hardcoding "cuda" is
        # deliberate and host-safe: BaseTrainer only ever enables autocast for
        # CUDA, and constructing a disabled "cuda" context works on CPU/MPS,
        # whereas device-generic device_type crashes on MPS with torch 2.4.x.
        with torch.amp.autocast("cuda", enabled=False):
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
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

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
