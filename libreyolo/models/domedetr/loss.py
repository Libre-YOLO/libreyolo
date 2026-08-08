"""Dome-DETR training criterion.

Ported from Dome-DETR (https://github.com/RicePasteM/Dome-DETR),
commit 2dde3bc1946a3e9fad9abd0612b59fc39bd6b861, Apache License 2.0.
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
Modified from D-FINE (https://github.com/Peterande/D-FINE).
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

Matching, box losses, the FDR local loss and the denoising bookkeeping are
D-FINE's and are inherited from ``models/dfine/loss.py``. Dome-DETR adds two
things:

**A padding-aware classification normalizer.** PAQI gives each image its own
query count, so a batch is padded to the maximum. Those padded rows still
produce decoder logits, and without masking they are scored as confident
background and dragged into the loss. The mask zeroes them *before* reduction
while leaving the ``/ num_boxes`` denominator alone, which is what upstream
does. It is applied by overriding the two classification losses rather than by
threading an argument through every call site, so the auxiliary, denoising and
encoder paths pick it up automatically via ``self`` dispatch.

**Two DeFE losses.** A density-map regression that supervises the map MWAS and
PAQI consume, with an asymmetric penalty that punishes under-estimation
(missing a crowded region costs more than hallucinating one), and an
object-count regression on the auxiliary ``reg_value`` head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..dfine.loss import DFINECriterion


class DomeCriterion(DFINECriterion):
    """D-FINE's criterion plus padding-aware scoring and the DeFE losses."""

    def __init__(
        self,
        *args,
        defe_density_map_weight: float = 1.0,
        density_recall_penalty: float = 0.3,
        defe_reg_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.defe_density_map_weight = defe_density_map_weight
        self.density_recall_penalty = density_recall_penalty
        self.defe_reg_loss_weight = defe_reg_loss_weight
        # Set per forward; read by the classification-loss overrides.
        self._batch_queries_num: list[int] | None = None

    # -- padding-aware classification -------------------------------------

    def _valid_query_mask(self, src_logits: torch.Tensor) -> torch.Tensor | None:
        """``(B, N, 1)`` mask of real (non-padded) queries, or None."""
        if self._batch_queries_num is None:
            return None
        counts = torch.as_tensor(
            self._batch_queries_num, device=src_logits.device
        ).reshape(-1, 1)
        positions = torch.arange(src_logits.shape[1], device=src_logits.device)[None, :]
        return (positions < counts).unsqueeze(-1)

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        result = super().loss_labels_vfl(outputs, targets, indices, num_boxes, values)
        return self._rescore(result, "loss_vfl", outputs, targets, indices, num_boxes, values)

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        result = super().loss_labels_focal(outputs, targets, indices, num_boxes)
        return self._rescore(result, "loss_focal", outputs, targets, indices, num_boxes, None)

    def _rescore(self, result, key, outputs, targets, indices, num_boxes, values):
        """Recompute the classification loss with padded queries masked out.

        The mask has to be applied to the per-element loss before reduction, so
        the parent's scalar cannot be corrected after the fact. When there is
        no padding (single image, or every image landing on the same query
        count) the parent's value is already right and is returned untouched.
        """
        src_logits = outputs["pred_logits"]
        mask = self._valid_query_mask(src_logits)
        if mask is None or bool(mask.all()):
            return result

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        if key == "loss_vfl":
            if values is None:
                from ..dfine.box_ops import box_cxcywh_to_xyxy, box_iou

                src_boxes = outputs["pred_boxes"][idx]
                target_boxes = torch.cat(
                    [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
                )
                ious, _ = box_iou(
                    box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
                )
                ious = torch.diag(ious).detach()
            else:
                ious = values

            target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
            target_score_o[idx] = ious.to(target_score_o.dtype)
            target_score = target_score_o.unsqueeze(-1) * target

            pred_score = F.sigmoid(src_logits).detach()
            weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
            loss = F.binary_cross_entropy_with_logits(
                src_logits, target_score, weight=weight, reduction="none"
            )
        else:
            target_score = target.to(src_logits.dtype)
            loss = torchvision_sigmoid_focal_loss(
                src_logits, target_score, self.alpha, self.gamma
            )

        loss = loss * mask
        scaled = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {key: scaled}

    # -- DeFE supervision ---------------------------------------------------

    def _defe_losses(self, outputs, targets) -> dict:
        defe = outputs.get("defe")
        if not defe or "gt_density_map" not in defe:
            return {}

        losses = {}

        density_map = defe.get("defe_feature")
        gt_density_map = defe.get("gt_density_map")
        if density_map is not None and isinstance(gt_density_map, torch.Tensor):
            if density_map.shape[-2:] != gt_density_map.shape[-2:]:
                gt_density_map = F.interpolate(
                    gt_density_map,
                    size=density_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            diff = density_map - gt_density_map
            # Under-estimating a dense region loses detections; over-estimating
            # only wastes queries. Penalise the first harder.
            underestimation = (density_map < gt_density_map).to(diff.dtype)
            penalty = 1 + self.density_recall_penalty * gt_density_map * underestimation
            losses["loss_defe_density"] = (
                (penalty * diff.pow(2)).mean() * self.defe_density_map_weight
            )

        reg_value = defe.get("reg_value")
        min_select, max_select = defe.get("min_num_select"), defe.get("max_num_select")
        if reg_value is not None and min_select is not None and max_select != min_select:
            reg_targets = []
            for target in targets:
                count = int(target["labels"].shape[0])
                count = min(max(count, min_select), max_select)
                reg_targets.append((count - min_select) / (max_select - min_select))
            # NOTE: upstream casts this to int64, which truncates every value
            # below 1.0 to 0 and makes the loss pull reg_value to zero whatever
            # the object count is. Kept as float here: the cast is plainly
            # accidental, and reg_value feeds nothing at inference, so neither
            # version can change detection quality.
            reg_target = torch.as_tensor(
                reg_targets, dtype=reg_value.dtype, device=reg_value.device
            ).reshape(reg_value.shape)
            diff = reg_value - reg_target
            weights = torch.where(diff < 0, 2.0, 1.0)
            losses["loss_defe_reg"] = (
                (weights * diff.pow(2)).mean() * self.defe_reg_loss_weight
            )

        return losses

    def forward(self, outputs, targets, **kwargs):
        self._batch_queries_num = outputs.get("batch_queries_num")
        try:
            losses = super().forward(outputs, targets, **kwargs)
        finally:
            self._batch_queries_num = None
        losses.update(self._defe_losses(outputs, targets))
        return losses


def torchvision_sigmoid_focal_loss(inputs, targets, alpha, gamma):
    """Unreduced sigmoid focal loss, matching torchvision's formulation."""
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss
