"""RT-DETRv2 loss.

RT-DETRv2 keeps the same matcher and per-layer detection losses as v1
(:class:`libreyolo.models.rtdetr.loss.SetCriterion`), but supervises the
encoder query-selection head differently.

v1 (``models/rtdetr/nn.py``) *appends* the encoder top-k predictions to the
``aux_outputs`` list, so the shared ``SetCriterion`` already trains
``enc_score_head``/``enc_bbox_head`` as if they were one more decoder aux layer.

v2 (``models/rtdetrv2/decoder.py``) instead emits those predictions under a
dedicated ``enc_aux_outputs`` key (plus an ``enc_meta`` flag), matching upstream
``rtdetrv2_pytorch``. The shared ``SetCriterion`` only consumes ``aux_outputs``
and ``dn_aux_outputs`` — it never reads ``enc_aux_outputs`` — so with the v1
criterion the v2 encoder heads receive **zero** gradient and stay at their
random init (masked in DDP by ``find_unused_parameters=True``).

:class:`RTDETRCriterionv2` restores that supervision by adding one matched
detection loss per ``enc_aux_outputs`` entry, using the exact same
matcher + ``get_loss`` path as a normal aux layer (mirroring upstream's
``RTDETRCriterionv2`` encoder-aux term, including the class-agnostic relabel).
"""

from __future__ import annotations

import copy

import torch

from ..rtdetr.loss import HungarianMatcher, SetCriterion


class RTDETRCriterionv2(SetCriterion):
    """SetCriterion + encoder query-selection (``enc_aux_outputs``) supervision."""

    def forward(self, outputs, targets):
        # Parent computes the main + decoder-aux + denoising losses (and a
        # ``total_loss`` summing them). It ignores ``enc_aux_outputs`` entirely.
        losses = super().forward(outputs, targets)

        if "enc_aux_outputs" in outputs:
            enc_losses = self._compute_enc_aux_losses(outputs, targets)
            if enc_losses:
                losses.update(enc_losses)
                # Re-sum the total to fold in the encoder-aux terms. The parent
                # already put every other loss (and only losses) into the dict,
                # so summing all non-total tensors reproduces its total plus the
                # new ``*_enc_i`` terms — no double counting.
                losses["total_loss"] = sum(
                    v
                    for k, v in losses.items()
                    if k != "total_loss" and isinstance(v, torch.Tensor)
                )

        return losses

    def _compute_enc_aux_losses(self, outputs, targets):
        """Matched detection loss on each ``enc_aux_outputs`` entry vs GT."""
        assert "enc_meta" in outputs, "enc_aux_outputs requires enc_meta"

        device = outputs["pred_logits"].device
        # Same normalization denominator as the parent's main/aux losses.
        num_boxes = self._normalizer(
            sum(len(t["labels"]) for t in targets), device=device
        )

        class_agnostic = bool(outputs["enc_meta"].get("class_agnostic", False))
        if class_agnostic:
            # Upstream collapses all GT to a single foreground class so the
            # 1-way ``enc_score_head`` (query_select_method="agnostic") matches.
            orig_num_classes = self.num_classes
            self.num_classes = 1
            enc_targets = copy.deepcopy(targets)
            for t in enc_targets:
                t["labels"] = torch.zeros_like(t["labels"])
        else:
            enc_targets = targets

        losses = {}
        try:
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                # Cast to float32 for AMP stability, matching the parent's
                # ``_cast_to_float32`` treatment of aux entries.
                aux_outputs = {
                    k: v.float() if isinstance(v, torch.Tensor) else v
                    for k, v in aux_outputs.items()
                }
                indices = self.matcher(aux_outputs, enc_targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, enc_targets, indices, num_boxes
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict.get(k, 1.0)
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        finally:
            if class_agnostic:
                self.num_classes = orig_num_classes

        return losses


def RTDETRv2Loss(num_classes=80, use_vfl=True, distributed_normalize=True):
    """Create an RTDETRv2 criterion (matches ``RTDETRLoss`` settings).

    Identical matcher / weight_dict / losses as :func:`RTDETRLoss`, but returns
    the :class:`RTDETRCriterionv2` subclass so the encoder query-selection head
    (``enc_score_head``/``enc_bbox_head``) is supervised via ``enc_aux_outputs``.
    """
    matcher_weight_dict = {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}
    matcher = HungarianMatcher(
        weight_dict=matcher_weight_dict,
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )

    weight_dict = {
        "loss_vfl": 1.0,
        "loss_focal": 1.0,
        "loss_bce": 1.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
    }

    losses = ["vfl", "boxes"] if use_vfl else ["focal", "boxes"]

    return RTDETRCriterionv2(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        alpha=0.75,
        gamma=2.0,
        num_classes=num_classes,
        distributed_normalize=distributed_normalize,
    )
