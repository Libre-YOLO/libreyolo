"""Dome-DETR trainer.

Ported from Dome-DETR (https://github.com/RicePasteM/Dome-DETR),
commit 2dde3bc1946a3e9fad9abd0612b59fc39bd6b861, Apache License 2.0.
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

Upstream builds directly on D-FINE's training loop, so this subclasses
``DFINETrainer`` and changes only what actually differs: the criterion (which
adds the DeFE density and count losses and masks padded queries out of the
classification terms) and the loss-component reporting.

Everything else — the per-group LR split, the target translation into DETR's
``list[dict]`` form, gradient clipping, EMA, the flat-cosine schedule — is
inherited unchanged, because upstream's recipe is D-FINE's with different
constants and those live in ``DOMEDETRConfig``.
"""

from __future__ import annotations

from typing import Dict, Type

import torch

from ...training.config import DOMEDETRConfig, TrainConfig
from ..dfine.matcher import HungarianMatcher
from ..dfine.trainer import DFINETrainer
from .loss import DomeCriterion


class DOMEDETRTrainer(DFINETrainer):
    """D-FINE's trainer with Dome-DETR's criterion."""

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return DOMEDETRConfig

    def get_model_family(self) -> str:
        return "domedetr"

    def get_model_tag(self) -> str:
        return f"DOMEDETR-{self.config.size}"

    def build_criterion(self, *, distributed_normalize: bool = True):
        """Dome-DETR's criterion: D-FINE's losses plus DeFE supervision.

        Weights come from ``configs/dome/include/dome_hgnetv2.yml``, which
        keeps D-FINE's ``{vfl: 1, bbox: 5, giou: 2, fgl: 0.15, ddf: 1.5}``
        unchanged and adds the density term on top.
        """
        matcher = HungarianMatcher(
            weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
        )
        return DomeCriterion(
            matcher=matcher,
            weight_dict={
                "loss_vfl": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            },
            losses=["vfl", "boxes", "local"],
            alpha=0.75,
            gamma=2.0,
            num_classes=self.config.num_classes,
            reg_max=32,
            distributed_normalize=distributed_normalize,
            defe_density_map_weight=getattr(
                self.config, "defe_density_map_weight", 1.0
            ),
            density_recall_penalty=getattr(self.config, "density_recall_penalty", 0.3),
            defe_reg_loss_weight=getattr(self.config, "defe_reg_loss_weight", 1.0),
        ).to(self.device)

    def build_validation_loss_adapter(self, model):
        """Dome-DETR's own adapter: the inherited one type-checks for LibreDFINEModel."""
        from .validation_loss import DOMEDETRValidationLoss

        return DOMEDETRValidationLoss(
            model, self.build_criterion(distributed_normalize=False)
        )

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        """Report the DeFE terms alongside D-FINE's aggregated ones.

        They are the only signal that the density head is learning anything,
        and without them a run that silently stops supervising DeFE looks
        identical to a healthy one.

        ``defe_reg`` reading a flat 0 is usually correct rather than broken:
        the count target is ``(n_objects - min_select) / (max_select -
        min_select)`` clamped to the query budget, so on any dataset whose
        images hold fewer objects than ``min_select`` (250 or 300, i.e. most
        non-crowded data) every target is 0, and a converged head predicts 0
        too. ``defe_density`` is the term to watch on sparse data.
        """
        components = super().get_loss_components(outputs)
        for key in ("loss_defe_density", "loss_defe_reg"):
            value = outputs.get(key)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach()
            components[key.replace("loss_", "")] = float(value)
        return components
