"""SegFormer trainer — plugs into the shared semantic BaseTrainer path.

Unlike EoMT/PIDNet (inference-only) or DINOv2 (inherits RF-DETR's DETR-decoder
trainer machinery it doesn't need), SegFormer has no query decoder, matcher,
or NestedTensor plumbing — it is a plain encoder + dense head. This trainer
implements only what ``task="semantic"`` actually requires:
``BaseTrainer._setup_data`` dispatches straight to ``_setup_semantic_data``
and ``_run_semantic_validation`` without ever calling ``create_transforms``.
"""

from __future__ import annotations

from typing import Dict, Type

import torch

from ...training.config import SegformerConfig, TrainConfig
from ...training.scheduler import FlatCosineScheduler, LinearLRScheduler
from ...training.trainer import BaseTrainer


class SegformerTrainer(BaseTrainer):
    """Trainer for the LibreSegformer semantic-segmentation family."""

    best_metric_key: str = "metrics/mIoU"

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return SegformerConfig

    def get_model_family(self) -> str:
        return "segformer"

    def get_model_tag(self) -> str:
        return f"LibreSegformer-{self.config.size}"

    def create_transforms(self):
        raise NotImplementedError(
            "SegFormer is semantic-only; create_transforms() is never called "
            "for task='semantic' (BaseTrainer._setup_data routes straight to "
            "_setup_semantic_data)."
        )

    def create_scheduler(self, iters_per_epoch: int):
        scheduler_name = str(self.config.scheduler).lower()
        if scheduler_name == "linear":
            return LinearLRScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=self.config.warmup_lr_start,
                min_lr_ratio=self.config.min_lr_ratio,
            )
        if scheduler_name in ("cosine", "flat_cosine", "cos"):
            return FlatCosineScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=self.config.warmup_lr_start,
                no_aug_epochs=getattr(self.config, "no_aug_epochs", 0),
                min_lr_ratio=self.config.min_lr_ratio,
            )
        raise ValueError(f"Unknown SegFormer scheduler: {self.config.scheduler!r}")

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        return self.model(imgs, targets=targets)

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        value = outputs.get("sem", 0)
        return {"sem": value.item() if isinstance(value, torch.Tensor) else float(value)}


__all__ = ["SegformerTrainer"]
