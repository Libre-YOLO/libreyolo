"""LibreConvNeXt classification trainer.

Thin :class:`BaseTrainer` subclass. The data, validation, and head-rebuild
machinery is the shared classification path in ``BaseTrainer``
(``_setup_classify_data`` / ``_run_classify_validation``); this class only
supplies the cross-entropy forward and a cosine LR schedule.
"""

from __future__ import annotations

from typing import Dict, Type

import torch
import torch.nn.functional as F

from ...training.config import TrainConfig
from ...training.scheduler import WarmupCosineScheduler
from ...training.trainer import BaseTrainer
from .config import ConvNeXtConfig


class ConvNeXtTrainer(BaseTrainer):
    """Cross-entropy fine-tuning trainer for ConvNeXt classification."""

    # Non-detect task: select the best checkpoint by top-1 accuracy.
    best_metric_key = "metrics/accuracy_top1"

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return ConvNeXtConfig

    def get_model_family(self) -> str:
        return "convnext"

    def get_model_tag(self) -> str:
        return f"ConvNeXt-{self.config.size}"

    def create_transforms(self):
        # Classification uses the ImageFolder pipeline in ``_setup_classify_data``,
        # so the detection transform factory is never invoked for this task.
        return None, None

    def create_scheduler(self, iters_per_epoch: int):
        return WarmupCosineScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            plateau_epochs=self.config.no_aug_epochs,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        logits = self.model(imgs)
        loss = F.cross_entropy(logits, targets)
        return {"total_loss": loss, "loss_ce": loss.detach()}

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        return {"ce": float(outputs.get("loss_ce", outputs["total_loss"]))}
