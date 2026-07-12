"""
YOLOX Trainer for LibreYOLO.

Thin subclass of BaseTrainer with YOLOX-specific transforms, scheduler,
loss extraction, and bias initialisation.
"""

import torch
from typing import Dict, Type

from libreyolo.training.trainer import BaseTrainer
from libreyolo.training.config import (
    TrainConfig,
    YOLOXConfig,
    require_training_choice,
)
from ...training.scheduler import WarmupCosineScheduler
from ...training.augment import TrainTransform, MosaicMixupDataset


class YOLOXTrainer(BaseTrainer):
    """YOLOX-specific trainer."""

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return YOLOXConfig

    def get_model_family(self) -> str:
        return "yolox"

    def get_model_tag(self) -> str:
        return f"YOLOX-{self.config.size}"

    def create_transforms(self):
        preproc = TrainTransform(
            max_labels=50,
            flip_prob=self.config.flip_prob,
            hsv_prob=self.config.hsv_prob,
            flipud=getattr(self.config, "flipud", 0.0),
        )
        return preproc, MosaicMixupDataset

    def create_scheduler(self, iters_per_epoch: int):
        require_training_choice(
            self.config.scheduler,
            field="scheduler",
            supported=("yoloxwarmcos",),
            family=self.get_model_family(),
        )
        return WarmupCosineScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            warmup_lr_start=self.config.warmup_lr_start,
            plateau_epochs=self.config.no_aug_epochs,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        return {
            "iou": outputs.get("iou_loss", 0),
            "obj": outputs.get("obj_loss", 0),
            "cls": outputs.get("cls_loss", 0),
            "l1": outputs.get("l1_loss", 0),
        }

    def on_setup(self):
        # Only seed focal-loss bias priors for a fresh (from-scratch) head.
        # on_setup runs after any pretrained/resume checkpoint has been loaded
        # in __init__, so unconditionally calling initialize_biases would wipe
        # the learned cls/obj priors on every warm-start.
        if getattr(self.wrapper_model, "model_path", None):
            return
        raw = getattr(self.model, "module", self.model)
        if hasattr(raw, "head") and hasattr(raw.head, "initialize_biases"):
            raw.head.initialize_biases(0.01)

    def on_mosaic_disable(self):
        self.train_loader.dataset.close_mosaic()
        raw = getattr(self.model, "module", self.model)
        raw.head.use_l1 = True

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        return self.model(imgs, targets)
