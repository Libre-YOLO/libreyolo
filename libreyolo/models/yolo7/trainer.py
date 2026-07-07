"""YOLOv7 trainer for LibreYOLO.

Thin :class:`BaseTrainer` subclass. v7 is anchor-based but reuses LibreYOLO's
YOLOX-style training pipeline (mosaic/mixup ``TrainTransform`` + SimOTA loss),
so this mirrors :class:`~libreyolo.models.yolox.trainer.YOLOXTrainer`. The loss
itself lives in the model forward (``YOLOv7Model.forward(imgs, targets)`` →
:class:`~libreyolo.models.yolo7.loss.YOLOv7Loss`).
"""

from typing import Dict, Type

import torch

from libreyolo.training.trainer import BaseTrainer
from libreyolo.training.config import TrainConfig, YOLOv7Config
from ...training.scheduler import WarmupCosineScheduler
from ...training.augment import TrainTransform, MosaicMixupDataset


class YOLOv7Trainer(BaseTrainer):
    """YOLOv7-specific trainer."""

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return YOLOv7Config

    def get_model_family(self) -> str:
        return "yolo7"

    def get_model_tag(self) -> str:
        return f"LibreYOLO7-{self.config.size}"

    def create_transforms(self):
        preproc = TrainTransform(
            max_labels=50,
            flip_prob=self.config.flip_prob,
            hsv_prob=self.config.hsv_prob,
        )
        return preproc, MosaicMixupDataset

    def create_scheduler(self, iters_per_epoch: int):
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
        }

    def on_setup(self):
        # Seed obj/cls bias priors only for a fresh (from-scratch) head; skip on
        # warm-start so loaded v7.pt priors aren't wiped (on_setup runs after
        # any checkpoint load in __init__).
        if getattr(self.wrapper_model, "model_path", None):
            return
        raw = getattr(self.model, "module", self.model)
        if hasattr(raw, "initialize_biases"):
            raw.initialize_biases(0.01)

    def on_mosaic_disable(self):
        self.train_loader.dataset.close_mosaic()

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        return self.model(imgs, targets)
