"""YOLOv7 trainer for LibreYOLO.

Subclasses :class:`~libreyolo.models.yolox.trainer.YOLOXTrainer` (house
precedent: ``ECTrainer(DFINETrainer)``) because v7 shares YOLOX's SimOTA-style
training pipeline; only the real differences are overridden.

Input-domain note: v7 serves RGB /255 (``models/yolo7/utils.py`` preprocess,
``val_preprocessor_class = YOLO9ValPreprocessor``), so training reuses YOLO9's
RGB//255 transform family — NOT YOLOX's ``TrainTransform``, which finalizes
BGR/0-255 and would train a model in a different input domain than every
inference/validation surface evaluates. ``YOLO9TrainTransform`` emits labels as
normalized xyxy ``[cls, x1, y1, x2, y2]``; :meth:`on_forward` converts them to
the pixel cxcywh format :class:`~libreyolo.models.yolo7.loss.YOLOv7Loss`
expects.
"""

from typing import Dict, Type

import torch

from libreyolo.training.trainer import BaseTrainer
from libreyolo.training.config import TrainConfig, YOLOv7Config
from ..yolo9.transforms import YOLO9TrainTransform, YOLO9MosaicMixupDataset
from ..yolox.trainer import YOLOXTrainer


class YOLOv7Trainer(YOLOXTrainer):
    """YOLOv7-specific trainer."""

    # Opt into the standard per-epoch results.csv / summary.json artifacts.
    artifact_model_families = ("yolo7",)

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return YOLOv7Config

    def get_model_family(self) -> str:
        return "yolo7"

    def get_model_tag(self) -> str:
        return f"LibreYOLO7-{self.config.size}"

    def create_transforms(self):
        # RGB//255 to match v7's inference/val contract (see module docstring).
        preproc = YOLO9TrainTransform(
            max_labels=getattr(self.config, "max_labels", 100),
            flip_prob=self.config.flip_prob,
            vertical_flip_prob=getattr(self.config, "flipud", 0.0),
            hsv_prob=self.config.hsv_prob,
        )
        return preproc, YOLO9MosaicMixupDataset

    def on_setup(self):
        # Bias priming is decided in LibreYOLO7.train() (it knows whether the
        # head is fresh, pretrained-loaded, or nc-rebuilt); the YOLOX override
        # keys on model_path, which is wrong for the rebuild flow.
        pass

    def on_mosaic_disable(self):
        # v7's net has no head.use_l1 stage (unlike YOLOX); just close mosaic.
        BaseTrainer.on_mosaic_disable(self)

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        # YOLO9TrainTransform labels are [cls, x1, y1, x2, y2] normalized to
        # [0, 1]; YOLOv7Loss wants [cls, cx, cy, w, h] in input pixels.
        # Zero-padded rows stay all-zero through this conversion.
        h, w = imgs.shape[-2:]
        x1 = targets[..., 1] * w
        y1 = targets[..., 2] * h
        x2 = targets[..., 3] * w
        y2 = targets[..., 4] * h
        converted = torch.stack(
            [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1
        )
        targets = torch.cat([targets[..., :1], converted], dim=-1)
        return self.model(imgs, targets)
