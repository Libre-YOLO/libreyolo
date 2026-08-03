"""PICODET trainer.

Thin subclass of :class:`BaseTrainer` that:

* Preprocesses training data to match PICODET's inference/val distribution:
  an RGB **stretch** resize (aspect ratio not preserved) with ImageNet
  mean/std applied in 0-255 space (see :class:`PICODETTrainTransform`). This
  replaces the shared YOLOX BGR-letterbox path, which mismatched inference on
  three axes (letterbox vs stretch, BGR vs RGB, raw 0-255 vs normalized).
  A documented gap vs Bo's upstream ``MinIoURandomCrop`` /
  ``PhotoMetricDistortion`` / multiscale resize remains: v1 ships with hflip +
  ImageNet normalisation; the upstream-faithful augmentations land in a
  follow-up.
* Converts BaseTrainer's padded ``(B, max_labels, 5)`` ``[class, cx, cy, w, h]``
  pixel-coord targets into the ``(gt_boxes_xyxy, gt_labels)`` per-image
  lists that :class:`PICODETLoss` consumes.
* Returns the loss dict.
"""

from __future__ import annotations

import random
from typing import Dict, Type

import cv2
import numpy as np
import torch

from ...data.augment.boxes import xyxy2cxcywh
from ...data.augment.color import augment_hsv
from ...data.augment.geometry import mirror
from ...training.augment import MosaicMixupDataset
from ...training.config import PICODETConfig, TrainConfig
from ...training.scheduler import WarmupCosineScheduler
from ...training.trainer import BaseTrainer
from .loss import PICODETLoss
from .utils import IMAGENET_MEAN, IMAGENET_STD


class PICODETTrainTransform:
    """Training transform reproducing PICODET's inference/val input distribution.

    PICODET infers and validates on an RGB **stretch** resize (aspect ratio not
    preserved) with ImageNet mean/std applied in 0-255 space — see
    :func:`libreyolo.models.picodet.utils.preprocess_numpy` and
    ``PICODETValPreprocessor``. The shared YOLOX ``TrainTransform`` instead
    produced a raw 0-255 **BGR letterbox**, mismatching inference on three axes
    (letterbox vs stretch, BGR vs RGB, unnormalized vs normalized).

    This transform mirrors the eval pipeline exactly — a single stretch resize,
    BGR->RGB, then ImageNet normalization — and adds training-time hflip/hsv. It
    emits ``[class, cx, cy, w, h]`` padded labels in the resized pixel frame,
    the contract :class:`PICODETTrainer.on_forward` consumes.

    ``wants_unresized_image = True`` asks the dataset for the original-resolution
    image plus original-coordinate labels so the stretch is a single resample
    (no letterbox-then-stretch double resize), matching ``PICODETValPreprocessor``.
    """

    # ImageNet stats in RGB order, 0-255 space (single source of truth: utils).
    _MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
    _STD = np.array(IMAGENET_STD, dtype=np.float32)

    wants_unresized_image = True

    def __init__(self, max_labels: int = 50, flip_prob: float = 0.5, hsv_prob: float = 0.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        input_h, input_w = int(input_dim[0]), int(input_dim[1])
        orig_h, orig_w = image.shape[:2]

        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()

        # Photometric + geometric augs run in the original BGR pixel space.
        if self.hsv_prob > 0 and random.random() < self.hsv_prob:
            augment_hsv(image)
        image, boxes = mirror(image, boxes, self.flip_prob)

        # BGR -> RGB, then a single stretch resize to the square model canvas.
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        resized = cv2.resize(
            rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        # ImageNet mean/std in 0-255 RGB space (matches inference/val exactly).
        resized = (resized - self._MEAN) / self._STD
        img_chw = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)

        padded_labels = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(boxes) > 0:
            # Stretch box scaling (independent x/y factors), same as the val path.
            scale_x = input_w / orig_w
            scale_y = input_h / orig_h
            boxes[:, 0::2] *= scale_x
            boxes[:, 1::2] *= scale_y
            boxes = xyxy2cxcywh(boxes)
            mask = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
            boxes = boxes[mask]
            labels = labels[mask]
            n = min(len(boxes), self.max_labels)
            if n > 0:
                padded_labels[:n, 0] = labels[:n]
                padded_labels[:n, 1:] = boxes[:n]

        return img_chw, padded_labels


class PICODETTrainer(BaseTrainer):
    """PICODET trainer."""

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return PICODETConfig

    def get_model_family(self) -> str:
        return "picodet"

    def get_model_tag(self) -> str:
        return f"PICODET-{self.config.size}"

    def create_transforms(self):
        # Mosaic prob 0 + mixup prob 0 in PICODETConfig makes this a plain
        # hflip + stretch-resize + ImageNet-normalisation pipeline that matches
        # the inference/val distribution (RGB stretch + ImageNet mean/std).
        preproc = PICODETTrainTransform(
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
            "cls": float(outputs.get("loss_cls", 0)),
            "bbox": float(outputs.get("loss_bbox", 0)),
            "dfl": float(outputs.get("loss_dfl", 0)),
        }

    def on_setup(self) -> None:
        # Build the loss module once with the model's actual class count.
        self._loss_fn = self.build_criterion()

    def build_criterion(self, *, distributed_normalize: bool = True) -> PICODETLoss:
        """Build the training criterion.

        Validation loss builds a second one with ``distributed_normalize``
        off, so both stay defined in exactly one place.
        """
        nc = getattr(self.model.head, "num_classes", 80)
        reg_max = getattr(self.model.head, "reg_max", 7)
        strides = tuple(getattr(self.model.head, "strides", (8, 16, 32, 64)))
        return PICODETLoss(
            num_classes=nc,
            reg_max=reg_max,
            strides=strides,
            distributed_normalize=distributed_normalize,
        ).to(self.device)

    def validate_validation_loss_config(self) -> None:
        if not getattr(self.config, "val_loss", False):
            return
        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        if task != "detect":
            raise ValueError(
                "val_loss=True currently supports picodet detection only; "
                "other tasks are not supported"
            )

    def build_validation_loss_adapter(self, model: torch.nn.Module):
        del model  # The validator's raw head output is what the criterion takes.
        from ..base.dense_head_validation_loss import DenseHeadValidationLoss

        return DenseHeadValidationLoss(
            self.build_criterion(distributed_normalize=False),
            num_classes=getattr(self.model.head, "num_classes", 80),
            device=self.device,
            family="picodet",
        )

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        cls_scores, bbox_preds = self.model(imgs)

        # Targets: (B, max_labels, 5) [class, cx, cy, w, h] in pixel coords,
        # zero-padded. Convert to per-image (gt_boxes_xyxy, gt_labels) lists.
        gt_boxes_list = []
        gt_labels_list = []
        for b in range(targets.shape[0]):
            t = targets[b]
            valid = (t[:, 3:5] > 0).all(dim=1)  # w > 0 and h > 0
            t = t[valid]
            if t.shape[0] == 0:
                gt_boxes_list.append(t.new_zeros((0, 4)))
                gt_labels_list.append(t.new_zeros((0,), dtype=torch.long))
                continue
            cls = t[:, 0].long()
            cx, cy, w, h = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
            x1 = cx - w * 0.5
            y1 = cy - h * 0.5
            x2 = cx + w * 0.5
            y2 = cy + h * 0.5
            gt_boxes_list.append(torch.stack([x1, y1, x2, y2], dim=-1))
            gt_labels_list.append(cls)

        return self._loss_fn(cls_scores, bbox_preds, gt_boxes_list, gt_labels_list)
