"""Semantic-segmentation validator for LibreYOLO.

Computes mean IoU and pixel accuracy from a confusion matrix accumulated over
a dense-mask validation split, reusing the :class:`BaseValidator` template
(setup -> iterate -> finalize). Predictions and targets are compared at the
model input resolution; ignore pixels (255 by default) are excluded.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.semantic_dataset import (
    IGNORE_INDEX,
    SemanticDataset,
    resolve_semantic_data,
    semantic_collate_fn,
)
from .base import BaseValidator
from .contracts import (
    require_class_ids,
    require_finite,
    require_matching_batch_sizes,
)

logger = logging.getLogger(__name__)


class SemanticValidator(BaseValidator):
    """mIoU / pixel-accuracy validator for the semantic task."""

    task = "semantic"

    def _setup_dataloader(self) -> DataLoader:
        if not self.config.data:
            raise ValueError("Semantic validation requires data= (a dataset YAML).")
        data_config = resolve_semantic_data(
            self.config.data,
            allow_scripts=getattr(self.config, "allow_download_scripts", False),
        )
        split = self.config.split or "val"

        divisor = getattr(self.model, "semantic_imgsz_divisor", None)
        if divisor and self.config.imgsz % int(divisor):
            raise ValueError(
                f"Semantic validation imgsz={self.config.imgsz} must be "
                f"divisible by {int(divisor)} for this model family."
            )
        resize_mode = getattr(self.model, "semantic_resize_mode", "letterbox")
        self._native_preprocess = resize_mode == "native"
        dataset = SemanticDataset(
            data_config,
            split=split,
            imgsz=self.config.imgsz,
            augment=False,
            resize_mode=resize_mode,
        )

        model_nc = getattr(self.model, "nb_classes", None)
        if model_nc is not None and int(model_nc) != dataset.nc:
            raise ValueError(
                f"Semantic dataset has {dataset.nc} classes but the model "
                f"predicts {int(model_nc)}. Use a matching dataset/checkpoint."
            )

        self._num_classes = dataset.nc
        self._class_names = dict(dataset.names)
        self._ignore_index = dataset.ignore_index
        batch_size = self.config.batch_size
        if self._native_preprocess:
            if batch_size != 1:
                logger.info(
                    "Native-geometry semantic validation runs one image at a "
                    "time (batch_size=%d ignored).",
                    batch_size,
                )
            batch_size = 1
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=semantic_collate_fn,
        )

    def _init_metrics(self) -> None:
        nc = getattr(self, "_num_classes", 0)
        self._confusion = torch.zeros((nc, nc), dtype=torch.int64)

    def _preprocess_batch(self, batch: Any) -> tuple:
        images, targets, img_info, img_ids = batch
        if getattr(self, "_native_preprocess", False):
            if len(img_info) != 1:
                raise ValueError(
                    "Native-geometry semantic validation requires exactly one "
                    "image per batch."
                )
            info = img_info[0]
            tensor, _, original_size, ratio = self.model._preprocess(
                info["img_path"],
                color_format="rgb",
                input_size=self.config.imgsz,
            )
            expected_size = (int(info["orig_shape"][1]), int(info["orig_shape"][0]))
            if tuple(original_size) != expected_size:
                raise ValueError(
                    "Semantic native preprocessing changed the source canvas: "
                    f"reported {tuple(original_size)}, expected {expected_size}."
                )
            self._native_original_size = tuple(original_size)
            self._native_ratio = float(ratio)
            images = tensor
        return images, targets, img_info, img_ids

    def _postprocess_predictions(self, preds: Any, batch: Any) -> Any:
        """Decode raw model output into ``[B, H, W]`` class maps."""
        if getattr(self, "_native_preprocess", False):
            decoded = self.model._postprocess(
                preds,
                conf_thres=self.config.conf_thres,
                iou_thres=self.config.iou_thres,
                original_size=self._native_original_size,
                ratio=self._native_ratio,
                input_size=self.config.imgsz,
            )
            predictions = (
                decoded.get("semantic") if isinstance(decoded, dict) else decoded
            )
            predictions = torch.as_tensor(predictions)
            if predictions.ndim == 2:
                predictions = predictions.unsqueeze(0)
            targets = torch.as_tensor(batch[1])
            if predictions.ndim != 3:
                raise ValueError(
                    "Native semantic postprocessing must return [H, W] or "
                    f"[B, H, W], got {tuple(predictions.shape)}."
                )
            require_matching_batch_sizes(
                "Semantic validation", predictions=predictions, targets=targets
            )
            if predictions.shape != targets.shape:
                raise ValueError(
                    "Native semantic prediction and target canvases must match, "
                    f"got {tuple(predictions.shape)} and {tuple(targets.shape)}."
                )
            require_class_ids(
                predictions, self._num_classes, "Semantic validation predictions"
            )
            return predictions

        logits = preds
        if isinstance(logits, dict):
            logits = logits.get("semantic_logits", logits.get("logits"))
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        logits = torch.as_tensor(logits)

        targets = torch.as_tensor(batch[1])
        if targets.ndim != 3:
            raise ValueError(
                "Semantic validation expects [B, H, W] targets, got shape "
                f"{tuple(targets.shape)}."
            )
        target_hw = tuple(targets.shape[-2:])
        if logits.ndim != 4:
            raise ValueError(
                f"Semantic validation expects [B, C, H, W] logits, got shape "
                f"{tuple(logits.shape)}."
            )
        if logits.shape[1] != self._num_classes:
            raise ValueError(
                "Semantic validation class count mismatch: logits have "
                f"{logits.shape[1]} classes but the dataset has "
                f"{self._num_classes}."
            )
        require_matching_batch_sizes(
            "Semantic validation", predictions=logits, targets=targets
        )
        if tuple(logits.shape[-2:]) != target_hw:
            logits = F.interpolate(
                logits.float(), size=target_hw, mode="bilinear", align_corners=False
            )
        valid = targets != self._ignore_index
        require_class_ids(
            targets[valid], self._num_classes, "Semantic validation targets"
        )
        require_finite(
            logits,
            "Semantic validation logits",
            where=valid.unsqueeze(1).expand_as(logits),
        )
        return logits.argmax(dim=1)

    def _update_metrics(
        self, preds: Any, targets: Any, img_info: Any, img_ids: Any = None
    ) -> None:
        pred_maps = torch.as_tensor(preds).detach().cpu()
        target_maps = torch.as_tensor(targets).detach().cpu()
        if pred_maps.ndim != 3 or target_maps.ndim != 3:
            raise ValueError(
                "Semantic validation metrics expect [B, H, W] prediction and "
                f"target maps, got {tuple(pred_maps.shape)} and "
                f"{tuple(target_maps.shape)}."
            )
        require_matching_batch_sizes(
            "Semantic validation", predictions=pred_maps, targets=target_maps
        )
        if pred_maps.shape != target_maps.shape:
            raise ValueError(
                "Semantic validation prediction and target shapes must match, got "
                f"{tuple(pred_maps.shape)} and {tuple(target_maps.shape)}."
            )

        pred_maps = pred_maps.reshape(-1)
        target_maps = target_maps.reshape(-1)

        valid = target_maps != self._ignore_index
        if not bool(valid.any()):
            return
        target_valid = require_class_ids(
            target_maps[valid], self._num_classes, "Semantic validation targets"
        )
        pred_valid = require_class_ids(
            pred_maps[valid], self._num_classes, "Semantic validation predictions"
        )

        index = target_valid * self._num_classes + pred_valid
        counts = torch.bincount(index, minlength=self._num_classes**2)
        self._confusion += counts.reshape(self._num_classes, self._num_classes)

    def _per_class_iou(self) -> torch.Tensor:
        confusion = self._confusion.double()
        true_positive = confusion.diag()
        union = confusion.sum(dim=0) + confusion.sum(dim=1) - true_positive
        iou = torch.full_like(true_positive, float("nan"))
        present = union > 0
        iou[present] = true_positive[present] / union[present]
        return iou

    def _compute_metrics(self) -> Dict[str, float]:
        total = self._confusion.sum()
        iou = self._per_class_iou()
        observed = ~torch.isnan(iou)
        miou = float(iou[observed].mean()) if bool(observed.any()) else 0.0
        accuracy = float(self._confusion.diag().sum() / total) if total > 0 else 0.0
        return {
            "metrics/mIoU": miou,
            "metrics/pixel_accuracy": accuracy,
            "fitness": miou,
        }

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 50)
        logger.info("Semantic Segmentation Validation Results")
        logger.info("=" * 50)
        iou = self._per_class_iou()
        for class_id in range(self._num_classes):
            value = iou[class_id]
            if torch.isnan(value):
                continue
            name = self._class_names.get(class_id, str(class_id))
            logger.info("  IoU %-20s %.4f", name, float(value))
        logger.info("  mIoU:           %.4f", metrics.get("metrics/mIoU", 0.0))
        logger.info(
            "  pixel accuracy: %.4f", metrics.get("metrics/pixel_accuracy", 0.0)
        )
        logger.info("=" * 50)


__all__ = ["SemanticValidator", "IGNORE_INDEX"]
