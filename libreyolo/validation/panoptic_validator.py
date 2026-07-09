"""Panoptic-segmentation validator for LibreYOLO.

Scores a panoptic model with Panoptic Quality (PQ = SQ x RQ) over a
COCO-panoptic split, reusing the :class:`BaseValidator` template
(setup -> iterate -> finalize).

Panoptic Quality is defined at the ground-truth resolution, and each model
family owns its own preprocessing geometry (EoMT pads COCO inputs but splits
ADE20K ones into sliding windows). So this validator delegates both ends to the
model: :meth:`_preprocess_batch` calls the family's ``_preprocess`` and
:meth:`_postprocess_predictions` calls its ``_postprocess``, which returns a
segment-id map already resized onto the original canvas. Images are therefore
processed one at a time.

The metric itself lives in :mod:`libreyolo.validation.panoptic_quality`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from torch.utils.data import DataLoader

from ..data.panoptic_dataset import (
    PanopticDataset,
    panoptic_collate_fn,
    resolve_panoptic_data,
)
from .base import BaseValidator
from .panoptic_quality import PQ_IOU_THRESHOLD, PanopticQuality

logger = logging.getLogger(__name__)


class PanopticValidator(BaseValidator):
    """Panoptic Quality (PQ = SQ x RQ) validator for the ``panoptic`` task."""

    task = "panoptic"

    # Populated by _setup_dataloader; defaulted so _init_metrics is never a
    # surprise AttributeError when the hooks are exercised out of order.
    _thing_class_ids: set = frozenset()

    def _setup_dataloader(self) -> DataLoader:
        if not self.config.data:
            raise ValueError("Panoptic validation requires data= (a dataset YAML).")
        data_config = resolve_panoptic_data(
            self.config.data,
            allow_scripts=getattr(self.config, "allow_download_scripts", False),
        )
        split = self.config.split or "val"
        dataset = PanopticDataset(data_config, split=split)

        model_nc = getattr(self.model, "nb_classes", None)
        if model_nc is not None and int(model_nc) != dataset.nc:
            raise ValueError(
                f"Panoptic dataset has {dataset.nc} classes but the model predicts "
                f"{int(model_nc)}. Use a matching dataset/checkpoint."
            )

        # A thing/stuff split that disagrees with the checkpoint would not crash;
        # it would quietly mis-partition PQ_things / PQ_stuff. Fail loudly instead.
        model_things = getattr(self.model, "thing_class_ids", None)
        if model_things and set(int(i) for i in model_things) != dataset.thing_train_ids:
            raise ValueError(
                "The checkpoint's thing_class_ids disagree with the dataset's "
                "'isthing' categories. PQ_things / PQ_stuff would be mis-split. "
                "Check that the dataset YAML 'names' order matches the checkpoint."
            )

        self._thing_class_ids = dataset.thing_train_ids
        self._class_names = dict(dataset.names)

        if self.config.batch_size != 1:
            logger.info(
                "Panoptic validation runs one image at a time (batch_size=%d ignored); "
                "segment maps have per-image resolutions.",
                self.config.batch_size,
            )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=False,
            collate_fn=panoptic_collate_fn,
        )

    def _init_metrics(self) -> None:
        self._pq = PanopticQuality(thing_class_ids=set(self._thing_class_ids))

    def _preprocess_batch(self, batch: Any) -> tuple:
        img_paths, seg_maps, segments_infos, infos = batch
        # Let the model apply its own resize/pad/split geometry.
        tensor, _, original_size, _ = self.model._preprocess(img_paths[0])
        self._original_size = original_size
        # The 4-tuple carries segments_info in the img_ids slot.
        return tensor, seg_maps, infos, segments_infos

    def _postprocess_predictions(self, preds: Any, batch: Any) -> Any:
        """Ask the model for a segment-id map on the original image canvas."""
        detections = self.model._postprocess(
            preds,
            self.config.conf_thres,
            self.config.iou_thres,
            self._original_size,
        )
        if not isinstance(detections, dict) or "panoptic" not in detections:
            raise ValueError(
                f"{type(self.model).__name__}._postprocess did not return a "
                "'panoptic' segment-id map. Does this family implement the "
                "panoptic task?"
            )
        return detections

    def _update_metrics(
        self, preds: Any, targets: Any, img_info: Any, img_ids: Any = None
    ) -> None:
        gt_map = targets[0].numpy()
        gt_segments = img_ids[0]
        pred_map = preds["panoptic"].cpu().numpy()
        pred_segments = preds.get("segments_info") or []
        self._pq.update(gt_map, gt_segments, pred_map, pred_segments)

    def _compute_metrics(self) -> Dict[str, float]:
        return self._pq.compute()

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 50)
        logger.info("LibreYOLO Panoptic Segmentation Validation Results")
        logger.info("=" * 50)
        logger.info("  PQ:         %.4f", metrics.get("metrics/PQ", 0.0))
        logger.info("  SQ:         %.4f", metrics.get("metrics/SQ", 0.0))
        logger.info("  RQ:         %.4f", metrics.get("metrics/RQ", 0.0))
        logger.info("  PQ things:  %.4f", metrics.get("metrics/PQ_things", 0.0))
        logger.info("  PQ stuff:   %.4f", metrics.get("metrics/PQ_stuff", 0.0))
        logger.info("  categories: %d", int(metrics.get("metrics/categories", 0)))
        logger.info("=" * 50)


__all__ = ["PanopticValidator", "PQ_IOU_THRESHOLD"]
