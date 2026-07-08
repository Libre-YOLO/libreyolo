"""Zero-shot classification validator for LibreSigLIP2.

Extends :class:`ClassifyValidator` like :class:`CLIPClassifyValidator`, but with
SigLIP's preprocessing:

* **SigLIP preprocessing** - symmetric [-1, 1] normalization (mean/std 0.5) and a
  square bilinear resize to the native resolution (no aspect-preserving resize +
  center crop). Getting this wrong silently lowers zero-shot accuracy.
* **Open-vocabulary indexing** - the model's display names are the humanized
  prompt labels (e.g. ``"tench"``), which deliberately differ from wnid folder
  names. ``LibreSigLIP2.val`` calls ``set_classes`` on the train-split folder
  names *in sorted order*, so the model's logit index ``i`` already lines up with
  the dataset's sorted-folder label ``i``; we let the dataset drive the label
  indices (return ``None`` from ``_model_class_names``).
"""

from __future__ import annotations

from .classify_validator import ClassifyValidator


class SigLIP2ClassifyValidator(ClassifyValidator):
    """Top-1/top-5 zero-shot validator with SigLIP preprocessing."""

    def _model_class_names(self):
        # Indices come from the sorted train folders, which LibreSigLIP2.val
        # mirrored via set_classes; the humanized display names intentionally
        # differ from wnid folder names, so do not enforce a name match.
        return None

    def _dataset_transform_kwargs(self) -> dict:
        from torchvision.transforms import InterpolationMode

        from ..models.siglip2.model import SIGLIP_MEAN, SIGLIP_STD

        return {
            "mean": SIGLIP_MEAN,
            "std": SIGLIP_STD,
            "interpolation": InterpolationMode.BILINEAR,
            "crop_pct": 1.0,
            "square_resize": True,
        }
