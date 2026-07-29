"""Zero-shot classification validator for LibreCLIP.

Extends :class:`ClassifyValidator` in two ways:

* **CLIP preprocessing** — CLIP's own mean/std + bicubic resize (a 1.0 crop
  ratio), not the ImageNet stats the trained classifiers use. Getting this
  wrong silently lowers zero-shot accuracy.
* **Open-vocabulary indexing** — the model's display names are the humanized
  prompt labels (e.g. ``"tench"``), which deliberately differ from wnid folder
  names (``"n01440764"``). ``LibreCLIP.val`` calls ``set_classes`` on the
  train-split folder names *in sorted order*, so the model's logit index ``i``
  already lines up with the dataset's sorted-folder label ``i``. We therefore
  let the dataset drive the label indices (return ``None`` from
  ``_model_class_names``) instead of demanding a name match.
"""

from __future__ import annotations

from .classify_validator import ClassifyValidator


class CLIPClassifyValidator(ClassifyValidator):
    """Top-1/top-5 zero-shot validator with CLIP preprocessing."""

    def _model_class_names(self):
        # Indices come from the sorted train folders, which LibreCLIP.val
        # mirrored via set_classes; the humanized display names intentionally
        # differ from wnid folder names, so do not enforce a name match.
        return None

    def _resolve_class_names(self, train_classes: list[str]) -> list[str]:
        if not getattr(self.model, "frozen_classes", False):
            return train_classes

        from ..models.clip.labels import humanize_labels

        names = getattr(self.model, "names", None)
        if not isinstance(names, dict) or sorted(names) != list(range(len(names))):
            raise ValueError(
                "Frozen CLIP artifacts require contiguous ordered class names."
            )
        artifact_classes = [str(names[index]) for index in range(len(names))]
        dataset_classes = humanize_labels(train_classes)
        if artifact_classes != dataset_classes:
            raise ValueError(
                "Frozen CLIP class order does not match the dataset's humanized "
                f"train-folder order: artifact={artifact_classes}, "
                f"dataset={dataset_classes}. Re-export after set_classes() with "
                "this dataset's sorted train folders."
            )
        # The ImageFolder still indexes the raw directory names. Exact ordered
        # comparison above proves that these indices match the frozen text head.
        return train_classes

    def _dataset_transform_kwargs(self) -> dict:
        from torchvision.transforms import InterpolationMode

        from ..models.clip.model import CLIP_MEAN, CLIP_STD

        return {
            "mean": CLIP_MEAN,
            "std": CLIP_STD,
            "interpolation": InterpolationMode.BICUBIC,
            "crop_pct": 1.0,
        }
