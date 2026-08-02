"""Register DINO-DETR with the checkpoint-driven LibreYOLO factory."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from ...validation.preprocessors import DeformableDETRValPreprocessor
from ..base import BaseModel
from .nn import LibreDINODETRModel


class LibreDINODETR(BaseModel):
    """DINO, the 2022 DETR-lineage detector that introduced improved DN anchors."""

    FAMILY = "dinodetr"
    FILENAME_PREFIX = "LibreDINODETR"
    INPUT_SIZES = {"r50": 800, "r50s5": 800, "swinl": 800}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = None
    val_preprocessor_class = DeformableDETRValPreprocessor
    TTA_FIXED_SIZE = True

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Recognize DINO without claiming other DETR or DINO families."""
        tgt = weights_dict.get("transformer.tgt_embed.weight")
        return (
            "label_enc.weight" in weights_dict
            and isinstance(tgt, torch.Tensor)
            and tuple(tgt.shape) == (900, 256)
            and "transformer.enc_out_class_embed.weight" in weights_dict
            and "transformer.enc_out_bbox_embed.layers.2.weight" in weights_dict
            and "transformer.decoder.ref_point_head.layers.0.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Infer the R50 scale count or Swin-L backbone from tensor structure."""
        if not cls.can_load(weights_dict):
            return None
        levels = weights_dict.get("transformer.level_embed")
        if not isinstance(levels, torch.Tensor):
            return None
        if "backbone.0.patch_embed.proj.weight" in weights_dict:
            return "swinl" if int(levels.shape[0]) == 5 else None
        if "backbone.0.body.conv1.weight" in weights_dict:
            return {4: "r50", 5: "r50s5"}.get(int(levels.shape[0]))
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        head = weights_dict.get("class_embed.0.weight")
        if not isinstance(head, torch.Tensor):
            return None
        width = int(head.shape[0])
        return 80 if width == 91 else width

    def _init_model(self) -> nn.Module:
        architecture_classes = 91 if self.nb_classes == 80 else self.nb_classes
        return LibreDINODETRModel(size=self.size, nc=architecture_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "transformer": self.model.transformer,
            "class_embed": self.model.class_embed,
            "bbox_embed": self.model.bbox_embed,
        }

    @staticmethod
    def _get_preprocess_numpy():
        from ..deformable_detr.utils import preprocess_numpy

        return preprocess_numpy

    def _preprocess(self, image, color_format: str = "auto", input_size=None):
        from ..deformable_detr.utils import preprocess_image

        return preprocess_image(
            image,
            input_size=self.input_size if input_size is None else input_size,
            color_format=color_format,
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "DINO-DETR postprocessing is added only after upstream parity passes."
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "DINO-DETR is inference-only; contrastive denoising training is out of scope."
        )


__all__ = ["LibreDINODETR"]
