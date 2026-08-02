"""Register DINO-DETR with the checkpoint-driven LibreYOLO factory.

DINO was published by IDEA in 2022 as a DETR-lineage detector combining
contrastive denoising, mixed query selection, and improved anchor boxes. This
inference-only port preserves the three released detector configurations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from ...postprocess.dinodetr import postprocess
from ...utils.coco import COCO91_TO_COCO80
from ...utils.image_loader import ImageInput
from ...utils.serialization import load_untrusted_torch_file
from ...validation.preprocessors import DeformableDETRValPreprocessor
from ..base import BaseModel
from .nn import LibreDINODETRModel
from .utils import unwrap_dinodetr_checkpoint

_COCO91_HEAD_WIDTH = 91


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
        width = cls._arch_classes_from_state_dict(weights_dict)
        if width is None:
            return None
        return 80 if width == _COCO91_HEAD_WIDTH else width

    @staticmethod
    def _arch_classes_from_state_dict(weights_dict: dict) -> Optional[int]:
        head = weights_dict.get("class_embed.0.weight")
        return None if not isinstance(head, torch.Tensor) else int(head.shape[0])

    def __init__(
        self,
        model_path,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        architecture_classes = self._peek_arch_classes(model_path)
        if architecture_classes is None:
            architecture_classes = (
                _COCO91_HEAD_WIDTH if nb_classes == 80 else nb_classes
            )
        self._arch_num_classes = architecture_classes
        public_classes = (
            80 if architecture_classes == _COCO91_HEAD_WIDTH else architecture_classes
        )
        super().__init__(
            model_path=None,
            size=size,
            nb_classes=public_classes,
            device=device,
            **kwargs,
        )
        self.model_path = model_path if isinstance(model_path, (str, Path)) else None
        if isinstance(model_path, dict):
            state_dict = self._prepare_state_dict(
                self._strip_ddp_prefix(unwrap_dinodetr_checkpoint(model_path))
            )
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device).eval()
        elif model_path is not None:
            self._load_weights(str(model_path))

    @staticmethod
    def _peek_arch_classes(model_path) -> Optional[int]:
        if isinstance(model_path, dict):
            return LibreDINODETR._arch_classes_from_state_dict(
                unwrap_dinodetr_checkpoint(model_path)
            )
        if not isinstance(model_path, (str, Path)):
            return None
        path = Path(BaseModel._resolve_weights_path(str(model_path)))
        if not path.exists():
            return None
        try:
            loaded = load_untrusted_torch_file(
                str(path),
                map_location="cpu",
                context="DINO-DETR head inspection",
            )
        except Exception:
            return None
        if not isinstance(loaded, dict):
            return None
        return LibreDINODETR._arch_classes_from_state_dict(
            unwrap_dinodetr_checkpoint(loaded)
        )

    def _init_model(self) -> nn.Module:
        return LibreDINODETRModel(size=self.size, nc=self._arch_num_classes)

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        self._arch_num_classes = new_nb_classes
        super()._rebuild_for_new_classes(new_nb_classes)

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

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ):
        from ..deformable_detr.utils import preprocess_image

        return preprocess_image(
            image,
            input_size=self.input_size if input_size is None else input_size,
            color_format=color_format,
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        if isinstance(output, tuple):
            output = {"pred_logits": output[0], "pred_boxes": output[1]}
        class_map = (
            COCO91_TO_COCO80
            if self._arch_num_classes == _COCO91_HEAD_WIDTH and self.nb_classes == 80
            else None
        )
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=min(max_det, self.model.num_select),
            class_map=class_map,
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "DINO-DETR is inference-only; contrastive denoising training is out of scope."
        )


__all__ = ["LibreDINODETR"]
