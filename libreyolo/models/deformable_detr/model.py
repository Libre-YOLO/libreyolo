"""LibreDeformableDETR BaseModel wrapper.

Deformable DETR (SenseTime, 2020) replaced DETR's dense cross-attention with
sparse multi-scale sampling and made transformer detectors practical to train.
This inference-only museum family preserves all five released ResNet-50
variants while using a portable pure-PyTorch attention implementation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput
from ...utils.serialization import load_untrusted_torch_file
from ...validation.preprocessors import RFDETRValPreprocessor
from ..base import BaseModel
from .nn import LibreDeformableDETRModel
from .utils import preprocess_image, unwrap_deformable_detr_checkpoint

_COCO91_HEAD_WIDTH = 91


class LibreDeformableDETR(BaseModel):
    """Standalone Deformable DETR, covering all five upstream R50 variants."""

    FAMILY = "deformable_detr"
    FILENAME_PREFIX = "LibreDeformableDETR"
    INPUT_SIZES = {
        "r50ss": 800,
        "r50ssdc5": 800,
        "r50": 800,
        "r50refine": 800,
        "r50twostage": 800,
    }
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = None
    val_preprocessor_class = RFDETRValPreprocessor
    TTA_FIXED_SIZE = True

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Recognize the original architecture without claiming its descendants."""
        return (
            "backbone.0.body.conv1.weight" in weights_dict
            and "transformer.encoder.layers.0.self_attn.sampling_offsets.weight"
            in weights_dict
            and "input_proj.0.0.weight" in weights_dict
            and "class_embed.0.weight" in weights_dict
            and any(key.startswith("bbox_embed.0.") for key in weights_dict)
        )

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        detected = super().detect_size_from_filename(filename)
        if detected is not None:
            return detected

        # Keep parent components: Hugging Face assets are commonly named only
        # ``model.safetensors`` and carry the variant in the repository folder.
        normalized = re.sub(r"[-_.]+", "-", str(filename).lower())
        aliases = (
            ("single-scale-dc5", "r50ssdc5"),
            ("iterative-bbox-refinement-plus-plus-two-stage", "r50twostage"),
            ("with-box-refine-two-stage", "r50twostage"),
            ("iterative-bbox-refinement", "r50refine"),
            ("with-box-refine", "r50refine"),
            ("single-scale", "r50ss"),
        )
        for marker, size in aliases:
            if marker in normalized:
                return size
        if "deformable-detr" in normalized:
            return "r50"
        return None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Infer structural variants; SS and SS-DC5 require metadata/filename."""
        level_embed = weights_dict.get("transformer.level_embed")
        if level_embed is None:
            return None
        num_levels = int(level_embed.shape[0])
        if num_levels == 1:
            # Dilation changes stride metadata, not parameter shapes or names.
            return None
        if num_levels != 4:
            return None
        if (
            "transformer.enc_output.weight" in weights_dict
            and "query_embed.weight" not in weights_dict
        ):
            return "r50twostage"

        first = weights_dict.get("class_embed.0.weight")
        second = weights_dict.get("class_embed.1.weight")
        if first is None:
            return None
        if second is None:
            return "r50"
        return "r50" if torch.equal(first, second) else "r50refine"

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        arch_nc = cls._arch_classes_from_state_dict(weights_dict)
        if arch_nc is None:
            return None
        return 80 if arch_nc == _COCO91_HEAD_WIDTH else arch_nc

    @staticmethod
    def _arch_classes_from_state_dict(weights_dict: dict) -> Optional[int]:
        weight = weights_dict.get("class_embed.0.weight")
        return None if weight is None else int(weight.shape[0])

    def __init__(
        self,
        model_path,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        arch_nc = self._peek_arch_classes(model_path)
        if arch_nc is None:
            arch_nc = _COCO91_HEAD_WIDTH if nb_classes == 80 else nb_classes
        self._arch_num_classes = arch_nc
        user_nb_classes = 80 if arch_nc == _COCO91_HEAD_WIDTH else arch_nc

        super().__init__(
            model_path=None,
            size=size,
            nb_classes=user_nb_classes,
            device=device,
            **kwargs,
        )

        self.model_path = model_path if isinstance(model_path, (str, Path)) else None
        if isinstance(model_path, dict):
            state_dict = self._prepare_state_dict(
                self._strip_ddp_prefix(unwrap_deformable_detr_checkpoint(model_path))
            )
            self.model.load_state_dict(state_dict, strict=self._strict_loading())
            self.model.to(self.device).eval()
        elif model_path is not None:
            self._load_weights(str(model_path))

    @staticmethod
    def _peek_arch_classes(model_path) -> Optional[int]:
        if isinstance(model_path, dict):
            return LibreDeformableDETR._arch_classes_from_state_dict(
                unwrap_deformable_detr_checkpoint(model_path)
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
                context="Deformable DETR head inspection",
            )
        except Exception:
            return None
        if not isinstance(loaded, dict):
            return None
        return LibreDeformableDETR._arch_classes_from_state_dict(
            unwrap_deformable_detr_checkpoint(loaded)
        )

    def _init_model(self) -> nn.Module:
        return LibreDeformableDETRModel(size=self.size, nc=self._arch_num_classes)

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        self._arch_num_classes = new_nb_classes
        super()._rebuild_for_new_classes(new_nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "transformer": self.model.transformer,
            "encoder": self.model.transformer.encoder,
            "decoder": self.model.transformer.decoder,
            "class_embed": self.model.class_embed,
            "bbox_embed": self.model.bbox_embed,
        }

    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy

        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        return preprocess_image(
            image,
            input_size=input_size if input_size is not None else self.input_size,
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
        raise NotImplementedError(
            "Deformable DETR postprocessing is added after the upstream parity gate."
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Deformable DETR is currently inference-only; its Hungarian-matching "
            "and focal-loss training recipe is not implemented."
        )

    def _strict_loading(self) -> bool:
        return True


__all__ = ["LibreDeformableDETR"]
