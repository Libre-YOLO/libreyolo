"""LibreDOMEDETR — BaseModel wrapper for the Dome-DETR tiny-object family."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.domedetr import postprocess
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import DOMEDETRValPreprocessor
from ..base import BaseModel
from .nn import DEFAULT_VARIANT, VARIANT_QUERY_BUDGET, LibreDOMEDETRModel
from .utils import preprocess_image, unwrap_domedetr_checkpoint

logger = logging.getLogger(__name__)

# The stride-4 encoder projection width is the cheapest size fingerprint:
# B0/B2/B4 stage-1 outputs are 64/96/128 channels.
_STEM_CHANNELS_TO_SIZE = {64: "s", 96: "m", 128: "l"}

# Published class counts per dataset variant. Used only as a fallback when a
# checkpoint carries no explicit variant marker.
_NC_TO_VARIANT = {9: "aitod", 12: "visdrone"}


class LibreDOMEDETR(BaseModel):
    """LibreYOLO wrapper for Dome-DETR (ACM MM 2025).

    A tiny-object specialist for aerial, drone and remote-sensing imagery, not
    a general-purpose detector. It is D-FINE plus three modules: DeFE predicts
    a density map, MWAS restricts encoder attention to occupied windows, and
    PAQI sizes the query set from that density instead of using a fixed 300.

    Scope notes that matter before reaching for this family:

    - **No COCO checkpoint exists.** Upstream publishes AI-TOD-V2 (9 classes)
      and VisDrone (12 classes) weights only, so canonical filenames always
      carry a dataset suffix (``LibreDOMEDETRs-visdrone.pt``) and ``names``
      comes from checkpoint metadata, never from a family constant.
    - **The advantage narrows as objects grow.** Upstream's own ablation moves
      AP-verytiny 14.0 -> 17.8 but AP-medium only 45.4 -> 46.4. It sits beside
      D-FINE rather than replacing it.
    - **Inference-only in LibreYOLO today.** Upstream ships an Apache-2.0
      training recipe, so a gated-experimental trainer is portable, but it
      needs the density-map criterion and a VisDrone convergence run that this
      port does not yet include.
    - **Weights are not rehosted.** The upstream model card states no license,
      so they are linked, not mirrored (the YOLO-NAS precedent).
    """

    FAMILY = "domedetr"
    FILENAME_PREFIX = "LibreDOMEDETR"
    INPUT_SIZES = {"s": 800, "m": 800, "l": 800}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TASK_INPUT_SIZES = {"detect": INPUT_SIZES}
    TRAIN_CONFIG = None
    WEIGHT_VARIANTS = ("aitod", "visdrone")
    val_preprocessor_class = DOMEDETRValPreprocessor
    TTA_FIXED_SIZE = True  # fixed square resize; multi-scale TTA is a no-op
    # PAQI's query count is data dependent (boolean masking + a greedy NMS
    # loop), so the forward has host syncs and a shape that changes per image.
    # That is exactly what CUDA graph capture cannot do.
    SUPPORTS_CUDA_GRAPH = False

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # DeFE lives only in Dome-DETR. Deliberately *not* keyed on
        # ``decoder.pre_bbox_head.``: Dome-DETR is a D-FINE derivative and
        # carries that key too, which is why this family must also register
        # ahead of LibreDFINE.
        return any(k.startswith("encoder.DeFE.") for k in weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        key = "encoder.input_proj.0.conv.weight"
        if key not in weights_dict:
            return None
        return _STEM_CHANNELS_TO_SIZE.get(int(weights_dict[key].shape[1]))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "decoder.enc_score_head.weight"
        if key not in weights_dict:
            return None
        return int(weights_dict[key].shape[0])

    @classmethod
    def detect_weight_variant(
        cls, weights_dict: dict, checkpoint: dict | None = None
    ) -> str:
        """Resolve the dataset variant, which sets the PAQI query budget.

        AI-TOD-V2 runs 300..1500 queries, VisDrone 250..500, so getting this
        wrong changes the proposal set rather than just a label. Prefer the
        explicit marker the converter writes; fall back to the class count.
        """
        if checkpoint:
            marker = checkpoint.get("weight_variant")
            if marker in VARIANT_QUERY_BUDGET:
                return marker

        nc = cls.detect_nb_classes(weights_dict)
        variant = _NC_TO_VARIANT.get(nc)
        if variant is None:
            logger.warning(
                "Dome-DETR checkpoint carries no weight_variant marker and nc=%s "
                "matches neither AI-TOD-V2 (9) nor VisDrone (12); defaulting to "
                "the %r query budget.",
                nc,
                DEFAULT_VARIANT,
            )
            return DEFAULT_VARIANT
        return variant

    def _init_model(self) -> nn.Module:
        return LibreDOMEDETRModel(
            config=self.size,
            nb_classes=self.nb_classes,
            variant=getattr(self, "weight_variant", DEFAULT_VARIANT),
            eval_spatial_size=(self.input_size, self.input_size),
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "backbone_stem": self.model.backbone.stem,
            "encoder": self.model.encoder,
            "encoder_input_proj": self.model.encoder.input_proj,
            "encoder_defe": self.model.encoder.DeFE,
            "encoder_mwas": self.model.encoder.mwas_processor,
            "encoder_fpn": self.model.encoder.fpn_blocks,
            "encoder_pan": self.model.encoder.pan_blocks,
            "decoder": self.model.decoder,
            "decoder_input_proj": self.model.decoder.input_proj,
            "dec_bbox_head": self.model.decoder.dec_bbox_head,
            "dec_score_head": self.model.decoder.dec_score_head,
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
        effective_size = input_size if input_size is not None else self.input_size
        return preprocess_image(image, input_size=effective_size, color_format=color_format)

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
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
            **kwargs,
        )

    def _strict_loading(self) -> bool:
        # Anchors and valid_mask are regenerated at forward time from
        # eval_spatial_size rather than restored, as in every DETR family here.
        return False

    @staticmethod
    def unwrap_checkpoint(checkpoint):
        return unwrap_domedetr_checkpoint(checkpoint)

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreDOMEDETR is inference-only in LibreYOLO. Upstream ships an "
            "Apache-2.0 training recipe (loss, matcher, density supervision), so "
            "a gated-experimental trainer is portable, but it is not wired here: "
            "it needs the DeFE density-map criterion plus a VisDrone convergence "
            "run before it could be trusted. Use LibreDFINE to train a detector "
            "on your own data."
        )
