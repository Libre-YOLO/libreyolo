"""LibreLWDETR — BaseModel wrapper for the LW-DETR detection family.

LW-DETR (arXiv 2406.03459, Baidu / Atten4Vis, 2024) showed a plain-ViT
encoder with a shallow DETR decoder beating the contemporary real-time YOLO
detectors on their own accuracy/latency curve, and is the direct ancestor of
RF-DETR — Roboflow forked and modified this architecture. LibreYOLO ships both.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...utils.coco import COCO91_TO_COCO80
from ...utils.image_loader import ImageInput
from ...utils.serialization import load_untrusted_torch_file
from ...validation.preprocessors import LWDETRValPreprocessor
from ..base import BaseModel
from .nn import LWDETR_CONFIGS, SIZE_DIVISOR, LibreLWDETRModel
from .utils import preprocess_image, unwrap_lwdetr_checkpoint
from ...postprocess.lwdetr import postprocess

# Width of the classification head on the released COCO checkpoints. Upstream
# trains on raw COCO annotations, so the head has one column per COCO category
# id (``max_obj_id + 1``) rather than one per annotated class.
_COCO91_HEAD_WIDTH = 91


class LibreLWDETR(BaseModel):
    """LW-DETR: plain-ViT encoder, multi-scale projector, deformable DETR decoder.

    Inference-only. Upstream trains with Group-DETR one-to-many supervision and
    an IoU-aware classification loss; that recipe is not wired here, so
    ``train()`` raises. Sizes ``t/s/m/l/x`` map to upstream
    tiny/small/medium/large/xlarge.
    """

    FAMILY = "lwdetr"
    FILENAME_PREFIX = "LibreLWDETR"
    INPUT_SIZES = {"t": 640, "s": 640, "m": 640, "l": 640, "x": 640}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = None  # inference-only
    val_preprocessor_class = LWDETRValPreprocessor
    TTA_FIXED_SIZE = True  # fixed square canvas; multi-scale TTA is a no-op

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Fingerprint LW-DETR's plain-ViT encoder.

        ``backbone.0.encoder.patch_embed.proj`` plus the CAE-v2 ``q_bias``
        parameter is unique to this family: RF-DETR — the closest relative,
        sharing the decoder and projector lineage — swapped the encoder for
        DINOv2 and nests its keys under ``backbone.0.encoder.encoder.*`` with
        no ``patch_embed.proj`` or ``q_bias`` anywhere.
        """
        has_patch_embed = "backbone.0.encoder.patch_embed.proj.weight" in weights_dict
        has_cae_bias = any(
            k.startswith("backbone.0.encoder.blocks.") and k.endswith(".attn.q_bias")
            for k in weights_dict
        )
        return has_patch_embed and has_cae_bias

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        detected = super().detect_size_from_filename(filename)
        if detected is not None:
            return detected
        m = re.search(
            r"lwdetr_(tiny|small|medium|large|xlarge)(?:_|\.|$)", filename.lower()
        )
        if m:
            return {
                "tiny": "t",
                "small": "s",
                "medium": "m",
                "large": "l",
                "xlarge": "x",
            }[m.group(1)]
        return None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Infer the size code from ViT width, depth, and projector levels.

        ``t``/``s`` share the 192-wide vit_tiny encoder and are separated by
        block count (6 vs 10); ``m`` is vit_small with one projector level,
        ``l`` is vit_small with two, ``x`` is vit_base.
        """
        patch_key = "backbone.0.encoder.patch_embed.proj.weight"
        if patch_key not in weights_dict:
            return None
        embed_dim = int(weights_dict[patch_key].shape[0])

        depth = 1 + max(
            (
                int(k.split(".")[4])
                for k in weights_dict
                if k.startswith("backbone.0.encoder.blocks.")
            ),
            default=-1,
        )
        num_levels = len(
            {
                k.split(".")[4]
                for k in weights_dict
                if k.startswith("backbone.0.projector.stages.")
            }
        )

        if embed_dim == 192:
            return "t" if depth == 6 else "s"
        if embed_dim == 384:
            return "m" if num_levels == 1 else "l"
        if embed_dim == 768:
            return "x"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        """Return the *user-facing* class count.

        The factory feeds this straight to the ``nb_classes`` constructor
        argument, and LibreYOLO exposes the contiguous COCO-80 interface for a
        91-wide COCO head.
        """
        arch_nc = cls._arch_classes_from_state_dict(weights_dict)
        if arch_nc is None:
            return None
        return 80 if arch_nc == _COCO91_HEAD_WIDTH else arch_nc

    @staticmethod
    def _arch_classes_from_state_dict(weights_dict: dict) -> Optional[int]:
        key = "class_embed.weight"
        if key not in weights_dict:
            return None
        return int(weights_dict[key].shape[0])

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def __init__(
        self,
        model_path,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        # Size the classification head from the checkpoint rather than guessing
        # from nb_classes: the released COCO weights carry a 91-column head
        # (one per COCO category id) behind an 80-class user interface.
        arch_nc = self._peek_arch_classes(model_path)
        if arch_nc is None:
            # No weights to inspect. The COCO default reproduces the released
            # architecture; any other class count builds a contiguous head.
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
                self._strip_ddp_prefix(unwrap_lwdetr_checkpoint(model_path))
            )
            self.model.load_state_dict(state_dict, strict=self._strict_loading())
            self.model.to(self.device).eval()
        elif model_path is not None:
            self._load_weights(str(model_path))

    @staticmethod
    def _peek_arch_classes(model_path) -> Optional[int]:
        """Read the head width from a checkpoint without building anything."""
        if isinstance(model_path, dict):
            return LibreLWDETR._arch_classes_from_state_dict(
                unwrap_lwdetr_checkpoint(model_path)
            )
        if not isinstance(model_path, (str, Path)):
            return None
        path = Path(BaseModel._resolve_weights_path(str(model_path)))
        if not path.exists():
            return None
        try:
            loaded = load_untrusted_torch_file(
                str(path), map_location="cpu", context="LW-DETR head inspection"
            )
        except Exception:
            return None
        if not isinstance(loaded, dict):
            return None
        return LibreLWDETR._arch_classes_from_state_dict(
            unwrap_lwdetr_checkpoint(loaded)
        )

    def _init_model(self) -> nn.Module:
        return LibreLWDETRModel(size=self.size, nc=self._arch_num_classes)

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        # A rebuild always means a fresh, contiguous head — the 91-column COCO
        # layout only exists in the released upstream weights.
        self._arch_num_classes = new_nb_classes
        super()._rebuild_for_new_classes(new_nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "encoder": self.model.backbone[0].encoder,
            "encoder_patch_embed": self.model.backbone[0].encoder.patch_embed,
            "encoder_blocks": self.model.backbone[0].encoder.blocks,
            "projector": self.model.backbone[0].projector,
            "transformer": self.model.transformer,
            "decoder": self.model.transformer.decoder,
            "decoder_layers": self.model.transformer.decoder.layers,
            "class_embed": self.model.class_embed,
            "bbox_embed": self.model.bbox_embed,
        }

    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy

        return preprocess_numpy

    def _validate_imgsz(self, imgsz: int, *, name: str = "imgsz") -> int:
        """The ViT tiles the patch grid 4x4, so sides must be multiples of 64."""
        imgsz = int(imgsz)
        if imgsz <= 0 or imgsz % SIZE_DIVISOR != 0:
            raise ValueError(
                f"LW-DETR {name} must be a positive multiple of {SIZE_DIVISOR} "
                f"(the ViT splits the patch grid into 4x4 windows), got {imgsz}."
            )
        return imgsz

    def _get_val_preprocessor(self, img_size: int | None = None):
        if img_size is not None:
            img_size = self._validate_imgsz(img_size, name="validation imgsz")
        return super()._get_val_preprocessor(img_size=img_size)

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        effective_size = self._validate_imgsz(
            input_size if input_size is not None else self.input_size,
            name="inference imgsz",
        )
        return preprocess_image(
            image, input_size=effective_size, color_format=color_format
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
            # Exported graphs emit (pred_logits, pred_boxes) in that order.
            output = {"pred_logits": output[0], "pred_boxes": output[1]}

        # Upstream never returns more than ``num_select`` detections; honour a
        # smaller user budget but never exceed the model's own top-k.
        effective_max_det = min(max_det, self.model.num_select)

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
            max_det=effective_max_det,
            class_map=class_map,
        )

    def train(self, *args, **kwargs):
        """Not implemented — LW-DETR ships as an inference-only exhibit.

        Reproducing upstream means Group-DETR one-to-many supervision (13 query
        groups), an IoU-aware BCE classification loss, and the two-stage encoder
        losses. None of that is wired, and a trainer that silently used the
        generic DETR recipe would quietly under-train the model.
        """
        raise NotImplementedError(
            "LW-DETR is shipped inference-only. Its Group-DETR one-to-many "
            "training recipe (group_detr=13 query groups, IoU-aware "
            "classification loss, two-stage encoder losses) is not implemented. "
            "Use LibreRFDETR — LW-DETR's descendant — for a trainable model of "
            "the same lineage, or LibreYOLO9 / LibreDFINE."
        )

    def _strict_loading(self) -> bool:
        return True
