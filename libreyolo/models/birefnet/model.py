"""LibreBiRefNet: background removal / dichotomous image segmentation (matte task).

BiRefNet (https://github.com/ZhengPeng7/BiRefNet, MIT) predicts a soft alpha
matte separating foreground subject from background. That is exactly LibreYOLO's
``matte`` task primitive (``Results.matte``, ``(H, W)`` float in ``[0, 1]`` on
the original canvas), so this family reuses the shared matte Results, validator,
and checkpoint schema: ``predict`` (and ``cutout`` / transparent-PNG ``save``)
and zero-shot ``val`` (MAE / structure) work out of the box.

Sizes: ``t`` = BiRefNet_lite (Swin-T tier, fast), ``l`` = BiRefNet general
(Swin-L tier, quality default). Both run at a fixed native 1024x1024 (Swin
relative-position tables are resolution-tied; a non-native resolution silently
interpolates them badly), and the matte is resized back to the original canvas.

The sibling ``feynobg`` family (LibreFeyNobg) shares this architecture with a
deeper stage 3; its checkpoints carry 24 stage-3 blocks and are rejected here
so each family claims only its own weights.

Inference-only in v1. Fine-tuning is a documented follow-up (see :meth:`train`);
the paired-data schema is specced in ``docs/dataset_schema.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.birefnet import postprocess as _birefnet_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .nn import LibreBiRefNetModel
from .utils import preprocess_image, preprocess_numpy


class LibreBiRefNet(BaseModel):
    """BiRefNet background removal: image -> soft alpha matte."""

    FAMILY = "birefnet"
    FILENAME_PREFIX = "LibreBiRefNet"
    # Capture covers the encoder only; the deformable decoder runs eagerly.
    # See _get_graph_runner below.
    SUPPORTS_CUDA_GRAPH = True
    WEIGHT_EXT = ".pt"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"t": 1024, "l": 1024}
    SUPPORTED_TASKS = ("matte",)
    DEFAULT_TASK = "matte"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None  # inference-only in v1
    SUPPORTS_BATCHED_PREDICT = True
    TTA_ENABLED = False

    _UPSTREAM_URL = "https://github.com/ZhengPeng7/BiRefNet"

    # The Swin patch-embed width identifies the backbone tier. FeyNobg shares
    # the Swin-L width (192) but has 24 stage-3 blocks; its marker key makes
    # detect_size return None so the feynobg family claims those checkpoints.
    _EMBED_DIM_TO_SIZE: ClassVar[Dict[int, str]] = {96: "t", 192: "l"}
    _FEYNOBG_MARKER = "bb.layers.2.blocks.23.norm1.weight"

    # ====================================================================
    # Checkpoint detection
    # ====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # BiRefNet-unique tokens: the squeeze module, the bilateral-reference
        # gradient-attention branch, and the image-patch injection blocks. No
        # other family combines these three prefixes.
        has_squeeze = any(k.startswith("squeeze_module.") for k in weights_dict)
        has_gdt_attn = any("gdt_convs_attn" in k for k in weights_dict)
        has_ipt = any(k.startswith("decoder.ipt_blk") for k in weights_dict)
        return has_squeeze and has_gdt_attn and has_ipt and cls.detect_size(weights_dict) is not None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        proj = weights_dict.get("bb.patch_embed.proj.weight")
        if proj is None or getattr(proj, "ndim", 0) != 4:
            return None
        if cls._FEYNOBG_MARKER in weights_dict:
            return None  # FeyNobg checkpoint: belongs to the feynobg family
        return cls._EMBED_DIM_TO_SIZE.get(int(proj.shape[0]))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # A soft matte has no classes; the single slot is a schema placeholder.
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "matte" if cls.can_load(state_dict) else None

    # ====================================================================
    # Construction
    # ====================================================================

    def __init__(
        self,
        model_path=None,
        size: str = "l",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,  # always 1 ("matte"); ignore any caller-provided value
            device=device,
            task=task,
            **kwargs,
        )
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        # Enforce the single matte slot after any loading path, matching the
        # depth/restore precedent ({0: "matte"} instead of a class-name fallback).
        self.nb_classes = 1
        self.names = {0: "matte"}
        self.model.eval()

    def _init_model(self) -> nn.Module:
        return LibreBiRefNetModel(size=self.size)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "bb": self.model.bb,
            "squeeze_module": self.model.squeeze_module,
            "decoder": self.model.decoder,
        }

    # ====================================================================
    # Inference surface
    # ====================================================================

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        res = input_size if input_size is not None else self.input_size
        return preprocess_image(image, input_size=res, color_format=color_format)

    # The decoder's deformable ASPP blocks call torchvision's deform_conv2d,
    # whose CUDA kernel replays to a different result under graph capture (it
    # is a compiled extension, so there is nothing to shim from here). The
    # encoder in front of it captures bit-identically and is the bulk of the
    # work, so capture stops there and the decoder runs eagerly on the replayed
    # features. Results are identical to the fully eager path.
    GRAPH_DISPATCH_IN_FORWARD = True

    def _get_graph_runner(self):
        if getattr(self, "_graph_runner", None) is None:
            from ..base.cuda_graph import GraphRunner

            self._graph_runner = GraphRunner(
                forward_fn=lambda x: self.model.forward_enc(x),
                family=self.family,
            )
        return self._graph_runner

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        mode = getattr(self, "_cuda_graph_mode", None)
        if mode is None:
            return self.model(input_tensor)
        # forward_enc/forward_dec are called as plain methods, so the root
        # __call__ hooks that implement the float32 I/O contract for
        # half-precision checkpoints (fp16 cast recipe, fp16-remainder
        # finalized quant) never fire; apply the same contract at the seam.
        param_dtype = next(self.model.parameters()).dtype
        if input_tensor.dtype == torch.float32 and param_dtype != torch.float32:
            input_tensor = input_tensor.to(param_dtype)
        feats = self._get_graph_runner().run(input_tensor, auto=(mode == "auto"))
        out = self.model.forward_dec(input_tensor, feats)
        return out.float() if out.dtype != torch.float32 else out

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        del conf_thres, iou_thres, max_det, ratio, kwargs
        return _birefnet_postprocess(output, original_size)

    # ====================================================================
    # Out of scope (v1)
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training/fine-tuning LibreBiRefNet is not wired in this release "
            "(matte v1 is inference + val only). The upstream repo ships MIT "
            "training code and LibreYOLO documents the paired image/matte dataset "
            "layout in docs/dataset_schema.md; fine-tuning support is a planned "
            f"follow-up. To fine-tune today, train upstream at {self._UPSTREAM_URL} "
            "and convert the result with weights/convert_birefnet_weights.py."
        )


__all__ = ["LibreBiRefNet"]
