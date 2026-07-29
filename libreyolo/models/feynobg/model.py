"""LibreFeyNobg: FeyNobg background removal (matte task).

FeyNobg (https://huggingface.co/feyninc/FeyNobg, code and weights Apache-2.0,
Copyright (c) 2026 Feyn Inc.) is a background-removal model built on BiRefNet:
the third Swin-L stage is deepened from 18 to 24 blocks (263M parameters) and
the model is retrained. It shares BiRefNet's forward path, preprocessing
(ImageNet-normalized fixed 1024x1024) and single-logit output contract, so the
architecture code is reused from the ``birefnet`` family; family identity,
checkpoint detection, and weights are FeyNobg's own.

Single released size ``l`` (Swin-L tier). Like its sibling family, the native
resolution is a fixed 1024x1024 (Swin relative-position tables are
resolution-tied) and the matte is resized back to the original canvas.

Inference-only, matching the matte v1 contract (ADR 0010).
"""

from __future__ import annotations

from typing import ClassVar, Dict, Optional

import torch.nn as nn

from ...postprocess.feynobg import postprocess as _feynobg_postprocess
from ..birefnet.model import LibreBiRefNet
from .nn import LibreFeyNobgModel


class LibreFeyNobg(LibreBiRefNet):
    """FeyNobg background removal: image -> soft alpha matte."""

    FAMILY = "feynobg"
    FILENAME_PREFIX = "LibreFeyNobg"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"l": 1024}

    _UPSTREAM_URL = "https://github.com/feyninc/nobg"

    # FeyNobg = BiRefNet keys plus a 24-block stage 3; the marker key for
    # block 24 (index 23) separates it from every birefnet checkpoint.
    _FEYNOBG_MARKER = "bb.layers.2.blocks.23.norm1.weight"
    _SWIN_L_EMBED_DIM = 192

    # ====================================================================
    # Checkpoint detection (disjoint from birefnet's by construction)
    # ====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        has_squeeze = any(k.startswith("squeeze_module.") for k in weights_dict)
        has_gdt_attn = any("gdt_convs_attn" in k for k in weights_dict)
        has_ipt = any(k.startswith("decoder.ipt_blk") for k in weights_dict)
        return (
            has_squeeze
            and has_gdt_attn
            and has_ipt
            and cls.detect_size(weights_dict) is not None
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        proj = weights_dict.get("bb.patch_embed.proj.weight")
        if proj is None or getattr(proj, "ndim", 0) != 4:
            return None
        if int(proj.shape[0]) != cls._SWIN_L_EMBED_DIM:
            return None
        return "l" if cls._FEYNOBG_MARKER in weights_dict else None

    # ====================================================================
    # Construction / inference
    # ====================================================================

    def _init_model(self) -> nn.Module:
        return LibreFeyNobgModel(size=self.size)

    def _postprocess(self, output, conf_thres, iou_thres, original_size, max_det=300, ratio=1.0, **kwargs):
        del conf_thres, iou_thres, max_det, ratio, kwargs
        return _feynobg_postprocess(output, original_size)

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training/fine-tuning LibreFeyNobg is not wired in this release "
            "(matte v1 is inference + val only). The upstream nobg library "
            "ships Apache-2.0 training code; to fine-tune today, train "
            f"upstream at {self._UPSTREAM_URL} and convert the result with "
            "weights/convert_feynobg_weights.py."
        )


__all__ = ["LibreFeyNobg"]
