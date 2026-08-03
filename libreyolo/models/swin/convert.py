"""Microsoft and timm Swin V1 checkpoint key normalization.

The original Microsoft checkpoints store the classifier directly as
``head.weight`` and attach each patch-merging layer to the preceding stage.
The shared LibreYOLO tower follows timm's current layout: ``head.fc`` and the
same patch-merging tensors on the following stage. The remap below is purely
syntactic and drops only regenerated attention buffers.
"""

from __future__ import annotations

import re
from typing import Dict

import torch

_GENERATED_BUFFER_TOKENS = ("relative_position_index", "attn_mask")


def convert_upstream(state_dict: dict) -> Dict[str, torch.Tensor]:
    """Return native Swin keys for a released Microsoft or timm state dict."""
    old_layout = "head.weight" in state_dict and "head.fc.weight" not in state_dict
    converted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if any(token in key for token in _GENERATED_BUFFER_TOKENS):
            continue
        new_key = key
        if old_layout:
            new_key = re.sub(
                r"^layers\.(\d+)\.downsample",
                lambda match: f"layers.{int(match.group(1)) + 1}.downsample",
                new_key,
            )
            if new_key.startswith("head."):
                new_key = "head.fc." + new_key[len("head.") :]
        converted[new_key] = value
    return converted


def is_upstream_state_dict(state_dict: dict) -> bool:
    """Recognize only Swin V1 classifier layouts, not shared backbones/V2."""
    converted = convert_upstream(state_dict)
    patch = converted.get("patch_embed.proj.weight")
    head = converted.get("head.fc.weight")
    bias = converted.get(
        "layers.0.blocks.0.attn.relative_position_bias_table"
    )
    return bool(
        patch is not None
        and patch.ndim == 4
        and tuple(patch.shape[1:]) == (3, 4, 4)
        and head is not None
        and head.ndim == 2
        and bias is not None
        and bias.ndim == 2
        and int(bias.shape[0]) == 169
        and not any(
            "cpb_mlp" in key or key.endswith("attn.logit_scale")
            for key in converted
        )
    )


__all__ = ["convert_upstream", "is_upstream_state_dict"]
