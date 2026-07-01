"""Native RF-DETR-SO network assembly for LibreYOLO.

RF-DETR-SO ("small object") is RF-DETR with three architectural changes
implemented clean-room from the TinyFormer paper (arXiv 2605.25046):

1. a 3-level feature pyramid (``projector_scale=("P3", "P4", "P5")``,
   strides 8/16/32) instead of the single stride-16 level used by every
   stock detection size,
2. an SSA branch: stride-2 convolutions on the raw image whose stride-8
   output is fused into the P3 level (deconv-upsampled ViT features alone
   cannot recover detail lost at tokenization),
3. a PBM neck: two parallel bi-fusion blocks refining P3 and P4 with their
   neighboring levels before the decoder.

Everything downstream (deformable decoder, two-stage query selection,
criterion, postprocess) is the stock RF-DETR code path, which is
level-count agnostic; the decoder's deformable cross-attention simply gets
``n_levels=3``.

Stock RF-DETR-S detect checkpoints transfer via :func:`remap_stock_state_for_so`:
the projector's P4 stage shifts from index 0 to index 1, and the decoder's
deformable-attention ``sampling_offsets`` / ``attention_weights`` tensors get
their per-level axis replicated from 1 to 3 levels, so every new level starts
with the sampling pattern the model learned for its single level. New modules
(SDE, fusion, PBM, projector P3/P5 stages) start fresh.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn

from ..rfdetr.lwdetr import PostProcess, build_model
from ..rfdetr.nn import (
    RFDETR_CONFIGS,
    RFDETRSizeConfig,
    LibreRFDETRModel,
    _make_args,
    _unwrap_state_dict,
)
from .backbone import BackboneSO

RFDETRSO_CONFIGS: dict[str, RFDETRSizeConfig] = {
    "s": replace(
        RFDETR_CONFIGS["s"],
        projector_scale=("P3", "P4", "P5"),
    ),
}

# SDE base channel per size (TinyFormer scales this with model capacity).
SSA_BASE_CHANNELS: dict[str, int] = {"s": 32}

# Index of the P4 (identity-scale) stage inside the 3-level projector.
_P4_STAGE_INDEX = 1

_PROJECTOR_STAGE_RE = re.compile(
    r"^(backbone\.0\.projector\.stages(?:_sampling)?)\.0\.(.*)$"
)
_SO_KEY_PREFIXES = (
    "backbone.0.ssa_sde.",
    "backbone.0.ssa_fuse.",
    "backbone.0.pbm3.",
    "backbone.0.pbm4.",
)


def is_so_state_dict(state_dict: dict[str, Any]) -> bool:
    """Return True when the tensor dict carries RF-DETR-SO module keys."""
    return any(key.startswith(_SO_KEY_PREFIXES) for key in state_dict)


def _expand_deformable_level_axis(
    tensor: torch.Tensor,
    n_heads: int,
    n_points: int,
    trailing: int,
    num_levels: int,
) -> torch.Tensor:
    """Replicate the per-level axis of a deformable-attention parameter.

    ``sampling_offsets`` tensors factor as (heads, levels, points, 2[, dim]);
    ``attention_weights`` as (heads, levels, points[, dim]). ``trailing`` is
    2 for offsets and 1 for weights. Tensors already sized for
    ``num_levels`` pass through unchanged.
    """
    per_level = n_heads * n_points * trailing
    rows = tensor.shape[0]
    if rows == per_level * num_levels:
        return tensor
    if rows != per_level:
        return tensor
    tail = tensor.shape[1:]
    factored = tensor.reshape(n_heads, 1, n_points * trailing, *tail)
    expanded = factored.repeat(1, num_levels, *([1] * (factored.dim() - 2)))
    return expanded.reshape(per_level * num_levels, *tail)


def remap_stock_state_for_so(
    state_dict: dict[str, torch.Tensor],
    *,
    ca_nheads: int,
    dec_n_points: int,
    num_levels: int = 3,
) -> dict[str, torch.Tensor]:
    """Remap a stock (single-level) RF-DETR tensor dict onto the SO layout."""
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        match = _PROJECTOR_STAGE_RE.match(key)
        if match:
            key = f"{match.group(1)}.{_P4_STAGE_INDEX}.{match.group(2)}"
        elif ".cross_attn.sampling_offsets." in key:
            value = _expand_deformable_level_axis(
                value, ca_nheads, dec_n_points, 2, num_levels
            )
        elif ".cross_attn.attention_weights." in key:
            value = _expand_deformable_level_axis(
                value, ca_nheads, dec_n_points, 1, num_levels
            )
        remapped[key] = value
    return remapped


class LibreRFDETRSOModel(LibreRFDETRModel):
    """RF-DETR-SO model built from LibreYOLO-local RF-DETR modules."""

    def __init__(
        self,
        config: str = "s",
        nb_classes: int = 80,
        device: str = "cpu",
        freeze_encoder: bool = True,
    ):
        # Detection-only variant: build a focused init instead of running the
        # parent's multi-task constructor against the stock config tables.
        nn.Module.__init__(self)

        if config not in RFDETRSO_CONFIGS:
            raise ValueError(
                f"Invalid RF-DETR-SO size: {config}. "
                f"Must be one of {sorted(RFDETRSO_CONFIGS)}"
            )

        self.classification = False
        self.semantic = False
        self.segmentation = False
        self.pose = False
        self.obb = False
        self.num_keypoints = 0
        self.num_keypoints_per_class = []

        self.config_name = config
        self.config = RFDETRSO_CONFIGS[config]
        self.nb_classes = nb_classes

        self.args = _make_args(
            self.config,
            nb_classes=nb_classes,
            device=device,
            segmentation=False,
        )
        self.args.freeze_encoder = bool(freeze_encoder)

        self.resolution = self.config.resolution
        self.hidden_dim = self.config.hidden_dim
        self.num_queries = self.config.num_queries
        self.num_select = self.config.num_select
        self.patch_size = self.config.patch_size
        self.num_windows = self.config.num_windows

        self.model = build_model(self.args)
        joiner = self.model.backbone  # Joiner(Backbone, position_embedding)
        joiner[0] = BackboneSO(
            joiner[0],
            hidden_dim=self.hidden_dim,
            ssa_channels=SSA_BASE_CHANNELS[config],
        )
        self.postprocess = PostProcess(num_select=self.num_select)

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        """Load SO checkpoints as-is; remap stock RF-DETR checkpoints first."""
        tensors = (
            _unwrap_state_dict(state_dict) if isinstance(state_dict, dict) else {}
        )
        if tensors and not is_so_state_dict(tensors):
            remapped = remap_stock_state_for_so(
                tensors,
                ca_nheads=self.args.ca_nheads,
                dec_n_points=self.args.dec_n_points,
                num_levels=len(self.config.projector_scale),
            )
            wrapped: dict[str, Any] = {"model": remapped}
            # Preserve the metadata the parent loader consumes.
            for key in ("args", "num_keypoints"):
                if isinstance(state_dict, dict) and key in state_dict:
                    wrapped[key] = state_dict[key]
            state_dict = wrapped
        return super().load_state_dict(state_dict, strict=strict)


__all__ = [
    "RFDETRSO_CONFIGS",
    "SSA_BASE_CHANNELS",
    "LibreRFDETRSOModel",
    "is_so_state_dict",
    "remap_stock_state_for_so",
]
