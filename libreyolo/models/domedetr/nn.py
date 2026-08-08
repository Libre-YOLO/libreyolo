"""Top-level LibreDOMEDETR module and per-size configuration table.

Size configs are transcribed from Dome-DETR's shipped YAMLs
(``configs/dome/Dome-{S,M,L}-{AITOD,VisDrone}.yml`` plus the shared include
``configs/dome/include/dome_hgnetv2.yml``).

Two knobs differ by *dataset variant* rather than by size: the PAQI query
budget (``min_num_select`` / ``max_num_select``) and the class count. AI-TOD-V2
runs 300..1500 queries over 9 classes; VisDrone runs 250..500 over 12. There is
no COCO checkpoint for this family.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from ..dfine.backbone import HGNetv2
from .decoder import DomeTransformer
from .encoder import DomeHybridEncoder


SIZE_CONFIGS: Dict[str, Dict] = {
    "s": {
        "backbone": "B0",
        "use_lab": True,
        "freeze_norm": False,
        "enc_in_channels": (64, 256, 512, 1024),
        "enc_expansion": 0.5,
        "enc_depth_mult": 0.34,
    },
    "m": {
        "backbone": "B2",
        "use_lab": True,
        "freeze_norm": False,
        "enc_in_channels": (96, 384, 768, 1536),
        "enc_expansion": 1.0,
        "enc_depth_mult": 0.67,
    },
    "l": {
        "backbone": "B4",
        "use_lab": False,
        "freeze_norm": False,
        "enc_in_channels": (128, 512, 1024, 2048),
        "enc_expansion": 1.0,
        "enc_depth_mult": 1.0,
    },
}

# Decoder depth is set per (size, variant), not per size: L runs 4 layers on
# AI-TOD-V2 but 6 on VisDrone. Reading it off the size alone silently builds
# the wrong model for LibreDOMEDETRl-visdrone.
DEC_NUM_LAYERS: Dict[tuple, int] = {
    ("s", "aitod"): 3,
    ("m", "aitod"): 4,
    ("l", "aitod"): 4,
    ("s", "visdrone"): 3,
    ("m", "visdrone"): 4,
    ("l", "visdrone"): 6,
}

# PAQI query budget per dataset variant.
VARIANT_QUERY_BUDGET: Dict[str, Dict[str, int]] = {
    "aitod": {"min_num_select": 300, "max_num_select": 1500},
    "visdrone": {"min_num_select": 250, "max_num_select": 500},
}
DEFAULT_VARIANT = "aitod"

# All shipped Dome-DETR configs evaluate at 800x800.
EVAL_SPATIAL_SIZE = (800, 800)


class LibreDOMEDETRModel(nn.Module):
    """HGNetv2 backbone + density-guided hybrid encoder + PAQI decoder."""

    def __init__(
        self,
        config: str,
        nb_classes: int = 9,
        variant: str = DEFAULT_VARIANT,
        eval_spatial_size: tuple[int, int] | None = EVAL_SPATIAL_SIZE,
        activation: str = "relu",
    ):
        super().__init__()
        if config not in SIZE_CONFIGS:
            raise ValueError(f"Unknown Dome-DETR size: {config!r}")
        if variant not in VARIANT_QUERY_BUDGET:
            raise ValueError(
                f"Unknown Dome-DETR weight variant: {variant!r} "
                f"(expected one of {tuple(VARIANT_QUERY_BUDGET)})"
            )
        cfg = SIZE_CONFIGS[config]
        self.config = config
        self.variant = variant

        self.backbone = HGNetv2(
            name=cfg["backbone"],
            use_lab=cfg["use_lab"],
            return_idx=(0, 1, 2, 3),
            freeze_stem_only=True,
            freeze_at=-1,
            freeze_norm=cfg["freeze_norm"],
            pretrained=False,
        )
        self.encoder = DomeHybridEncoder(
            in_channels=cfg["enc_in_channels"],
            feat_strides=(4, 8, 16, 32),
            hidden_dim=256,
            dim_feedforward=1024,
            expansion=cfg["enc_expansion"],
            depth_mult=cfg["enc_depth_mult"],
            use_encoder_idx=(3,),
            num_feature_levels=4,
            eval_spatial_size=eval_spatial_size,
        )
        self.decoder = DomeTransformer(
            num_classes=nb_classes,
            hidden_dim=256,
            feat_channels=(256, 256, 256, 256),
            feat_strides=(4, 8, 16, 32),
            num_levels=4,
            num_points=(4, 4, 4, 4),
            num_layers=DEC_NUM_LAYERS[(config, variant)],
            dim_feedforward=1024,
            eval_spatial_size=eval_spatial_size,
            eval_idx=-1,
            reg_scale=4.0,
            activation=activation,
            **VARIANT_QUERY_BUDGET[variant],
        )

    def forward(self, x: torch.Tensor, targets: List[dict] | None = None):
        feats = self.backbone(x)
        encoder_out = self.encoder(feats, img_inputs=x, targets=targets)
        return self.decoder(encoder_out, targets=targets)

    def deploy(self):
        """Fuse BN into conv and prune the non-eval decoder layers."""
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy") and m is not self:
                m.convert_to_deploy()
        return self
